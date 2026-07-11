"""为 100 图子集补齐两个地理编码缓存(离线打分前跑一次):
  1. 反向地理编码 GT (lat,lon) → "City, Country" 标签  → data/gt_labels_cache.json
     (zoom=10,与 run_closedset 口径一致;run_leak 的 gallery 即这些唯一标签)
  2. 正向地理编码每个唯一标签 → [lat,lon]            → data/forward_geocode_cache.json
     (供 mPL 距离矩阵;缺失标签会被绘图几何静默丢弃,故必须补全)
Nominatim 1.1s 限速 + 磁盘缓存,重复运行只补新点。
用法：python -m clue_leak.prep_geo100 [--subset data/subset100.jsonl]
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from geobayes.eval.geocode import forward_geocode

LABELS = os.path.join(ROOT, "data", "gt_labels_cache.json")
FWD = os.path.join(ROOT, "data", "forward_geocode_cache.json")
UA = {"User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"}
_last = [0.0]


def reverse_label(lat, lon, cache):
    key = f"{lat:.5f},{lon:.5f}"
    if key in cache:
        return cache[key]
    w = 1.1 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()
    url = ("https://nominatim.openstreetmap.org/reverse?"
           + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2",
                                     "zoom": 10, "accept-language": "en"}))
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read())
        a = d.get("address", {})
        place = a.get("city") or a.get("town") or a.get("village") or a.get("county") or a.get("state")
        country = a.get("country")
        label = ", ".join(x for x in (place, country) if x) or None
    except Exception:
        label = None
    cache[key] = label
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(ROOT, "data", "subset100.jsonl"))
    args = ap.parse_args()
    subset = [json.loads(l) for l in open(args.subset, encoding="utf-8")]

    # 1) 反向 → 标签
    lab = json.load(open(LABELS, encoding="utf-8")) if os.path.exists(LABELS) else {}
    n0 = len(lab)
    labeled = 0
    for it in subset:
        l = reverse_label(it["lat"], it["lon"], lab)
        if l:
            labeled += 1
    json.dump(lab, open(LABELS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"reverse: {labeled}/{len(subset)} labeled, gt_labels_cache {n0}→{len(lab)}")

    # 2) 唯一标签 → 正向坐标
    labels = sorted({lab.get(f"{it['lat']:.5f},{it['lon']:.5f}") for it in subset} - {None})
    fwd = json.load(open(FWD, encoding="utf-8")) if os.path.exists(FWD) else {}
    m0 = len(fwd)
    missing = [l for l in labels if l not in fwd]
    for l in missing:
        forward_geocode(l, cache=fwd)      # 内部限速;失败缓存 None
    json.dump(fwd, open(FWD, "w", encoding="utf-8"), ensure_ascii=False)
    ok = sum(1 for l in labels if fwd.get(l))
    print(f"forward: gallery {len(labels)} labels, geocoded {ok}/{len(labels)}, "
          f"cache {m0}→{len(fwd)}, new fetched {len(missing)}")
    none_labels = [l for l in labels if not fwd.get(l)]
    if none_labels:
        print(f"WARNING {len(none_labels)} labels un-geocodable (dropped from geometry):", none_labels[:8])


if __name__ == "__main__":
    main()
