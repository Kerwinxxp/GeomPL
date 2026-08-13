"""更新 gallery 到 200 张覆盖 + 升级为结构化记录(留 text/ref_image 空位)。

- 对 subset200 每张图:坐标→城市级地名(Nominatim reverse,zoom=10),缓存复用(已有 95 条不重查);
- 地名→代表坐标(forward geocode,与旧管线几何一致);
- 输出 data/gallery_v2.json = [{label, gps:[lat,lon], text:null, ref_image:null}, ...];
- 同步回写 gt_labels_cache / forward_geocode_cache。旧 gallery_labels.json 保留不动。
Nominatim 免费限速 1req/s。用法：python scripts/update_gallery.py
"""
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

UA = {"User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"}
GT = os.path.join(ROOT, "data", "gt_labels_cache.json")
FWD = os.path.join(ROOT, "data", "forward_geocode_cache.json")
SRC = os.path.join(ROOT, "data", "subset200_hires.jsonl")
OUT = os.path.join(ROOT, "data", "gallery_v2.json")
_last = [0.0]

# 城市级地名优先级(与现有标签形态对齐:"City, Country")
CITY_KEYS = ["city", "town", "village", "municipality", "county", "state_district",
             "state", "region"]


def _throttle():
    wait = 1.1 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def reverse_label(lat, lon):
    _throttle()
    url = (f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}"
           f"&format=jsonv2&zoom=10&accept-language=en")
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read())
        a = d.get("address", {})
        country = a.get("country")
        fine = next((a[k] for k in CITY_KEYS if a.get(k)), None)
        if not country:
            return None
        return f"{fine}, {country}" if fine else country
    except Exception as e:
        print("  reverse fail", lat, lon, e, flush=True)
        return None


def forward_coord(name):
    _throttle()
    url = ("https://nominatim.openstreetmap.org/search?"
           + urllib.parse.urlencode({"q": name, "format": "jsonv2", "limit": 1}))
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read())
        return [float(d[0]["lat"]), float(d[0]["lon"])] if d else None
    except Exception:
        return None


def main():
    gt = json.load(open(GT, encoding="utf-8")) if os.path.exists(GT) else {}
    fwd = json.load(open(FWD, encoding="utf-8")) if os.path.exists(FWD) else {}
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

    # 1) 每张图 → 城市级标签(缓存复用)
    labels = []
    new_rev = 0
    for i, it in enumerate(rows, 1):
        key = f"{it['lat']:.5f},{it['lon']:.5f}"
        if key not in gt or not gt[key]:
            gt[key] = reverse_label(it["lat"], it["lon"])
            new_rev += 1
            print(f"[rev {i}/{len(rows)}] {key} -> {gt[key]}", flush=True)
        if gt[key]:
            labels.append(gt[key])
    json.dump(gt, open(GT, "w", encoding="utf-8"), ensure_ascii=False)

    uniq = sorted(set(labels))
    print(f"\n唯一标签: {len(uniq)}(反查新增 {new_rev} 条)")

    # 2) 每个标签 → 代表坐标(forward,缓存复用)
    new_fwd = 0
    for lab in uniq:
        if lab not in fwd or not fwd[lab]:
            fwd[lab] = forward_coord(lab)
            new_fwd += 1
            print(f"  [fwd] {lab} -> {fwd[lab]}", flush=True)
    json.dump(fwd, open(FWD, "w", encoding="utf-8"), ensure_ascii=False)

    # 3) 结构化 gallery_v2(留 text/ref_image 空位)
    gallery = [{"label": lab, "gps": fwd.get(lab), "text": None, "ref_image": None}
               for lab in uniq if fwd.get(lab)]
    json.dump(gallery, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    dropped = [lab for lab in uniq if not fwd.get(lab)]
    print(f"\ngallery_v2: {len(gallery)} 个候选(带坐标)  | forward 新增 {new_fwd}")
    if dropped:
        print("无坐标丢弃:", dropped)
    print("saved", OUT)


if __name__ == "__main__":
    main()
