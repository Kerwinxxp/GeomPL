"""构建 Im2GPS3k 50 图子集（PLAN §Phase3 步骤 4 / map §6）。

- 固定 seed=42 打乱后顺序尝试，从 Flickr 静态服务器按需取图（整包 zip 免下载）；
- 失效图（404/太小）跳过并计数 → 可用性偏差写入 meta（plan §1.5 YFCC 失效图项）；
- GT 经纬度 → 国家：Nominatim reverse（1 req/s + 磁盘缓存 + UA）。
用法：python scripts/build_subset.py [--n 50]
"""
import argparse
import csv
import io
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "im2gps3k.csv")
IMG_DIR = os.path.join(ROOT, "data", "im2gps3k")
GEO_CACHE = os.path.join(ROOT, "data", "geocode_cache.json")
UA = {"User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"}

_last_nominatim = [0.0]


def flickr_url(img_id: str):
    m = re.match(r"(\d+)_([0-9a-f]+)_(\d+)_(.+)\.jpg", img_id)
    if not m:
        return None
    pid, secret, server, _ = m.groups()
    return f"https://live.staticflickr.com/{server}/{pid}_{secret}.jpg"


def fetch_image(img_id: str) -> str | None:
    from PIL import Image
    path = os.path.join(IMG_DIR, img_id)
    if os.path.exists(path):
        return path
    url = flickr_url(img_id)
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers=UA)
        data = urllib.request.urlopen(req, timeout=20).read()
        im = Image.open(io.BytesIO(data))
        if min(im.size) < 200:      # 太小的图无线索价值
            return None
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


def reverse_country(lat: float, lon: float, cache: dict):
    key = f"{lat:.5f},{lon:.5f}"
    if key in cache:
        return cache[key]
    wait = 1.1 - (time.time() - _last_nominatim[0])
    if wait > 0:
        time.sleep(wait)
    url = (f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}"
           f"&format=jsonv2&zoom=3&accept-language=en")
    req = urllib.request.Request(url, headers=UA)
    _last_nominatim[0] = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=20).read())
        result = {"country": d.get("address", {}).get("country"),
                  "code": d.get("address", {}).get("country_code")}
    except Exception as e:
        result = {"country": None, "code": None, "error": str(e)}
    cache[key] = result
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    OUT = os.path.join(ROOT, "data", f"subset{args.n}.jsonl")
    META = os.path.join(ROOT, "data", f"subset{args.n}_meta.json")

    os.makedirs(IMG_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    random.seed(42)
    random.shuffle(rows)

    cache = json.load(open(GEO_CACHE, encoding="utf-8")) if os.path.exists(GEO_CACHE) else {}
    kept, dead, no_country = [], 0, 0
    for row in rows:
        if len(kept) >= args.n:
            break
        path = fetch_image(row["IMG_ID"])
        if not path:
            dead += 1
            continue
        lat, lon = float(row["LAT"]), float(row["LON"])
        geo = reverse_country(lat, lon, cache)
        if not geo.get("country"):
            no_country += 1  # 公海/极地等无国家点：跳过（国家级 recall 无定义）
            continue
        kept.append({"image_id": row["IMG_ID"], "path": path.replace("\\", "/"),
                     "lat": lat, "lon": lon,
                     "gt_country": geo["country"], "gt_code": geo["code"]})
        print(f"[{len(kept)}/{args.n}] {row['IMG_ID'][:40]} -> {geo['country']}")

    json.dump(cache, open(GEO_CACHE, "w", encoding="utf-8"))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"n": len(kept), "dead_or_tiny_skipped": dead, "no_country_skipped": no_country,
            "seed": 42, "note": "availability bias: subset drawn from still-live Flickr images"}
    json.dump(meta, open(META, "w", encoding="utf-8"), indent=2)
    print("meta:", meta)


if __name__ == "__main__":
    main()
