"""把某个 subsetN 的图重抓成高清(Flickr _b=1024;回退 _c=800→无后缀=500)。

- image_id 内嵌 server/photoid/secret → 拼 live.staticflickr 静态图 URL;
- 已在 data/sample_images/ 的(此前已抓过的)直接复用,不重下;
- 删除/失效图(占位图 min<200 或全部尺寸拿不到)跳过并计数,如实报告;
- 输出 data/<name>_hires.jsonl(路径指向高清)+ data/<name>_hires_meta.json。
用法：python scripts/fetch_hires50.py [--src data/subset100.jsonl]
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "sample_images")
UA = {"User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"}


def parts(img_id):
    m = re.match(r"(\d+)_([0-9a-f]+)_(\d+)_(.+)\.jpg", img_id)
    if not m:
        return None
    pid, secret, server, _ = m.groups()
    return pid, secret, server


def fetch_hires(img_id):
    """返回 (relpath, w, h, size_tag) 或 None。已存在则复用(不重下)。"""
    from PIL import Image
    path = os.path.join(IMG_DIR, img_id)
    if os.path.exists(path):
        w, h = Image.open(path).size
        return f"data/sample_images/{img_id}", w, h, "cached"
    pr = parts(img_id)
    if not pr:
        return None
    pid, secret, server = pr
    base = f"https://live.staticflickr.com/{server}/{pid}_{secret}"
    for tag in ("_b", "_c", ""):                       # 1024 → 800 → 500
        url = f"{base}{tag}.jpg"
        try:
            req = urllib.request.Request(url, headers=UA)
            data = urllib.request.urlopen(req, timeout=25).read()
            im = Image.open(io.BytesIO(data))
            if min(im.size) < 200:                     # 占位/失效图
                continue
            with open(path, "wb") as f:
                f.write(data)
            return f"data/sample_images/{img_id}", im.size[0], im.size[1], tag or "orig"
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "subset50.jsonl"))
    args = ap.parse_args()
    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    stem = os.path.basename(src)[:-6]                       # subsetN.jsonl -> subsetN
    OUT = os.path.join(ROOT, "data", f"{stem}_hires.jsonl")
    META = os.path.join(ROOT, "data", f"{stem}_hires_meta.json")
    os.makedirs(IMG_DIR, exist_ok=True)
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    kept, dead = [], []
    tags = {}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        got = fetch_hires(r["image_id"])
        if not got:
            dead.append(r["image_id"])
            print(f"[{i}/{len(rows)}] DEAD {r['image_id'][:34]}", flush=True)
            continue
        rel, w, h, tag = got
        tags[tag] = tags.get(tag, 0) + 1
        kept.append({**r, "path": rel})
        print(f"[{i}/{len(rows)}] {r['image_id'][:34]} {w}x{h} ({tag}) -> {r['gt_country']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"requested": len(rows), "kept": len(kept), "dead": dead,
            "size_tag_counts": tags,
            "note": "hi-res re-fetch of subset50 via Flickr _b(1024)/_c(800)/orig(500) fallback"}
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n== kept {len(kept)}/{len(rows)}, dead {len(dead)}, sizes {tags} ==")
    if dead:
        print("dead:", dead)


if __name__ == "__main__":
    main()
