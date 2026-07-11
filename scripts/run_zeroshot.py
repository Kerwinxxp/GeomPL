"""论文 baseline：zero-shot 单次直接猜层级地名（无贝叶斯框架/搜索/验证循环）。

与 GeoBayes 用同一 build_client（同模型、同 smart_resize 预处理），保证公平对照。
每图输出 results_zeroshot/<id>.json = {"zero_shot": {...}, "name": ..., "image": ...}。
用法：python scripts/run_zeroshot.py [--limit N]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from run import build_client, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
OUT = os.path.join(ROOT, "results_zeroshot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    try:  # Windows 控制台默认 GBK，地名含非 ASCII（Zürich 等）会崩 print
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    config = load_config(os.path.join(ROOT, "config.yaml"))
    client = build_client(config)
    subset = [json.loads(l) for l in open(SUBSET, encoding="utf-8")]
    if args.limit:
        subset = subset[: args.limit]

    t0 = time.time()
    for i, it in enumerate(subset, 1):
        out_path = os.path.join(OUT, it["image_id"] + ".json")
        if os.path.exists(out_path):
            continue
        try:
            zs = client.zero_shot(Image.open(it["path"]))
        except Exception as e:
            print(f"[{i}] ERROR {it['image_id'][:36]}: {e}", flush=True)
            continue
        rec = {"zero_shot": zs, "name": zs.get("name", ""), "image": it["path"]}
        json.dump(rec, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[{i}/{len(subset)}] {it['image_id'][:30]} -> {zs.get('name','')[:50]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("zero-shot done:", len(os.listdir(OUT)))


if __name__ == "__main__":
    main()
