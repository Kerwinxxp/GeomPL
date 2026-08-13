"""【实验性 · 可整体删除】对比三种信念引出方式在"遮蔽线索"下的行为。

核心问题:遮掉全部线索后,攻击者对**真值**的信念应该**下降**。
现有 independent 方案不降反升(干扰项塌缩伪影)。allocate / logprob 能修好吗?

对每种方法报:
  p_true            归一化后真值概率
  logodds(true|top-distractor)   真值 vs 最强干扰项的对数几率 —— 这才是 mPL 真正用的量
  集中度 1-H/Hmax   信念有多"尖"
  Δ                 遮后 − 原图(应为负!)
用法：python -m belief_elicit.run_compare [--image 171046893] [--nlab 20] [--methods independent,allocate,logprob]
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.elicit import METHODS
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from run import build_client, load_config

OUT = os.path.join(os.path.dirname(__file__), "compare_results.json")


def load_image_and_masks(client, iid):
    subset = {}
    import glob
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = client.prepare(Image.open(p))
    W, H = img.size
    rec = json.load(open(os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json"),
                         encoding="utf-8"))
    ms, names = [], []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        for i in c["instances"]:
            if (not i.get("degenerate")) and i.get("mask_rle"):
                m = rle_to_mask(i["mask_rle"])
                if m.shape == (H, W):
                    ms.append(m)
        names.append(c["cue"])
    return img, ms, names, subset[iid]


def stats(prior, tl, labels):
    n = len(labels)
    v = np.array([prior[l] for l in labels]); v = v / v.sum()
    ent = -(v * np.log2(v + 1e-12)).sum()
    conc = 1 - ent / math.log2(n)
    others = [l for l in labels if l != tl]
    top_d = max(others, key=lambda l: prior[l]) if others else None
    lo = math.log(prior[tl] / prior[top_d]) if top_d else float("nan")
    return {"p_true": prior[tl], "logodds_vs_topdistractor": lo,
            "top_distractor": top_d, "concentration": conc, "entropy_bits": float(ent)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="171046893", help="image_id 前缀")
    ap.add_argument("--nlab", type=int, default=20,
                    help="候选数:取**原图上最有竞争力**的前 N(必含真值),而非字母序前 N;"
                         "logprob 方法每候选 1 次调用")
    ap.add_argument("--methods", default="independent,allocate,logprob")
    args = ap.parse_args()

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))

    import glob
    iid = next(os.path.basename(f)[:-5]
               for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
               if os.path.basename(f).startswith(args.image))
    img, masks, names, meta = load_image_and_masks(client, iid)
    tl = labels_cache.get(f"{meta['lat']:.5f},{meta['lon']:.5f}")

    # 候选集:先在**全 77 gallery** 上跑一次独立打分,取最有竞争力的前 N(必含真值)。
    # 否则字母序前 N 全是 Alexandria/Anaheim 这类无竞争候选,题目太easy,测不出差异。
    from belief_elicit.elicit import score_independent
    rank = score_independent(img, gallery)["raw"]
    comp = sorted((l for l in gallery if l != tl), key=lambda l: -rank[l])[:args.nlab - 1]
    labels = [tl] + comp
    masked_img = mask_solid_from_masks(img, masks)
    print(f"image {iid[:16]} | true={tl} | {len(names)} cues masked | {len(labels)} candidates")
    print(f"cues: {names}")
    print(f"竞争者(按原图独立打分排序): {[(l[:24], rank[l]) for l in comp[:6]]}\n")

    results = {}
    for mname in args.methods.split(","):
        fn = METHODS[mname]
        kw = {}
        if mname == "logprob":
            kw["progress"] = lambda i, n, l, v: print(f"    [{i}/{n}] P(yes|{l[:26]}) = {v:.4f}",
                                                      flush=True)
        print(f"--- {mname} ---", flush=True)
        print("  full image:", flush=True)
        full = fn(img, labels, **kw)
        print("  cues masked:", flush=True)
        mask = fn(masked_img, labels, **kw)
        sf, sm = stats(full["prior"], tl, labels), stats(mask["prior"], tl, labels)
        results[mname] = {"full": {**sf, "raw": full["raw"]},
                          "masked": {**sm, "raw": mask["raw"]}}
        print(f"  p_true            {sf['p_true']:.4f} -> {sm['p_true']:.4f}  "
              f"(Δ {sm['p_true']-sf['p_true']:+.4f})")
        print(f"  logodds vs top-d  {sf['logodds_vs_topdistractor']:+.3f} -> "
              f"{sm['logodds_vs_topdistractor']:+.3f}  "
              f"(Δ {sm['logodds_vs_topdistractor']-sf['logodds_vs_topdistractor']:+.3f})")
        print(f"  concentration     {sf['concentration']*100:.1f}% -> {sm['concentration']*100:.1f}%")
        print(flush=True)

    json.dump({"image_id": iid, "true_label": tl, "cues": names,
               "labels": labels, "results": results},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT)
    print("\n判读:遮掉全部线索后,p_true 和 logodds 都应**下降**(Δ<0)。"
          "\n若某方法 Δ>0,说明该引出方式存在'干扰项塌缩'伪影,不适合做遮蔽实验。")


if __name__ == "__main__":
    main()
