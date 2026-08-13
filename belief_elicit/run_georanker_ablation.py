"""【实验性 · 可整体删除】GeoRanker 信念计:5 图全子集消融(替换 GeoCLIP 第一波)。

与 run_geoclip_ablation.py 同构:gallery_v2(138)、raw mPL 同口径、同输出 schema
(多存 variant / rewards 方便复查)。prompt 变体由 --variant 指定(体检选定后传入)。
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_ablation --variant B
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.georanker_belief import format_negatives, score_labels
from belief_elicit.run_georanker_check import build_geometry, mpl
from clue_leak.combo import nonempty_subsets
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask

OUT = os.path.join(os.path.dirname(__file__), "georanker_ablation_results.json")
DEFAULT = ["261517384", "847733166", "370717727", "181848051", "311344213"]


def per_cue_masks(iid, subset):
    rec = json.load(open(os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json"),
                         encoding="utf-8"))
    W, H = rec["image_size"]
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = Image.open(p).resize((W, H)).convert("RGB")
    masks, names = [], []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        good = [i for i in c["instances"] if not i.get("degenerate") and i.get("mask_rle")]
        if not good:
            continue
        u = np.zeros((H, W), bool)
        for i in good:
            m = rle_to_mask(i["mask_rle"])
            if m.shape == (H, W):
                u |= m
        masks.append(u); names.append(c["cue"])
    return img, masks, names, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(DEFAULT))
    ap.add_argument("--variant", default="B", choices=["A", "B", "C"])
    args = ap.parse_args()

    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    label_gps = {g["label"]: g["gps"] for g in gv}
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gv)

    results = []
    t0 = time.time()
    for pref in args.images.split(","):
        iid = next(os.path.basename(f)[:-5]
                   for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
                   if os.path.basename(f).startswith(pref))
        img, masks, names, (W, H) = per_cue_masks(iid, subset)
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        if not masks or not tl:
            print(f"skip {pref}"); continue

        post, rewards = score_labels(img, gv, variant=args.variant, batch_size=4)
        negatives = None
        if args.variant == "C":              # 负例=原图倒5,固定后全条件复用
            srt = sorted(post, key=post.get)
            negatives = format_negatives([(label_gps[l][0], label_gps[l][1], l) for l in srt[:5]])
            post, rewards = score_labels(img, gv, variant="C", negatives=negatives, batch_size=4)
        arg = max(post, key=post.get)
        print(f"\n=== {tl.split(',')[0]} ({pref}) | {len(masks)} cues | "
              f"p_true={post.get(tl,0):.3f} argmax={arg.split(',')[0]} ({time.time()-t0:.0f}s) ===", flush=True)

        combos = []
        for S in nonempty_subsets(len(masks)):
            u = np.zeros((H, W), bool)
            for k in S:
                u |= masks[k]
            pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant=args.variant,
                                  negatives=negatives, batch_size=4)
            combos.append({"subset": list(S), "cov": float(u.sum() / (W * H)),
                           "p_true": pri.get(tl, 0.0), "mpl_raw": mpl(pri, post, rep, clusters, dist),
                           "mpl": mpl(pri, post, rep, clusters, dist)})
            c = combos[-1]
            print(f"  S={'+'.join(str(k+1) for k in S):10s} cov={c['cov']*100:4.0f}%  "
                  f"p_true={c['p_true']:.3f}  mPL={c['mpl_raw']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        results.append({"image_id": iid, "place": tl.split(",")[0], "true_label": tl,
                        "n_maskable": len(masks), "cue_names": names,
                        "variant": args.variant, "negatives": negatives,
                        "posterior": post, "post_p_true": post.get(tl, 0.0),
                        "post_argmax": arg,
                        "per_cue_cov": [float(m.sum() / (W * H)) for m in masks],
                        "combos": combos})
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
