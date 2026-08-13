"""【实验性 · 可整体删除】用 MC-logprob 引出 + 带尾部保护的 mPL,重跑 5 图全子集消融。

两处修正,针对我们诊断出的三个病:
  引出:嘴报 0-1 分(糊+量化) → MC 单字母 logprob(尖锐、连续)      [治病A]
  mPL: 加①质量阈值(只算 max(pr,po)>=tau 的簇)②clamp 到 tau       [治"尾部噪声放大"]
       —— 否则尖分布下,1e-12 级的尾部会被 log 比值放大成假信号(NYC 那个 TV=0 却 mPL=4)
冗余(病B)本就无法靠这些修——它是图像性质,预期表现为"遮少数线索时 mPL≈0"。

候选集:原图嘴报最强 N(<=26,含真值),全子集共用。先验=遮子集S后 MC-logprob。
后验=原图 MC-logprob。mPL(先验_S→后验) 逐子集。
用法：python -m belief_elicit.run_logprob_ablation --images 261517384,847733166,370717727,181848051,311344213
"""
import argparse
import glob
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

from belief_elicit.elicit import score_independent
from belief_elicit.mc_logprob import score_mc_logprob
from belief_elicit.run_noise import build_geometry
from clue_leak.combo import nonempty_subsets
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from run import build_client, load_config

OUT = os.path.join(os.path.dirname(__file__), "logprob_ablation_results.json")
TAU = 1e-3        # 质量阈值 + clamp 下限:低于此的簇不参与,且 log 比值被限幅


def mpl_robust(prior, post, rep, clusters, dist, tau=TAU):
    """带尾部保护的 mPL:只对 max(pr,po)>=tau 的簇计;log 比值用 clamp(_,tau) 限幅。
    返回 (robust, raw) 便于对照 clamp 的效果。"""
    from geobayes.eval.candidates import merge_distribution
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)

    def _mpl(keys, clamp):
        llr = {}
        for k in keys:
            a = max(po.get(k, 0), tau) if clamp else po.get(k, 0)
            b = max(pr.get(k, 0), tau) if clamp else pr.get(k, 0)
            if a > 0 and b > 0:
                llr[k] = math.log(a / b)
        ks = list(llr)
        vals = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                d = dist(ks[i], ks[j])
                if d and d > 0:
                    vals.append(abs(llr[ks[i]] - llr[ks[j]]) / d * 1000)
        return sum(vals) / len(vals) if vals else 0.0

    robust_keys = [k for k in clusters if max(pr.get(k, 0), po.get(k, 0)) >= tau]
    raw_keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    return _mpl(robust_keys, clamp=True), _mpl(raw_keys, clamp=False)


def per_cue_masks(client, iid, subset):
    rec = json.load(open(os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json"),
                         encoding="utf-8"))
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = client.prepare(Image.open(p)); W, H = img.size
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
    return img, masks, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="261517384,847733166,370717727,181848051,311344213")
    ap.add_argument("--nlab", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=3)
    args = ap.parse_args()

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gallery)

    results = []
    for pref in args.images.split(","):
        iid = next(os.path.basename(f)[:-5]
                   for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
                   if os.path.basename(f).startswith(pref))
        img, masks, names = per_cue_masks(client, iid, subset)
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        m = len(masks)
        if m == 0:
            print(f"skip {pref} (0 maskable)"); continue
        W, H = img.size
        rank = score_independent(img, gallery)["raw"]
        cand = [tl] + sorted((l for l in gallery if l != tl), key=lambda l: -rank[l])[:args.nlab - 1]
        post = score_mc_logprob(img, cand, n_perm=args.n_perm)["prior"]
        print(f"\n=== {tl.split(',')[0]} ({pref}) | {m} cues | post p_true={post[tl]:.3f} ===", flush=True)

        combos = []
        for S in nonempty_subsets(m):
            u = np.zeros((H, W), bool)
            for k in S:
                u |= masks[k]
            prior = score_mc_logprob(mask_solid_from_masks(img, [u]), cand, n_perm=args.n_perm)["prior"]
            rob, raw = mpl_robust(prior, post, rep, clusters, dist)
            cov = float(u.sum() / (W * H))
            combos.append({"subset": list(S), "mpl": rob, "mpl_raw": raw, "cov": cov,
                           "p_true": prior[tl]})
            lbl = "+".join(str(k + 1) for k in S)
            print(f"  S={lbl:10s} cov={cov*100:4.0f}%  p_true={prior[tl]:.3f}  "
                  f"mPL(clamp)={rob:.4f}  mPL(raw)={raw:.4f}", flush=True)

        results.append({"image_id": iid, "place": tl.split(",")[0], "true_label": tl,
                        "n_maskable": m, "cue_names": names, "candidates": cand,
                        "posterior": post, "post_p_true": post[tl],
                        "per_cue_cov": [float(mk.sum() / (W * H)) for mk in masks],
                        "combos": combos})
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
