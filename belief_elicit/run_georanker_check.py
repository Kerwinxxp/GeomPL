"""【实验性 · 可整体删除】GeoRanker 仪器体检(多图 × 变体 A/B/C 可选)。

对每张图 × 每个 prompt 变体:
  原图 → 138 候选 reward → softmax → 真值 rank / p_true / top5;
  全部遮蔽子集 → p_true 变化 / raw mPL(几何与 GeoCLIP 口径完全一致);
体检三条(GEORANKER_PLAN.md §四):准确性 / 遮蔽响应 / mPL 形态。
变体 C 负例 = 变体 A 在【原图】上的倒数 5 名(单仪器;固定后对该图全部条件复用)。
注:选 C 而未选 A 时会自动先跑一次 A 原图以取负例。

注意:本 venv(.venv_gr)没有 geoclip;只 import 纯 Python 几何(geobayes.eval)。
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_check \
      [--images 261517384,...] [--variants A,B,C] [--tag okazaki]
"""
import glob
import json
import math
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
from clue_leak.combo import nonempty_subsets
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

OUTDIR = os.path.dirname(__file__)


def build_geometry(gv, merge_km=25.0):
    coords = {g["label"]: g["gps"] for g in gv if g["gps"]}
    rep = cluster_representatives(coords, merge_km)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mpl(prior, post, rep, clusters, dist):
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)
    keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    llr = {k: math.log(po[k] / pr[k]) for k in keys}
    ks = list(llr)
    vals = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = dist(ks[i], ks[j])
            if d and d > 0:
                vals.append(abs(llr[ks[i]] - llr[ks[j]]) / d * 1000)
    return sum(vals) / len(vals) if vals else 0.0


def load_case(iid, subset):
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="261517384")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--tag", default="check")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]

    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gv)
    label_gps = {g["label"]: g["gps"] for g in gv}

    all_out = []
    for pref in args.images.split(","):
        iid = next(os.path.basename(f)[:-5]
                   for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
                   if os.path.basename(f).startswith(pref))
        true = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        img, masks, names, (W, H) = load_case(iid, subset)
        if not masks or not true or true not in label_gps:
            print(f"skip {pref} (masks={len(masks)}, true={true})"); continue
        print(f"\n############ {true.split(',')[0]} ({pref}) | {len(masks)} cues: {names} ############",
              flush=True)

        results = {}
        negatives = None
        for variant in variants:
            if variant == "C" and negatives is None:   # C 需负例:先跑一次 A 原图取倒5
                pa, _ = score_labels(img, gv, variant="A", batch_size=4)
                bottom5 = sorted(pa, key=pa.get)[:5]
                negatives = format_negatives([(label_gps[l][0], label_gps[l][1], l)
                                              for l in bottom5])
            t0 = time.time()
            post, rewards = score_labels(img, gv, variant=variant, negatives=negatives,
                                         batch_size=4)
            srt = sorted(post, key=post.get, reverse=True)
            rank = srt.index(true) + 1
            if variant == "A" and negatives is None:
                bottom5 = srt[-5:]
                negatives = format_negatives([(label_gps[l][0], label_gps[l][1], l)
                                              for l in bottom5])
            print(f"\n=== 变体 {variant} | 原图 {time.time()-t0:.0f}s ===", flush=True)
            print(f"  真值 rank={rank}  p_true={post[true]:.4f}  "
                  f"reward范围=[{min(rewards):.2f},{max(rewards):.2f}]")
            print(f"  top5: {[(l.split(',')[0], round(post[l],3)) for l in srt[:5]]}", flush=True)

            combos = []
            for S in nonempty_subsets(len(masks)):
                u = np.zeros((H, W), bool)
                for k in S:
                    u |= masks[k]
                pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant=variant,
                                      negatives=negatives, batch_size=4)
                combos.append({"subset": list(S), "cov": float(u.sum() / (W * H)),
                               "p_true": pri[true], "mpl": mpl(pri, post, rep, clusters, dist)})
                c = combos[-1]
                print(f"  S={'+'.join(str(k+1) for k in S):10s} cov={c['cov']*100:4.0f}% "
                      f"p_true={c['p_true']:.4f}  mPL={c['mpl']:.4f}", flush=True)
            results[variant] = {"rank_true": rank, "p_true_full": post[true],
                                "posterior_top5": {l: post[l] for l in srt[:5]},
                                "rewards_minmax": [min(rewards), max(rewards)],
                                "combos": combos}
        all_out.append({"image_id": iid, "true": true, "cues": names,
                        "negatives_used": negatives, "results": results})
        out_f = os.path.join(OUTDIR, f"georanker_check_{args.tag}.json")
        json.dump(all_out, open(out_f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 体检判定 ----
    print("\n================ 体检汇总 ================")
    for rec_ in all_out:
        nm = len([c for c in rec_["results"][variants[0]]["combos"] if len(c["subset"]) == 1])
        for v, r in rec_["results"].items():
            allmask = max(r["combos"], key=lambda c: len(c["subset"]))
            covs = [c["cov"] for c in r["combos"]]; mpls = [c["mpl"] for c in r["combos"]]
            rho = float(np.corrcoef(covs, mpls)[0, 1]) if len(set(covs)) > 1 else float("nan")
            d = allmask["p_true"] - r["p_true_full"]
            print(f"{rec_['true'].split(',')[0][:12]:12s} 变体{v}: "
                  f"①rank={r['rank_true']:3d}(top3? {'Y' if r['rank_true']<=3 else 'N'})  "
                  f"②全遮 p_true {r['p_true_full']:.3f}→{allmask['p_true']:.3f} (Δ{d:+.3f})  "
                  f"③cov-mPL r={rho:+.2f}")
    print("done")


if __name__ == "__main__":
    main()
