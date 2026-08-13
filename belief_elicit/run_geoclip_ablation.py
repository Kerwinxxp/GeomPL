"""【实验性 · 可整体删除】用 GeoCLIP 连续 softmax 做信念计,重跑 5 图全子集消融。

本地、无 API、确定性 → 可自由跑全部子集。gallery = 完整 77 标签(有坐标的),
先验=遮子集S后 GeoCLIP softmax,后验=原图 GeoCLIP softmax,mPL 同主口径。
另出 raw 与 clamp(尾部保护)两版对照。
运行:cue_extract/.venv/Scripts/python.exe -m belief_elicit.run_geoclip_ablation
"""
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

from belief_elicit.geoclip_belief import get_model, score_labels
from clue_leak.combo import nonempty_subsets
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

OUT = os.path.join(os.path.dirname(__file__), "geoclip_ablation_results.json")
TAU = 1e-3
DEFAULT = ["261517384", "847733166", "370717727", "181848051", "311344213"]


def build_geometry(labels, cache):
    coords = {l: cache.get(l) for l in labels if cache.get(l)}
    rep = cluster_representatives(coords, 25.0)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mpl(prior, post, rep, clusters, dist, tau=0.0):
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)
    if tau > 0:
        keys = [k for k in clusters if max(pr.get(k, 0), po.get(k, 0)) >= tau]
        llr = {k: math.log(max(po.get(k, 0), tau) / max(pr.get(k, 0), tau)) for k in keys}
    else:
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(DEFAULT))
    args = ap.parse_args()

    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    cache = json.load(open(os.path.join(ROOT, "data", "forward_geocode_cache.json"), encoding="utf-8"))
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gallery, cache)
    model = get_model()

    results = []
    for pref in args.images.split(","):
        iid = next(os.path.basename(f)[:-5]
                   for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
                   if os.path.basename(f).startswith(pref))
        img, masks, names, (W, H) = per_cue_masks(iid, subset)
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        m = len(masks)
        if m == 0:
            print(f"skip {pref}"); continue
        post = score_labels(img, gallery, cache, model=model)
        print(f"\n=== {tl.split(',')[0]} ({pref}) | {m} cues | post p_true={post.get(tl,0):.3f} "
              f"(argmax={max(post,key=post.get).split(',')[0]}) ===", flush=True)
        combos = []
        for S in nonempty_subsets(m):
            u = np.zeros((H, W), bool)
            for k in S:
                u |= masks[k]
            prior = score_labels(mask_solid_from_masks(img, [u]), gallery, cache, model=model)
            raw = mpl(prior, post, rep, clusters, dist, tau=0.0)
            rob = mpl(prior, post, rep, clusters, dist, tau=TAU)
            cov = float(u.sum() / (W * H))
            combos.append({"subset": list(S), "mpl_raw": raw, "mpl": rob, "cov": cov,
                           "p_true": prior.get(tl, 0.0)})
            print(f"  S={'+'.join(str(k+1) for k in S):10s} cov={cov*100:4.0f}%  "
                  f"p_true={prior.get(tl,0):.3f}  mPL(raw)={raw:.4f}  mPL(clamp)={rob:.4f}", flush=True)
        results.append({"image_id": iid, "place": tl.split(",")[0], "true_label": tl,
                        "n_maskable": m, "cue_names": names,
                        "posterior": post, "post_p_true": post.get(tl, 0.0),
                        "post_argmax": max(post, key=post.get),
                        "per_cue_cov": [float(mk.sum() / (W * H)) for mk in masks],
                        "combos": combos})
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
