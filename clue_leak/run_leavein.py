"""leave-one-IN:只留某条线索、遮掉其余(补集),测该线索**单独**的位置泄露。

对每条 maskable 线索 k:
  先验_k = 遮掉"所有其它 maskable 线索"的图(= 只留 k 可见)在 gallery 上打分;
  mPL(先验_k → 原图后验) = 只保留 k 时,信念离"看全图"还差多少
     —— 差得越小,说明 k 单独就能撑起接近全图的定位 → k 的绝对泄露越强。
另出对照口径 vs uniform:mPL(先验_k → uniform 无信息) 也存,便于"单独能定位多少"。

对比 leave-one-out(遮 k):若 in >> out,说明信号被冗余压在 out 里。
gallery / 几何 / 后验 复用 combo2_sam3;掩码来自 results_sam3。
用法：python -m clue_leak.run_leavein --ids id1,id2,...
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

from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km
from run import build_client, load_config

COMBODIR = os.path.join(os.path.dirname(__file__), "combo2_sam3_results")
CUEDIR = os.path.join(ROOT, "cue_extract", "results_sam3")
GALLERY = os.path.join(ROOT, "data", "gallery_labels.json")
FWD = os.path.join(ROOT, "data", "forward_geocode_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "leavein_results.json")


def build_geometry(all_labels):
    cache = json.load(open(FWD, encoding="utf-8"))
    coords = {l: cache.get(l) for l in all_labels if cache.get(l)}
    rep = cluster_representatives(coords, 25.0)
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
    vals = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = dist(keys[a], keys[b])
            if d and d > 0:
                vals.append(abs(llr[keys[a]] - llr[keys[b]]) / d * 1000)
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(GALLERY, encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    combos = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8"))
              for f in glob.glob(os.path.join(COMBODIR, "*.json"))}
    rep, clusters, dist = build_geometry({k for r in combos.values() for k in r["posterior"]})
    K = len(clusters)
    uniform = {c: 1.0 / K for c in clusters}

    out = []
    for iid in ids:
        r = combos[iid]
        post = r["posterior"]
        cue_rec = json.load(open(os.path.join(CUEDIR, iid + ".json"), encoding="utf-8"))
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = client.prepare(Image.open(p)); W, Hh = img.size
        mcues = [c for c in cue_rec["geo_privacy_cues"]
                 if c.get("maskable") and any((not i.get("degenerate")) and i.get("mask_rle") for i in c["instances"])]
        masks = []
        for c in mcues:
            u = np.zeros((Hh, W), bool)
            for i in c["instances"]:
                if (not i.get("degenerate")) and i.get("mask_rle"):
                    m = rle_to_mask(i["mask_rle"])
                    if m.shape == (Hh, W):
                        u |= m
            masks.append(u)
        # leave-one-out 复用 combo 里的单条 prior
        single = {tuple(c["subset"])[0]: c["prior"] for c in r["combos"] if len(c["subset"]) == 1}
        print(f"=== {r['true_label'].split(',')[0]} ({len(mcues)} cues) ===", flush=True)
        for k, c in enumerate(mcues):
            keep_only_k = ~masks[k]                                     # 遮掉"除 k 像素外的整张图"
            prior_in = client.score_candidates(mask_solid_from_masks(img, [keep_only_k]), gallery)["prior"]
            mpl_in_full = mpl(prior_in, post, rep, clusters, dist)      # 只留k → 离全图多远(越小=k越够用)
            mpl_in_uni = mpl(prior_in, uniform, rep, clusters, dist)    # 只留k → 离无信息多远(越大=k单独越能定位)
            mpl_out = mpl(single[k], post, rep, clusters, dist) if k in single else None
            area = round(masks[k].sum() / (W * Hh) * 100, 1)
            out.append({"image": iid, "place": r["true_label"].split(",")[0], "cue": c["cue"],
                        "area_pct": area, "mpl_out": mpl_out,
                        "mpl_in_vs_uniform": mpl_in_uni, "mpl_in_vs_full": mpl_in_full})
            print(f"  {c['cue'][:24]:24s} out(遮k)={mpl_out or 0:.3f}  "
                  f"in_vs_uniform(只留k离无信息)={mpl_in_uni:.3f}  in_vs_full(只留k离全图)={mpl_in_full:.3f}",
                  flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
