"""单个样本的线索 mPL 全展示:单条 + 所有组合子集。
左 = 原图 + 彩色掩码标注;右 = 每个子集一根横条(按子集大小分组),
值 = mean mPL(先验=遮该子集,后验=原图,74 簇全 pair)。
组合遮蔽 = 掩码**并集**(重叠像素只遮一次);柱旁标注并集覆盖率,重合大小可见:
若 cov(1+2) < cov(1)+cov(2),说明两线索区域有重合。
单条柱用线索色,组合柱用中性深灰。
用法：python -m clue_leak.plot_one_mpl <id前缀>   (默认 NYC 181848051)
"""
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

BASE = os.path.dirname(__file__)
# 目录可用环境变量覆盖(便于对比不同提取管线,如 SAM3):
#   CLUE_COMBO_DIR / CLUE_CUE_DIR / CLUE_FIG_DIR(相对 ROOT 或绝对)
_R = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
INDIR = _R(os.environ.get("CLUE_COMBO_DIR", os.path.join(BASE, "combo2_sam3_results")))
CUEDIR = _R(os.environ.get("CLUE_CUE_DIR", os.path.join(ROOT, "cue_extract", "results_sam3")))
OUTDIR = _R(os.environ.get("CLUE_FIG_DIR", os.path.join(BASE, "figures", "per_image_mpl_sam3")))
COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]
COMBO_COLOR = "#54626F"


def slug(true_label, pref):
    """'New York, United States' → 'newyork_181848051'(地名+id前缀,可读且唯一)。"""
    import re
    place = true_label.split(",")[0].strip().lower()
    place = re.sub(r"[^a-z0-9]+", "", place)[:14] or "img"
    return f"{place}_{pref}"
plt.rcParams.update({"font.family": "Microsoft YaHei", "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})


def build_geometry(rec):
    cache = json.load(open(os.path.join(ROOT, "data", "forward_geocode_cache.json"), encoding="utf-8"))
    labels = sorted(rec["posterior"])
    coords = {l: cache.get(l) for l in labels if cache.get(l)}
    rep = cluster_representatives(coords, 25.0)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mean_mpl(prior, post, rep, clusters, dist):
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


def _find_combo(arg):
    """arg 可为 image_id 前缀(如 370717727)或地名(如 cuba/newyork)。返回 combo json 路径。"""
    files = sorted(glob.glob(os.path.join(INDIR, "*.json")))
    hit = [f for f in files if os.path.basename(f).startswith(arg)]        # 先按 id 前缀
    if not hit:                                                            # 再按地名 slug
        key = re.sub(r"[^a-z0-9]+", "", arg.lower())
        hit = [f for f in files
               if key and key in re.sub(r"[^a-z0-9]+", "",
                                        json.load(open(f, encoding="utf-8"))["true_label"].lower())]
    if not hit:
        raise SystemExit(f"no combo result matches '{arg}' in {INDIR} "
                         f"(try an image-id prefix or place name: "
                         f"{[os.path.basename(f)[:12] for f in files]})")
    return hit[0]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "181848051"
    combo_f = _find_combo(arg)
    r = json.load(open(combo_f, encoding="utf-8"))
    iid = r["image_id"]
    cue_rec = json.load(open(os.path.join(CUEDIR, iid + ".json"), encoding="utf-8"))
    subset = {}                             # 合并所有 data/subset*.jsonl(样例集或全量)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset.setdefault(it["image_id"], it)
    p = subset[iid]["path"]
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = Image.open(p).resize(tuple(cue_rec["image_size"])).convert("RGB")
    rep, clusters, dist = build_geometry(r)
    m = r["n_maskable"]
    names = [cm["cue"] for cm in r["cue_meta"]]

    # 每条线索的掩码并集(与 run_combo2 的 maskable_cues 顺序一致)
    cue_union = {}
    k = 0
    for c in cue_rec["geo_privacy_cues"]:
        ms = [rle_to_mask(i["mask_rle"]) for i in c.get("instances", [])
              if (not i.get("degenerate")) and i.get("mask_rle")]
        if ms and c.get("maskable"):
            u = np.zeros_like(ms[0], dtype=bool)
            for mk in ms:
                u |= mk
            cue_union[k] = u
            k += 1
    total_px = img.width * img.height

    # 全部子集:mPL + 并集覆盖率(重叠只算一次 = 测量端同款 union)
    rows = []
    for c in r["combos"]:
        S = tuple(c["subset"])
        u = np.zeros((img.height, img.width), dtype=bool)
        for kk in S:
            if kk in cue_union and cue_union[kk].shape == u.shape:
                u |= cue_union[kk]
        rows.append({"S": S,
                     "label": "+".join(str(kk + 1) for kk in S),
                     "mpl": mean_mpl(c["prior"], r["posterior"], rep, clusters, dist),
                     "cov": u.sum() / total_px})
    rows.sort(key=lambda x: (len(x["S"]), x["S"]))

    # ---- 画 ----
    n = len(rows)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, max(5.2, 0.5 * n + 1.8)),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    overlay = np.asarray(img).astype(float).copy()
    for kk, u in cue_union.items():
        col = np.array(matplotlib.colors.to_rgb(COLORS[kk % len(COLORS)])) * 255
        overlay[u] = 0.45 * overlay[u] + 0.55 * col
    axL.imshow(overlay.astype(np.uint8))
    for kk, u in cue_union.items():
        ys, xs = np.nonzero(u)
        if len(xs):
            axL.text(xs.min() + 3, ys.min() + 3, str(kk + 1), color="white",
                     fontsize=12, fontweight="bold", va="top",
                     bbox=dict(boxstyle="round,pad=0.15", fc=COLORS[kk % len(COLORS)], ec="none"))
    axL.set_xticks([]); axL.set_yticks([])
    legend = "\n".join(f"{kk+1}. {nm}" for kk, nm in enumerate(names))
    axL.set_title(f"{r['true_label']}  —  {m} maskable cues\n{legend}", fontsize=9.5, loc="left")

    y = np.arange(n)[::-1]
    colors = [COLORS[row["S"][0] % len(COLORS)] if len(row["S"]) == 1 else COMBO_COLOR
              for row in rows]
    vals = [row["mpl"] for row in rows]
    axR.barh(y, vals, color=colors, edgecolor="white", zorder=3)
    axR.set_yticks(y)
    axR.set_yticklabels([row["label"] for row in rows], fontsize=9.5)
    vmax = max(vals) or 1e-9
    for yi, row in zip(y, rows):
        axR.text(row["mpl"] + vmax * 0.02, yi,
                 f"{row['mpl']:.3f}   (cov {row['cov']*100:.0f}%)",
                 va="center", fontsize=8, color="#333")
    # 子集大小分组分隔线
    sizes = [len(row["S"]) for row in rows]
    for i in range(1, n):
        if sizes[i] != sizes[i - 1]:
            axR.axhline(y[i] + 0.5, color="#CCCCCC", lw=0.8, ls=":")
    axR.set_xlabel("mean mPL  (nats / 1000 km)", fontsize=10)
    axR.set_title("mask subset S (mask = UNION of cue masks; overlap counted once)\n"
                  "singletons = cue color · combos = gray · cov = union coverage", fontsize=9)
    axR.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)
    axR.set_axisbelow(True)
    axR.set_xlim(0, vmax * 1.30)

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    stem = slug(r["true_label"], iid.split("_")[0]) + "_mpl"
    out = os.path.join(OUTDIR, stem + ".png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, f"({n} subsets)")


if __name__ == "__main__":
    main()
