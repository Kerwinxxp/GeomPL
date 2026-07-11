"""所有线索的 mPL 对比图(单条遮蔽,leave-one-out):
每条线索一根横条,值 = mean mPL(先验=只遮该线索,后验=原图,74 簇全 pair),
颜色 = 线索类别,标签 = 线索名(所在图)。按 mPL 降序。
用法：python -m clue_leak.plot_clue_mpl
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(__file__)
INDIR = os.path.join(BASE, "combo2_results")
OUTDIR = os.path.join(BASE, "figures")

CAT_COLOR = {
    "text/signage": "#C43D3D",
    "landmarks/buildings": "#2C6FB0",
    "architecture": "#8E24AA",
    "commercial/cultural": "#E19A28",
    "environment": "#43A047",
    "road/infrastructure": "#00ACC1",
    "vehicles/license plates": "#5E57C2",
    "other": "#8D6E63",
}
plt.rcParams.update({"font.family": "Microsoft YaHei",   # 含 CJK,日文线索名可显示
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})


def build_geometry(recs):
    cache = json.load(open(os.path.join(ROOT, "data", "forward_geocode_cache.json"), encoding="utf-8"))
    labels = sorted({k for r in recs for k in r["posterior"]})
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


def main():
    recs = [json.load(open(os.path.join(INDIR, f), encoding="utf-8"))
            for f in sorted(os.listdir(INDIR)) if f.endswith(".json")]
    rep, clusters, dist = build_geometry(recs)

    rows = []
    for r in recs:
        place = r["true_label"].split(",")[0]
        for c in r["combos"]:
            if len(c["subset"]) != 1:
                continue
            k = c["subset"][0]
            meta = r["cue_meta"][k]
            rows.append({"label": f"{meta['cue']}  ({place})",
                         "cat": meta.get("category") or "other",
                         "mpl": mean_mpl(c["prior"], r["posterior"], rep, clusters, dist)})
    rows.sort(key=lambda x: x["mpl"])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))
    ax.barh(y, [r["mpl"] for r in rows],
            color=[CAT_COLOR.get(r["cat"], "#8D6E63") for r in rows],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    for yi, r in zip(y, rows):
        ax.text(r["mpl"] + 0.002, yi, f"{r['mpl']:.3f}", va="center", fontsize=7.5, color="#444")
    ax.set_xlabel("mean mPL  (nats / 1000 km)  — masking ONLY this cue vs full image", fontsize=9.5)
    ax.set_title("Per-clue location leakage (mPL) — all clues from 5 pilot images, K=74 gallery",
                 fontsize=11, pad=10)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    seen = {r["cat"] for r in rows}
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CAT_COLOR if c in seen]
    ax.legend(handles, [c for c in CAT_COLOR if c in seen], frameon=False, fontsize=8,
              loc="lower right", title="cue category", title_fontsize=8.5)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "clue_mpl_ranking.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "clue_mpl_ranking.pdf"), bbox_inches="tight")
    print(f"saved figures/clue_mpl_ranking.png  ({len(rows)} clues)")


if __name__ == "__main__":
    main()
