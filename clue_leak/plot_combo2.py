"""逐线索 mPL 消融分析 v2(不规则 SAM 掩码,74 簇几何)。
每图一面板:各子集 mean-mPL(主指标,sup 已知不稳)+ KL bits;打印冗余/协同 + Shapley。
用法：python -m clue_leak.plot_combo2
"""
import itertools
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
FWD = os.path.join(ROOT, "data", "forward_geocode_cache.json")
INDIR = os.path.join(BASE, "combo2_sam3_results")
OUTDIR = os.path.join(BASE, "figures")
BAR = "#2C6FB0"; BAR2 = "#E19A28"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})


def build_geometry(recs):
    cache = json.load(open(FWD, encoding="utf-8"))
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


def mpl_stats(prior, posterior, rep, clusters, dist):
    pr, po = merge_distribution(prior, rep), merge_distribution(posterior, rep)
    keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    llr = {k: math.log(po[k] / pr[k]) for k in keys}
    vals = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = dist(keys[a], keys[b])
            if d and d > 0:
                vals.append(abs(llr[keys[a]] - llr[keys[b]]) / d * 1000)
    return (max(vals), sum(vals) / len(vals)) if vals else (0.0, 0.0)


def kl_to_uniform_bits(belief, rep, clusters):
    b = merge_distribution(belief, rep); K = len(clusters); cs = set(clusters)
    return sum(v * math.log2(v * K) for k, v in b.items() if k in cs and v > 0)


def shapley(m, v):
    import math as _m
    phi = [0.0] * m
    for k in range(m):
        for size in range(m):
            for R in itertools.combinations([p for p in range(m) if p != k], size):
                w = _m.factorial(size) * _m.factorial(m - size - 1) / _m.factorial(m)
                phi[k] += w * (v[frozenset(R) | {k}] - v[frozenset(R)])
    return phi


def short(s, n=16):
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    recs = [json.load(open(os.path.join(INDIR, f), encoding="utf-8"))
            for f in sorted(os.listdir(INDIR)) if f.endswith(".json")]
    rep, clusters, dist = build_geometry(recs)
    print(f"{len(recs)} images, K={len(clusters)} clusters\n")

    fig, axes = plt.subplots(1, len(recs), figsize=(4.6 * len(recs), 5.0))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, recs):
        m = r["n_maskable"]
        by = {}
        names, means, kls = [], [], []
        for c in r["combos"]:
            S = tuple(c["subset"])
            sup, mean = mpl_stats(c["prior"], r["posterior"], rep, clusters, dist)
            by[S] = {"sup": sup, "mean": mean, "kl": c["kl_bits"]}
            names.append("+".join(str(k + 1) for k in S)); means.append(mean); kls.append(c["kl_bits"])
        x = np.arange(len(names))
        ax.bar(x - 0.2, means, 0.4, color=BAR, label="mean mPL", zorder=3)
        ax2 = ax.twinx(); ax2.bar(x + 0.2, kls, 0.4, color=BAR2, label="KL bits", zorder=3)
        ax2.spines.top.set_visible(False)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("mean mPL", color=BAR); ax2.set_ylabel("KL bits", color=BAR2)
        leg = "\n".join(f"{k+1}. {short(cm['cue'],22)} [{cm['risk']}]" for k, cm in enumerate(r["cue_meta"]))
        ax.set_title(f"{r['true_label'][:26]} (m={m})\n{leg}", fontsize=7.6, pad=5, loc="left")
        ax.grid(axis="y", color="#EEE", linewidth=0.5, zorder=0); ax.set_axisbelow(True)

        print(f"=== {r['true_label']} ({r['image_id'][:24]}, m={m}) ===")
        for k, cm in enumerate(r["cue_meta"]):
            print(f"  cue {k+1}: {cm['cue']}  [{cm['category']}/{cm['risk']}]")
        for S, st in by.items():
            print(f"  mask {'+'.join(str(k+1) for k in S):10s} mean={st['mean']:.3f} sup={st['sup']:6.2f} KL={st['kl']:.2f}b")
        for (i, j) in itertools.combinations(range(m), 2):
            if all(t in by for t in [(i, j), (i,), (j,)]):
                p, si, sj = (by[t]["mean"] for t in [(i, j), (i,), (j,)])
                tag = "synergy" if p > si + sj + 1e-6 else ("redundant" if p < max(si, sj) + 1e-9 else "additive")
                print(f"  pair {i+1}+{j+1}: {p:.3f} vs {si:.3f}+{sj:.3f} -> {tag}")
        full = frozenset(range(m))
        v = {full: kl_to_uniform_bits(r["posterior"], rep, clusters)}
        for c in r["combos"]:
            v[full - frozenset(c["subset"])] = kl_to_uniform_bits(c["prior"], rep, clusters)
        if all(frozenset(s) in v for s in itertools.chain.from_iterable(
                itertools.combinations(range(m), sz) for sz in range(m + 1))):
            phi = shapley(m, v)
            for k in range(m):
                print(f"  Shapley cue {k+1}: {phi[k]:+.3f} bits")
        print()
    fig.suptitle("Per-clue mPL ablation (irregular SAM masks) — prior=mask subset S, posterior=full  ·  "
                 f"K={len(clusters)}", fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "combo2_leakage_5imgs.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "combo2_leakage_5imgs.pdf"), bbox_inches="tight")
    print("saved figures/combo2_leakage_5imgs.png/.pdf")


if __name__ == "__main__":
    main()
