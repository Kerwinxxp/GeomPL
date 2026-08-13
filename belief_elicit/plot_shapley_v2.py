"""【实验性 · 可整体删除】全量 Shapley 汇总图:
(a) 逐类别 φ(矫正)vs 单条 mPL(未矫正)配对条形;(b) 184 对交互指数直方图。
运行(主环境):python -m belief_elicit.plot_shapley_full
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
DATA = os.path.join(os.path.dirname(__file__), "shapley_v2_results.json")
OUT = os.path.join(os.path.dirname(__file__), "figures", "georanker_shapley.png")
BLUE, GRAY, GREEN, ORANGE, RED = "#1E88E5", "#C4CDD5", "#43A047", "#FB8C00", "#E53935"


def main():
    R = json.load(open(DATA, encoding="utf-8"))  # 全部 95 张(主结果)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))

    # (a) φ vs v 配对条形
    cat_phi, cat_v = defaultdict(list), defaultdict(list)
    for r in R:
        for c in r["cues"]:
            cat_phi[c["category"] or "unknown"].append(c["phi"])
            cat_v[c["category"] or "unknown"].append(c["v_single"])
    cats = sorted(cat_phi, key=lambda k: -np.median(cat_phi[k]))
    y = np.arange(len(cats))[::-1]
    h = 0.38
    phi_m = [np.median(cat_phi[k]) for k in cats]
    v_m = [np.median(cat_v[k]) for k in cats]
    ax[0].barh(y + h / 2, phi_m, h, color=BLUE, zorder=3,
               label="Shapley $\\varphi$ (corrected share)")
    ax[0].barh(y - h / 2, v_m, h, color=GRAY, zorder=3,
               label="single-cue mPL (uncorrected)")
    for yi, k, pm, vm in zip(y, cats, phi_m, v_m):
        ax[0].text(pm + 0.002, yi + h / 2, f"{pm:.3f}", va="center", fontsize=8,
                   color="#1565C0")
        ax[0].text(vm + 0.002, yi - h / 2, f"{vm:.3f} (n={len(cat_phi[k])})",
                   va="center", fontsize=8, color="#666")
    ax[0].set_yticks(y); ax[0].set_yticklabels(cats, fontsize=9.5)
    ax[0].set_xlabel("median (nats/1000 km)")
    ax[0].set_title(f"(a) Per-category leakage: Shapley-corrected vs single-cue\n"
                    f"(all {len(R)} maskable images)", fontsize=11)
    ax[0].legend(fontsize=9, loc="lower right")
    ax[0].grid(axis="x", color="#EEE", zorder=0); ax[0].set_axisbelow(True)

    # (b) 交互直方图
    Is = np.array([x for r in R for x in r["sii"].values()])
    n_ov = (Is < -0.01).sum(); n_ad = (np.abs(Is) <= 0.01).sum(); n_bk = (Is > 0.01).sum()
    bins = np.linspace(min(-0.6, Is.min()), max(0.6, Is.max()), 41)
    ax[1].hist(Is[Is < -0.01], bins=bins, color=BLUE, zorder=3,
               label=f"overlap (I<-0.01): {n_ov}")
    ax[1].hist(Is[np.abs(Is) <= 0.01], bins=bins, color=GREEN, zorder=3,
               label=f"additive: {n_ad}")
    ax[1].hist(Is[Is > 0.01], bins=bins, color=ORANGE, zorder=3,
               label=f"backup (I>0.01): {n_bk}")
    ax[1].axvline(np.median(Is), color=RED, lw=1.4,
                  label=f"median {np.median(Is):+.3f}")
    ax[1].set_xlabel("Shapley interaction index  I^SII(k,l)")
    ax[1].set_ylabel("cue pairs")
    ax[1].set_title(f"(b) Interaction structure across {len(Is)} cue pairs (SII)",
                    fontsize=11)
    ax[1].legend(fontsize=9)
    ax[1].grid(axis="y", color="#EEE", zorder=0); ax[1].set_axisbelow(True)

    fig.suptitle("Shapley attribution on all multi-cue images (m = 2–5, full subset "
                 "lattice, efficiency verified 80/80)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=120)
    print("saved", OUT)


if __name__ == "__main__":
    main()
