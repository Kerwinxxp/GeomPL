"""Figure for the protection-set experiment -> figures/protection_set.png.

(a) primary risk R1 vs the budget k, one line per strategy (`exhaustive` is the lower
    envelope by construction);
(b) the distribution of R1 regret per strategy at k = 1 and k = 2;
(c) minimal-budget curves: what fraction of images a strategy pushes below eta with a
    budget of at most k, one panel per eta, with the feasible ceiling drawn in;
(d) Shapley phi against the signed single contribution, coloured by category — the
    lower-right quadrant is "large phi but misleading", where attribution tells the user
    to remove a cue whose removal helps the attacker.

Run: python -m belief_elicit.plot_protection_set
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from belief_elicit import plotstyle
from belief_elicit.plotstyle import BLUE, CYAN, GRAY, GREEN, INK, ORANGE, PURPLE, RED
from belief_elicit.protection_set import ETAS, STRATEGIES
from belief_elicit.results import FIGDIR, PROTECTION_SET, load_json

COLOR = {"exhaustive": INK, "greedy": BLUE, "shapley_topk": RED,
         "signed_topk": GREEN, "largest_area": ORANGE, "random": GRAY}
LABEL = {"exhaustive": "exhaustive (optimal)", "greedy": "greedy",
         "shapley_topk": "Shapley top-k", "signed_topk": "signed top-k",
         "largest_area": "largest area", "random": "random (20 draws)"}
CAT_COLORS = [RED, BLUE, GREEN, ORANGE, PURPLE, CYAN, "#6D4C41", "#546E7A", "#D81B60"]


def _ks(table):
    return sorted({r["k"] for r in table})


def panel_a(ax, res):
    table = res["summary"]["table"]
    ks = _ks(table)
    for name in STRATEGIES:
        rows = {r["k"]: r for r in table if r["strategy"] == name}
        y = [rows[k]["R1"]["mean"] for k in ks if k in rows]
        x = [k for k in ks if k in rows]
        ax.plot(x, y, "-o", color=COLOR[name], lw=2.2 if name == "exhaustive" else 1.4,
                ms=5 if name == "exhaustive" else 4, label=LABEL[name],
                zorder=3 if name == "exhaustive" else 2)
    full = np.mean([im["R1_full"] for im in res["images"]])
    ax.axhline(full, color=GRAY, ls=":", lw=1.2)
    ax.text(ks[0], full, f" untreated image ({full:.2f})", va="bottom", ha="left",
            fontsize=7, color=GRAY)
    ax.axhline(0, color=INK, lw=0.8, ls="--", alpha=.5)
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.22, ks[-1] + 0.22)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.15), max(hi, full + 0.5))
    ax.set_xlabel("budget k (players removed)")
    ax.set_ylabel("mean R1: signed evidence vs $\\pi$ (nats)")
    ax.set_title("(a) residual evidence after treating k cues", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", frameon=True, framealpha=.92,
              facecolor="white", edgecolor="none")
    n = {k: max(r["n_images"] for r in table if r["k"] == k) for k in ks}
    for k in ks:
        ax.annotate(f"n={n[k]}", (k, ax.get_ylim()[1]), xytext=(0, -10),
                    textcoords="offset points", ha="center", fontsize=7, color=GRAY)


def panel_b(ax, res):
    ks = [1, 2]
    order = [s for s in STRATEGIES]
    data, pos, cols = [], [], []
    for i, name in enumerate(order):
        for j, k in enumerate(ks):
            vals = [im["selections"][str(k)][name]["regret_R1"]
                    for im in res["images"] if str(k) in im["selections"]]
            data.append(vals)
            pos.append(i + (j - 0.5) * 0.34)
            cols.append(COLOR[name])
    bp = ax.boxplot(data, positions=pos, widths=0.28, orientation="horizontal",
                    patch_artist=True,
                    showfliers=True, medianprops=dict(color=INK, lw=1.1),
                    flierprops=dict(marker=".", ms=3, alpha=.5, mec=GRAY, mfc=GRAY))
    for patch, c, j in zip(bp["boxes"], cols, range(len(cols))):
        patch.set_facecolor(c)
        patch.set_alpha(0.85 if j % 2 else 0.45)
        patch.set_edgecolor(c)
    ax.axvline(0, color=INK, lw=0.9, ls="--", alpha=.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[s] for s in order], fontsize=8)
    ax.set_xlabel("R1 regret vs the exhaustive optimum (nats)")
    ax.set_title("(b) regret distribution at k = 1 (light) and k = 2 (dark)",
                 loc="left", fontsize=10)
    ax.invert_yaxis()
    for i, name in enumerate(order):
        for j, k in enumerate(ks):
            row = next(r for r in res["summary"]["table"]
                       if r["strategy"] == name and r["k"] == k)
            ax.annotate(f"{row['pct_optimal']:.0f}% opt", (ax.get_xlim()[1], i + (j - .5) * .34),
                        xytext=(-2, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=6.5, color=INK, alpha=.75)


def panel_c(axes, res):
    n_img = len(res["images"])
    for ax, eta in zip(axes, ETAS):
        rec = res["summary"]["min_k"][str(eta)]
        ceiling = 100.0 * (1 - rec["infeasible_frac"])
        ks = sorted({int(k) for im in res["images"] for k in im["selections"]})
        for name in STRATEGIES:
            y = []
            for k in ks:
                hit = 0
                for im in res["images"]:
                    sel = im["selections"]
                    if any(str(kk) in sel and sel[str(kk)][name]["R1"] <= eta
                           for kk in range(1, k + 1)):
                        hit += 1
                y.append(100.0 * hit / n_img)
            ax.plot(ks, y, "-o", color=COLOR[name],
                    lw=2.2 if name == "exhaustive" else 1.3,
                    ms=4.5 if name == "exhaustive" else 3.5,
                    zorder=3 if name == "exhaustive" else 2)
        ax.axhline(ceiling, color=GRAY, ls=":", lw=1.2)
        ax.set_xticks(ks)
        ax.set_ylim(0, max(30, ceiling * 1.55))
        ax.set_title(f"$\\eta$ = {eta} nats", fontsize=9)
        ax.set_xlabel("budget k")
        ax.annotate(f"feasible {ceiling:.0f}%", (ks[-1], ceiling), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=6.5, color=GRAY)
    axes[0].set_ylabel("% of images with R1 $\\leq\\eta$")
    axes[0].annotate("(c) minimal budget to bring R1 below $\\eta$",
                     xy=(0, 1.13), xycoords="axes fraction", ha="left", va="bottom",
                     fontsize=10)


def panel_d(ax, res):
    rows = [dict(p, image_id=im["image_id"])
            for im in res["images"] for p in im["per_player"]]
    cats = sorted({p["category"] for p in rows},
                  key=lambda c: -sum(1 for p in rows if p["category"] == c))
    cmap = {c: CAT_COLORS[i % len(CAT_COLORS)] for i, c in enumerate(cats)}
    top1 = {(t["image_id"], t["cue"]) for t in res["summary"]["misleading"]["shapley_top1"]["rows"]}
    for c in cats:
        pts = [p for p in rows if p["category"] == c]
        ax.scatter([p["phi"] for p in pts], [p["signed_contrib"] for p in pts],
                   s=22, color=cmap[c], alpha=.75, lw=0, label=f"{c} ({len(pts)})")
    hi = [p for p in rows if (p["image_id"], p["cue"]) in top1 and p["signed_contrib"] < 0]
    ax.scatter([p["phi"] for p in hi], [p["signed_contrib"] for p in hi],
               s=64, facecolors="none", edgecolors=INK, lw=1.1, zorder=4,
               label=f"Shapley top-1 & misleading ({len(hi)})")
    ax.axhline(0, color=INK, lw=0.9, ls="--", alpha=.6)
    ax.axvline(0, color=INK, lw=0.9, ls="--", alpha=.6)

    ml = res["summary"]["misleading"]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * 1.20)                 # headroom for the legend
    xr = ax.get_xlim()
    ax.text(xr[1], ax.get_ylim()[0], "large $\\varphi$, but MISLEADING\n"
            "(removal helps the attacker) ",
            ha="right", va="bottom", fontsize=7.5, color=RED, style="italic")
    ax.text(xr[1], hi, " large $\\varphi$, genuinely leaking", ha="right", va="top",
            fontsize=7.5, color=INK, alpha=.65, style="italic")
    ax.set_xlabel("Shapley $\\varphi$ (unsigned mPL attribution)")
    ax.set_ylabel("signed single contribution (nats)")
    ax.set_title(f"(d) attribution vs protection value  "
                 f"($\\rho_s$ = {ml['corr_phi_signed_spearman']:.2f}, "
                 f"{ml['frac_negative'] * 100:.0f}% misleading)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=2, handletextpad=.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=PROTECTION_SET)
    ap.add_argument("--out", default=os.path.join(FIGDIR, "protection_set.png"))
    args = ap.parse_args()

    res = load_json(args.results)
    if res is None:
        raise SystemExit(f"missing {args.results} — run belief_elicit.protection_set first")

    plotstyle.apply_style()
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.75,
                  left=0.055, right=0.985, top=0.93, bottom=0.075)
    panel_a(fig.add_subplot(gs[0, 0:3]), res)
    panel_b(fig.add_subplot(gs[0, 3:6]), res)
    axc = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for a in axc[1:]:
        a.sharey(axc[0])
        a.tick_params(labelleft=False)
    panel_c(axc, res)
    panel_d(fig.add_subplot(gs[1, 3:6]), res)

    pr = res["prior"]
    fig.suptitle(f"Protection-set selection on {len(res['images'])} images "
                 f"(reference $\\pi$: {pr['source']}, H = {pr['entropy']:.2f} nats)",
                 fontsize=11, y=0.985)
    plotstyle.save(fig, args.out, dpi=150, close=True)


if __name__ == "__main__":
    main()
