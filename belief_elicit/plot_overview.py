"""Global overview: raw single-cue mPL -> Shapley phi for every cue, plus resolvability.

(a) per-cue v_single vs phi scatter (colour = resolvable, size = mask area);
(b) per-category bars comparing the two medians and the resolvable rate.

Run: python -m belief_elicit.plot_overview
     python -m belief_elicit.plot_overview --shapley belief_elicit/shapley_v3_results.json \
            --out belief_elicit/figures/georanker_overview_dedup.png   # de-dup arm
In the de-dup arm each cue carries `members` (the original pre-merge cue names); the control
maximum is the max over the members' controls and the area is their sum.
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from belief_elicit.plotstyle import BLUE, GRAY, GREEN, RED, apply_style, save
from belief_elicit.results import (CONTROLS as CTRL, FIGDIR, SHAPLEY_V2 as SHAP, by_image,
                                   load_controls, load_shapley)

apply_style()
OUT = os.path.join(FIGDIR, "georanker_overview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapley", default=SHAP, help="attribution JSON (default shapley_v2)")
    ap.add_argument("--controls", default=CTRL)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    S = load_shapley(a.shapley)
    C = by_image(load_controls(a.controls))
    # per-cue table: v_single, phi, area, resolvable (real > its own control max), category
    rows = []
    for r in S:
        cr = C.get(r["image_id"])
        cmap = {}
        if cr:
            for c in cr["cues"]:
                cm = [x["mpl"] for x in c["controls"]]
                cmap[c["cue"]] = (max(cm) if cm else None, c["area_frac"])
        for c in r["cues"]:
            names = c.get("members") or [c["cue"]]      # de-dup arm: merged player -> members
            got = [cmap[n] for n in names if n in cmap]
            cmax = max((g[0] for g in got if g[0] is not None), default=None)
            area = sum((g[1] for g in got if g[1] is not None), 0.0) or None
            rows.append(dict(cat=c["category"] or "unknown", v=c["v_single"],
                             phi=c["phi"], area=area or 0.05,
                             resolv=(cmax is not None and c["v_single"] > cmax)))
    v = np.array([x["v"] for x in rows]); phi = np.array([x["phi"] for x in rows])
    res = np.array([x["resolv"] for x in rows])
    area = np.array([x["area"] for x in rows])

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.05]})

    # (a) v_single -> phi scatter
    lim = max(v.max(), phi.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color="#999", ls="--", lw=1, label="φ = v (no correction)")
    ax[0].plot([0, lim], [0, lim / 2], color="#CCC", ls=":", lw=1, label="φ = v/2")
    for m, c, lab in [(res, GREEN, "resolvable (real>ctrl max)"),
                      (~res, RED, "not resolvable")]:
        ax[0].scatter(v[m], phi[m], s=20 + area[m] * 260, c=c, alpha=0.6,
                      edgecolors="white", linewidths=0.5,
                      label=f"{lab}  ({int(m.sum())})")
    ax[0].set_xlabel("single-cue mPL  v({k})  (baseline)")
    ax[0].set_ylabel("Shapley $\\varphi_k$  (corrected)")
    ax[0].set_title(f"(a) All {len(rows)} cues across 95 images\n"
                    "point size ∝ mask area; color = artifact-resolvable", fontsize=11)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim * 0.75)
    ax[0].grid(color="#EEE", zorder=0); ax[0].set_axisbelow(True)

    # (b) per category: median v / median phi / resolvable rate
    cat = defaultdict(lambda: dict(v=[], phi=[], res=[]))
    for x in rows:
        cat[x["cat"]]["v"].append(x["v"]); cat[x["cat"]]["phi"].append(x["phi"])
        cat[x["cat"]]["res"].append(x["resolv"])
    cats = sorted(cat, key=lambda k: -np.median(cat[k]["phi"]))
    cats = [c for c in cats if len(cat[c]["v"]) >= 3]
    y = np.arange(len(cats))[::-1]; h = 0.38
    vmed = [np.median(cat[c]["v"]) for c in cats]
    pmed = [np.median(cat[c]["phi"]) for c in cats]
    resr = [np.mean(cat[c]["res"]) for c in cats]
    ax[1].barh(y + h / 2, vmed, h, color=GRAY, zorder=3, label="single-cue mPL (median)")
    ax[1].barh(y - h / 2, pmed, h, color=BLUE, zorder=3, label="Shapley φ (median)")
    for yi, c, vm, pm, rr in zip(y, cats, vmed, pmed, resr):
        ax[1].text(vm + 0.002, yi + h / 2, f"{vm:.3f}", va="center", fontsize=8, color="#555")
        ax[1].text(pm + 0.002, yi - h / 2, f"{pm:.3f}", va="center", fontsize=8, color="#1565C0")
        ax[1].text(max(vm, pm) + 0.018, yi, f"resolvable {rr*100:.0f}%",
                   va="center", fontsize=8, color="#C62828")
    ax[1].set_yticks(y); ax[1].set_yticklabels([f"{c} (n={len(cat[c]['v'])})" for c in cats],
                                               fontsize=9)
    ax[1].set_xlabel("median mPL (nats/1000 km)")
    ax[1].set_title("(b) Per-category: baseline mPL → Shapley φ + resolvability",
                    fontsize=11)
    ax[1].legend(fontsize=8.5, loc="lower right")
    ax[1].set_xlim(0, max(vmed) * 1.55)
    ax[1].grid(axis="x", color="#EEE", zorder=0); ax[1].set_axisbelow(True)

    fig.suptitle("Per-cue location leakage across 95 images: raw mPL, Shapley correction, "
                 "and artifact-resolvability", fontsize=12.5, y=1.01)
    fig.tight_layout()
    save(fig, a.out, dpi=130)


if __name__ == "__main__":
    main()
