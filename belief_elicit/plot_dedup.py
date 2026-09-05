"""Before/after figure for the geometric de-duplication.

(a) per-category median phi, before vs after (paired bars);
(b) SII histogram, before vs after;
(c) merged-group scatter: sum phi(members, before) vs phi(merged, after), with y = x;
(d) a text panel listing the merged groups.

Input:  shapley_v2_results.json / shapley_v3_results.json / cue_dedup_groups.json
Output: belief_elicit/figures/dedup_before_after.png
Run: python -m belief_elicit.plot_dedup
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
from belief_elicit.results import (DEDUP_GROUPS as GROUPS, FIGDIR, SHAPLEY_V2 as V2,
                                   SHAPLEY_V3 as V3, by_image, load_json, load_shapley)

apply_style()
OUT = os.path.join(FIGDIR, "dedup_before_after.png")


def cat_median_phi(rows):
    d = defaultdict(list)
    for r in rows:
        for c in r["cues"]:
            d[c["category"] or "unknown"].append(c["phi"])
    return {k: (float(np.median(v)), len(v)) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=V2)
    ap.add_argument("--after", default=V3)
    ap.add_argument("--groups", default=GROUPS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    B = load_shapley(a.before)
    A = load_shapley(a.after)
    G = load_json(a.groups)["images"]
    Bi = by_image(B)
    Ai = by_image(A)

    cb, ca = cat_median_phi(B), cat_median_phi(A)
    cats = sorted(set(cb) | set(ca), key=lambda k: -ca.get(k, (-9, 0))[0])

    sii_b = np.array([x for r in B for x in r["sii"].values()], float)
    sii_a = np.array([x for r in A for x in r["sii"].values()], float)
    # before-pairs whose two ends land in the same merged group (= a purely geometric artefact)
    dup_overlap = 0
    for iid, d in G.items():
        gmap = d["index_map"]
        for key, val in Bi[iid]["sii"].items():
            k, l = (int(x) for x in key.split(","))
            if gmap[k] == gmap[l] and val < -0.01:
                dup_overlap += 1

    pts = []
    for iid, d in G.items():
        for g in d["groups"]:
            if len(g["members"]) < 2:
                continue
            sb = sum(Bi[iid]["cues"][k]["phi"] for k in g["members"])
            gi = [i for i, c in enumerate(Ai[iid]["cues"])
                  if c["member_indices"] == g["members"]][0]
            pts.append({"place": d["place"], "sum_before": sb,
                        "after": Ai[iid]["cues"][gi]["phi"],
                        "n": len(g["members"]), "m": d["n_cues"], "M": d["n_merged"],
                        "iou": max(x[2] for x in g["pair_ious"]),
                        "names": g["names"]})
    pts.sort(key=lambda x: -x["sum_before"])

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.34, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a) per-category median phi ----
    y = np.arange(len(cats))[::-1]
    h = 0.38
    mb = [cb.get(k, (np.nan, 0))[0] for k in cats]
    ma = [ca.get(k, (np.nan, 0))[0] for k in cats]
    ax_a.barh(y + h / 2, mb, h, color=GRAY, zorder=3, label="before (244 cues)")
    ax_a.barh(y - h / 2, ma, h, color=BLUE, zorder=3, label="after de-dup (223 players)")
    for yi, k, b_, a_ in zip(y, cats, mb, ma):
        ax_a.text(b_ + 0.002, yi + h / 2, f"{b_:.3f}", va="center", fontsize=7.5, color="#555")
        ax_a.text(a_ + 0.002, yi - h / 2, f"{a_:.3f}", va="center", fontsize=7.5, color="#1565C0")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{k} ({cb.get(k,(0,0))[1]}→{ca.get(k,(0,0))[1]})" for k in cats],
                         fontsize=8.5)
    ax_a.set_xlabel("median Shapley φ (nats / 1000 km)")
    ax_a.set_xlim(0, max(max(mb), max(ma)) * 1.22)
    ax_a.set_title("(a) Per-category median φ, before vs after\n"
                   "(counts in parentheses: cues → merged players)", fontsize=10.5)
    ax_a.legend(fontsize=8, loc="lower right")
    ax_a.grid(axis="x", color="#EEE", zorder=0)
    ax_a.set_axisbelow(True)

    # ---- (b) SII histogram ----
    lo = min(sii_b.min(), sii_a.min())
    hi = max(sii_b.max(), sii_a.max())
    bins = np.linspace(lo, hi, 46)
    ax_b.hist(sii_b, bins=bins, color=GRAY, alpha=0.95, zorder=3,
              label=f"before  (n={len(sii_b)}, median {np.median(sii_b):+.3f})")
    ax_b.hist(sii_a, bins=bins, histtype="step", lw=1.9, color=RED, zorder=4,
              label=f"after   (n={len(sii_a)}, median {np.median(sii_a):+.3f})")
    ax_b.axvline(0, color="#666", lw=0.9, ls="--", zorder=5)
    ax_b.axvspan(lo, -0.01, color=RED, alpha=0.05, zorder=1)
    ax_b.axvspan(0.01, hi, color=GREEN, alpha=0.06, zorder=1)
    ax_b.text(0.02, 0.05, "overlap\n(SII < −0.01)", transform=ax_b.transAxes, fontsize=7.5,
              color="#B71C1C", va="bottom")
    ax_b.text(0.98, 0.05, "backup\n(SII > 0.01)", transform=ax_b.transAxes, fontsize=7.5,
              color="#2E7D32", va="bottom", ha="right")
    nb_ov, na_ov = int((sii_b < -0.01).sum()), int((sii_a < -0.01).sum())
    ax_b.set_xlabel("Shapley Interaction Index (pairwise)")
    ax_b.set_ylabel("pairs")
    ax_b.set_title(f"(b) Pairwise interactions: overlap pairs (SII < −0.01)\n"
                   f"{nb_ov} → {na_ov}; {dup_overlap} before-pairs were pure geometric "
                   f"duplicates", fontsize=10.5)
    ax_b.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax_b.set_ylim(0, max(np.histogram(sii_b, bins=bins)[0].max(),
                         np.histogram(sii_a, bins=bins)[0].max()) * 1.28)
    ax_b.grid(color="#EEE", zorder=0)
    ax_b.set_axisbelow(True)

    # ---- (c) merged groups: sum phi before vs phi after ----
    xs = np.array([p["sum_before"] for p in pts])
    ys = np.array([p["after"] for p in pts])
    lim = max(xs.max(), ys.max()) * 1.12
    ax_c.plot([0, lim], [0, lim], color="#999", ls="--", lw=1, zorder=2,
              label="y = x (perfect consolidation)")
    full = np.array([p["M"] == 1 for p in pts])
    ax_c.scatter(xs[full], ys[full], s=40 + 26 * np.array([p["n"] for p in pts])[full],
                 c=GREEN, alpha=0.85, edgecolors="white", linewidths=0.7, zorder=4,
                 label=f"group = whole image (Δ = 0 exactly, {int(full.sum())})")
    ax_c.scatter(xs[~full], ys[~full], s=40 + 26 * np.array([p["n"] for p in pts])[~full],
                 c=BLUE, alpha=0.85, edgecolors="white", linewidths=0.7, zorder=4,
                 label=f"group ⊂ image ({int((~full).sum())})")
    # labels: sort by sum phi and rotate through four offsets so dense areas stay readable
    quads = [(9, 9), (9, -15), (-9, 9), (-9, -15)]
    for rank, i in enumerate(np.argsort(xs)):
        p = pts[i]
        dx, dy = quads[rank % 4]
        ax_c.annotate(f"{p['place'][:12]} ({p['n']})", (xs[i], ys[i]), fontsize=7,
                      xytext=(dx, dy), textcoords="offset points", color="#555",
                      ha="left" if dx > 0 else "right",
                      arrowprops=dict(arrowstyle="-", lw=0.5, color="#BBB",
                                      shrinkA=0, shrinkB=3))
    ax_c.set_xlim(-0.015, lim); ax_c.set_ylim(-0.015, lim)
    ax_c.set_xlabel("Σ φ of the members, before de-dup")
    ax_c.set_ylabel("φ of the merged player, after")
    ax_c.set_title("(c) Credit consolidation for the 14 merged groups\n"
                   "point size ∝ group size", fontsize=10.5)
    ax_c.legend(fontsize=8, loc="upper left")
    ax_c.grid(color="#EEE", zorder=0)
    ax_c.set_axisbelow(True)

    # ---- (d) merged-group listing ----
    ax_d.axis("off")
    lines = ["Merged groups (mask IoU ≥ 0.90, union-find)", ""]
    for p in sorted(pts, key=lambda x: (-x["iou"], x["place"])):
        lines.append(f"{p['place'][:15]:15s}  m {p['m']}→{p['M']}   IoU {p['iou']:.3f}")
        for nm in p["names"]:
            lines.append(f"      • {nm[:52]}")
    ax_d.text(0.0, 1.0, "\n".join(lines), transform=ax_d.transAxes, va="top", ha="left",
              fontsize=6.4, family="DejaVu Sans", linespacing=1.32,
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F9FA", edgecolor=GRAY, lw=0.8))
    ax_d.set_title("(d) The 14 groups that were merged", fontsize=10.5, loc="left")

    fig.suptitle("Geometric de-duplication of the GPT-4o cue inventory "
                 "(244 cues → 223 players on 95 images; 14 images affected)",
                 fontsize=13, y=0.975)
    save(fig, a.out, dpi=130)


if __name__ == "__main__":
    main()
