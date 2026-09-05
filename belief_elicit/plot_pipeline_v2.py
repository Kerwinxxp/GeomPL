"""Explainer figures for the v2 pipeline (owner-facing, English).

figures/pipeline_v2_walkthrough.png : 6-step end-to-end walkthrough on ONE real image
                                      (New York, id prefix 158307292, m=4 cues).
figures/pipeline_v2_diagram.png     : schematic boxes-and-arrows of the v2 pipeline.

Run:  python -m belief_elicit.plot_pipeline_v2
"""
import glob
import itertools
import json
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.order2_shapley import order2_shapley
from belief_elicit.precompute_inpaint import cue_masks_of, load_subset_paths

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
IID_PREFIX = "158307292"

BLUE, RED, GRAY, GREEN, ORANGE = "#1E88E5", "#E53935", "#B0BEC5", "#43A047", "#FB8C00"
CUE_COLORS = [BLUE, RED, GREEN, ORANGE]
INK = "#212121"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#607D8B",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#546E7A",
    "ytick.color": "#546E7A",
})


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fhead(fig, x, y, step, title, caption, fs=13.5, cfs=9.2):
    """Step header + caption placed in FIGURE coordinates (full layout control)."""
    fig.text(x, y, f"Step {step} — {title}", fontsize=fs, fontweight="bold",
             ha="left", va="bottom", color=INK)
    fig.text(x, y - 0.010, caption, fontsize=cfs, ha="left", va="top",
             color="#455A64", linespacing=1.45)


# ------------------------------------------------------------------ data ----

def load_case():
    sweep = json.load(open(os.path.join(HERE, "georanker_sweep_results.json"),
                           encoding="utf-8"))
    rec = [r for r in sweep if r["image_id"].startswith(IID_PREFIX)][0]
    iid = rec["image_id"]

    inp = {r["image_id"]: r for r in json.load(
        open(os.path.join(HERE, "georanker_inpaint_results.json"), encoding="utf-8"))}[iid]
    ctl = {r["image_id"]: r for r in json.load(
        open(os.path.join(HERE, "georanker_inpaint_control_results.json"),
             encoding="utf-8"))}[iid]
    ded = json.load(open(os.path.join(HERE, "cue_dedup_groups.json"),
                         encoding="utf-8"))
    cache = os.path.join(HERE, "inpaint_cache", iid)
    manifest = json.load(open(os.path.join(cache, "manifest.json"), encoding="utf-8"))

    cues, cats, masks, (W, H) = cue_masks_of(
        os.path.join(ROOT, "cue_extract", "results_sam3"), iid)
    p = load_subset_paths()[iid]["path"]
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = Image.open(p).resize((W, H)).convert("RGB")

    V = {v["spec"]: v for v in inp["variants"]}
    m = len(cues)
    singles = [V[f"s{k}"]["mpl"] for k in range(m)]
    pairs = {(k, l): V[f"p{k}-{l}"]["mpl"] for k, l in itertools.combinations(range(m), 2)}
    o2 = order2_shapley(singles, pairs, V["all"]["mpl"])

    ctrl_mpl = {v["spec"]: v["mpl"] for v in ctl["variants"]}
    c_img = float(np.mean(list(ctrl_mpl.values())))
    ctrl_by_cue = {k: [ctrl_mpl[s] for s in ctrl_mpl if s.startswith(f"c{k}-")]
                   for k in range(m)}

    return dict(iid=iid, sweep=rec, V=V, o2=o2, singles=singles, pairs=pairs,
                cues=cues, cats=cats, masks=masks, size=(W, H), img=img, m=m,
                cache=cache, manifest=manifest, c_img=c_img, ctrl_by_cue=ctrl_by_cue,
                dedup=ded["images"][iid], dedup_meta=ded["meta"],
                true_label=rec["true_label"])


# ------------------------------------------------------- panel 1: cues ------

SHORT = ["1. Statue", "2. Flag", "3. Building", "4. Trees"]


def panel_cues(ax, D):
    ax.imshow(D["img"])
    H, W = D["size"][1], D["size"][0]
    overlay = np.zeros((H, W, 4))
    for k, mk in enumerate(D["masks"]):
        r, g, b, _ = matplotlib.colors.to_rgba(CUE_COLORS[k])
        overlay[mk] = (r, g, b, 0.42)
        ax.contour(mk.astype(float), levels=[0.5], colors=[CUE_COLORS[k]], linewidths=1.6)
    ax.imshow(overlay)
    for k, mk in enumerate(D["masks"]):
        ys, xs = np.nonzero(mk)
        ax.text(xs.mean(), ys.mean(), str(k + 1), fontsize=12.5, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.30", fc=CUE_COLORS[k], ec="white", lw=1.8))
    bare(ax)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def cue_legend(fig, D, x, y):
    for k, c in enumerate(D["cues"]):
        fig.text(x, y - k * 0.0165, "■", fontsize=10, color=CUE_COLORS[k],
                 va="top", ha="left")
        fig.text(x + 0.010, y - k * 0.0165,
                 f"{k+1}. {c} · {D['cats'][k]} · {D['masks'][k].mean()*100:.1f}%",
                 fontsize=8.6, va="top", ha="left", color=INK)
    d, meta = D["dedup"], D["dedup_meta"]
    fig.text(x, y - 4 * 0.0165 - 0.007,
             f"Geometric de-dup (dedup_cues.py, IoU ≥ {meta['iou_threshold']}): "
             f"{d['n_cues']} → {d['n_merged']} players —\n"
             f"0 merges here (largest off-diagonal mask IoU = "
             f"{d['max_offdiag_iou']:.4f});\n"
             f"14 of the 95 images in the set do get merged.",
             fontsize=8.4, va="top", ha="left", color="#455A64", style="italic",
             linespacing=1.5)


# -------------------------------------------------- panel 2: removal op -----

def panel_removal(ax_g, ax_i, D, k=2):
    gray = mask_solid_from_masks(D["img"], [D["masks"][k]])
    lama = Image.open(os.path.join(D["cache"], f"s{k}.png")).convert("RGB")
    for ax, im, t, col in ((ax_g, gray, "gray fill  (v1)", "#78909C"),
                           (ax_i, lama, "LaMa inpaint  (v2)", GREEN)):
        ax.imshow(im); bare(ax)
        ax.text(0.5, 0.985, t, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.30", fc=col, ec="none", alpha=0.92))
        for sname in ("left", "right", "top", "bottom"):
            ax.spines[sname].set_visible(True)
            ax.spines[sname].set_color(col); ax.spines[sname].set_linewidth(2.0)


# --------------------------------------------------- panel 3: beliefs -------

def short(lbl):
    a = lbl.split(",")[0]
    return a if len(a) <= 17 else a[:16] + "…"


def fmt_p(v):
    return f"{v:.3f}" if v >= 0.001 else f"{v:.4f}"


def panel_belief(axes, D, k=2, topn=8):
    post = D["sweep"]["posterior"]
    order = [l for l, _ in sorted(post.items(), key=lambda kv: -kv[1])[:topn]]
    true = D["true_label"]
    dists = [("(i) full image\nadversary posterior", post, BLUE),
             (f"(ii) cue {k+1} inpainted\n(variant s{k})", D["V"][f"s{k}"]["prior"], ORANGE),
             ("(iii) all 4 cues inpainted\n(variant all)", D["V"]["all"]["prior"], RED)]
    y = np.arange(topn)[::-1]
    for ax, (t, dist, col) in zip(axes, dists):
        vals = [dist[l] for l in order]
        cols = [col if l == true else GRAY for l in order]
        ax.barh(y, vals, color=cols, height=0.70)
        ax.set_xlim(0, 0.50); ax.set_ylim(-0.75, topn - 0.25)
        ax.set_title(t, fontsize=9.4, color=INK, pad=7)
        ax.set_xlabel("P(label | image)", fontsize=8.4, labelpad=2)
        ax.tick_params(labelsize=8)
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_yticks(y)
        ax.set_yticklabels([short(l) for l in order] if ax is axes[0] else [], fontsize=8.4)
        for yy, v, l in zip(y, vals, order):
            ax.text(v + 0.014, yy, fmt_p(v), va="center", fontsize=7.6,
                    color=INK if l == true else "#78909C",
                    fontweight="bold" if l == true else "normal")
    top3 = sorted(D["V"]["all"]["prior"].items(), key=lambda kv: -kv[1])[:3]
    axes[2].text(0.97, 0.845,
                 "every US label is now\n≤ 0.0003 — the belief has\n"
                 "left the country. New top-3:\n" +
                 "\n".join(f"{a}   {b:.3f}" for a, b in top3),
                 transform=axes[2].transAxes, fontsize=8.0, ha="right", va="top",
                 color=RED, linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.40", fc="#FFEBEE", ec=RED, lw=0.9))


# --------------------------------------------------- panel 4: lattice -------

def panel_lattice(ax, D):
    m = D["m"]
    V = {(): 0.0}
    for k in range(m):
        V[(k,)] = D["V"][f"s{k}"]["mpl"]
    for k, l in itertools.combinations(range(m), 2):
        V[(k, l)] = D["V"][f"p{k}-{l}"]["mpl"]
    V[tuple(range(m))] = D["V"]["all"]["mpl"]
    vmax = max(V.values())

    levels = {0: [()], 1: [(k,) for k in range(m)],
              2: list(itertools.combinations(range(m), 2)), 3: [tuple(range(m))]}
    pos = {}
    for lv, nodes in levels.items():
        n = len(nodes)
        xs = np.linspace(0.5 / n, 1 - 0.5 / n, n) if n > 1 else [0.5]
        for x, nd in zip(xs, nodes):
            pos[nd] = (x, lv)
    for lv in (0, 1, 2):
        for a in levels[lv]:
            for b in levels[lv + 1]:
                if set(a) < set(b):
                    ax.plot([pos[a][0], pos[b][0]], [pos[a][1] + 0.22, pos[b][1] - 0.22],
                            color="#CFD8DC", lw=0.9, zorder=1)
    for nd, (x, lv) in pos.items():
        v = V[nd]
        name = "∅" if not nd else "{" + ",".join(str(i + 1) for i in nd) + "}"
        ax.text(x, lv, f"{name}\n{v:.3f}", ha="center", va="center", fontsize=8.2,
                zorder=3, color=INK, fontweight="bold" if lv == 3 else "normal",
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32",
                          fc=matplotlib.colors.to_rgba(BLUE, 0.10 + 0.55 * v / vmax),
                          ec=BLUE, lw=1.0))
    for lv, lab in ((0, "∅"), (1, "singles  s$k$"), (2, "pairs  p$k$-$l$"),
                    (3, "all = v(N)")):
        ax.text(-0.03, lv, lab, ha="right", va="center", fontsize=8.6, color="#546E7A")
    ax.set_xlim(-0.26, 1.06); ax.set_ylim(-0.60, 3.62)
    bare(ax)


# ------------------------------------------------ panel 5: attribution ------

def panel_attrib(ax, ax_i, D):
    m, o2 = D["m"], D["o2"]
    x = np.arange(m); w = 0.38
    ax.bar(x - w / 2, D["singles"], w, color=GRAY, label="naive single-cue  v({k})")
    ax.bar(x + w / 2, o2["phi"], w, color=BLUE,
           label="order-2 anchored Shapley  $\\varphi_k$")
    for xx, a, b in zip(x, D["singles"], o2["phi"]):
        ax.text(xx - w / 2, a + 0.013, f"{a:.3f}", ha="center", fontsize=7.8, color="#546E7A")
        ax.text(xx + w / 2, b + 0.013, f"{b:.3f}", ha="center", fontsize=7.8, color=BLUE,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8.6)
    ax.set_ylabel("mPL", fontsize=9)
    ax.set_ylim(0, max(max(D["singles"]), max(o2["phi"])) * 1.42)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left", handlelength=1.4,
              borderaxespad=0.2)
    ax.tick_params(labelsize=8)

    inter = o2["interactions"]
    keys = sorted(inter, key=lambda kl: inter[kl])
    yy = np.arange(len(keys))[::-1]
    vals = [inter[k] for k in keys]
    ax_i.barh(yy, vals, color=[RED if v < 0 else GREEN for v in vals], height=0.64)
    ax_i.set_yticks(yy)
    ax_i.set_yticklabels(["{%d,%d}" % (a + 1, b + 1) for a, b in keys], fontsize=7.8)
    ax_i.axvline(0, color="#607D8B", lw=0.8)
    ax_i.set_xlim(-0.40, 0.26)
    ax_i.set_xticks([-0.3, 0.0, 0.2])
    ax_i.tick_params(labelsize=7.6)
    ax_i.set_title("interaction  $d_{kl}$", fontsize=9, pad=7)
    for y_, v in zip(yy, vals):
        ax_i.text(v + (0.010 if v >= 0 else -0.010), y_, f"{v:+.3f}", va="center",
                  ha="left" if v >= 0 else "right", fontsize=7.2,
                  color=GREEN if v >= 0 else RED)
    ax_i.text(0.5, -0.115, "← overlap        backup →", transform=ax_i.transAxes,
              ha="center", va="top", fontsize=7.8, color="#455A64")


# ------------------------------------------------------ panel 6: null -------

def panel_null(ax, D):
    m, o2 = D["m"], D["o2"]
    thr = D["c_img"] / m
    x = np.arange(m)
    sens = [p > thr for p in o2["phi"]]
    ax.bar(x, o2["phi"], 0.55, color=[GREEN if s else GRAY for s in sens])
    for xx, p, s in zip(x, o2["phi"], sens):
        ax.text(xx, p + 0.015, f"{p:.3f}", ha="center", fontsize=8.2, fontweight="bold",
                color=GREEN if s else "#78909C")
    for k in range(m):
        cs = D["ctrl_by_cue"][k]
        ax.plot([k] * len(cs), cs, marker="D", ls="none", ms=4.5, mfc="white",
                mec="#546E7A", mew=1.1, zorder=4,
                label="this cue's own equal-area controls" if k == 0 else None)
    ax.axhline(thr, color=RED, ls="--", lw=1.7, zorder=5,
               label=f"inpaint null  $c_{{img}}/m$ = {thr:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8.6)
    ax.set_ylabel("mPL", fontsize=9)
    ax.set_ylim(0, max(o2["phi"]) * 1.52)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", handlelength=1.5,
              borderaxespad=0.2)


# ------------------------------------------------------------- figure 1 -----

def figure_walkthrough(D):
    fig = plt.figure(figsize=(18, 12), dpi=130)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.966, "GeoBayes v2 pipeline, end to end on one image  —  "
             f"{D['true_label']}   (id {IID_PREFIX}…,  m = 4 cues)",
             ha="center", va="bottom", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.5, 0.950,
             "mPL = mean over candidate-label pairs of |Δ log-likelihood-ratio| per "
             "1000 km between the belief before and after removal  —  higher = the "
             "removed pixels carried more geo-locating evidence.",
             ha="center", va="top", fontsize=9.8, color="#455A64")

    # ---------------- top row ----------------
    TOP_Y, TOP_H = 0.578, 0.286
    fhead(fig, 0.030, 0.907, 1, "Cues",
          "GPT-4o names them; SAM 3 segments each one\n"
          "(union of that cue's non-degenerate instances).")
    ax1 = fig.add_axes([0.030, TOP_Y, 0.150, TOP_H])
    panel_cues(ax1, D)
    cue_legend(fig, D, 0.028, TOP_Y - 0.010)

    fhead(fig, 0.222, 0.907, 2, "Removal operator",
          "Cue 3 (historic building) removed two ways. v2 removes by inpainting so\n"
          "the adversary cannot key on a conspicuous gray blob.")
    axg = fig.add_axes([0.222, TOP_Y, 0.122, TOP_H])
    axi = fig.add_axes([0.352, TOP_Y, 0.122, TOP_H])
    panel_removal(axg, axi, D, k=2)
    fig.text(0.222, TOP_Y - 0.012,
             "The same operator is applied to every scored subset S of cues:\n"
             "singles s$k$, pairs p$k$-$l$, and all.",
             fontsize=8.6, va="top", ha="left", color="#455A64", linespacing=1.5)

    fhead(fig, 0.560, 0.907, 3, "Adversary belief",
          "GeoRanker scores 138 candidate labels; labels within 2 km are merged into one\n"
          "alias. Same 8 labels, same order, in all three panels; coloured bar = true label.")
    bw = (0.985 - 0.560 - 2 * 0.012) / 3
    axb = [fig.add_axes([0.560 + i * (bw + 0.012), TOP_Y - 0.015, bw, TOP_H - 0.030])
           for i in range(3)]
    panel_belief(axb, D, k=2)

    # ---------------- bottom row ----------------
    BOT_Y, BOT_H = 0.190, 0.215
    fhead(fig, 0.036, 0.438, 4, "Leakage lattice  v(S)",
          "v(S) = mPL after inpainting exactly the cues in S; v(∅) = 0.")
    axl = fig.add_axes([0.036, BOT_Y, 0.250, BOT_H])
    panel_lattice(axl, D)
    fig.text(0.036, BOT_Y - 0.048,
             "The inpaint arm scores 4 + 6 + 1 = 11 images (singles, pairs, all);\n"
             "the gray-fill arm has the exact 2ᵐ = 16-subset lattice, which is what\n"
             "order2_shapley.py is validated against.",
             fontsize=8.4, va="top", ha="left", color="#455A64", linespacing=1.6)

    fhead(fig, 0.345, 0.438, 5, "Attribution",
          "$d_k$ = v({k});   $d_{kl}$ = v({k,l}) − v({k}) − v({l});\n"
          "$\\varphi_k$ = $d_k$ + ½Σ$d_{kl}$, then anchored so Σ$\\varphi$ = v(N).")
    axa = fig.add_axes([0.345, BOT_Y, 0.215, BOT_H])
    axii = fig.add_axes([0.615, BOT_Y, 0.090, BOT_H])
    panel_attrib(axa, axii, D)
    s = sum(D["singles"]); vN = D["V"]["all"]["mpl"]
    fig.text(0.345, BOT_Y - 0.048,
             f"Non-additive: Σ$_k$ v({{k}}) = {s:.3f}  vs  v(N) = {vN:.3f} "
             f"(super-additive by {vN - s:+.3f}).\n"
             f"Σ$_k\\varphi_k$ = v(N) holds by construction. Order-≥3 dividends are "
             f"not measured under the\ninpaint arm; they are folded into the anchor "
             f"({o2_share(D)*100:.0f}% of v(N)) and split equally over the 4 cues.",
             fontsize=8.4, va="top", ha="left", color="#455A64", linespacing=1.6)

    fhead(fig, 0.760, 0.438, 6, "Artifact null → sensitive set",
          "Is $\\varphi_k$ larger than the floor inpainting itself creates?")
    axn = fig.add_axes([0.760, BOT_Y, 0.185, BOT_H])
    panel_null(axn, D)
    fig.text(0.760, BOT_Y - 0.048,
             f"$c_{{img}}$ = mean mPL of this image's 8 equal-area\n"
             f"inpainted controls (variants c$k$-$j$) = {D['c_img']:.4f};\n"
             f"per-cue null = $c_{{img}}$/m = {D['c_img']/D['m']:.4f}.",
             fontsize=8.4, va="top", ha="left", color="#455A64", linespacing=1.6)

    fig.text(0.5, 0.048,
             "Result for this image  —  sensitive cue set = "
             "{ 1 Equestrian statue,  2 American flag,  3 Historic building architecture,  "
             "4 Trees with autumn foliage } : all four $\\varphi_k$ clear the "
             "inpainting-artifact null.",
             ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.55", fc="#E8F5E9", ec=GREEN, lw=1.4))
    fig.text(0.5, 0.020,
             "Caveat visible in Step 6: cue 4 (trees) clears the image-level null, but its "
             "$\\varphi_k$ = 0.102 sits below its own equal-area controls (0.206 / 0.174) "
             "— the weakest of the four.",
             ha="center", va="center", fontsize=8.6, color="#455A64")

    out = os.path.join(FIGDIR, "pipeline_v2_walkthrough.png")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print("saved", out)
    return out


def o2_share(D):
    return D["o2"]["residual_share"]


# ------------------------------------------------------------- figure 2 -----

def box(ax, x, y, w, h, title, sub, fc, ec, tfs=11, sfs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.7, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h * 0.235, sub, ha="center", va="center", fontsize=sfs,
            color="#455A64", family="DejaVu Sans Mono", zorder=3)


def arrow(ax, p0, p1, color="#607D8B", lw=1.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=17,
                                 lw=lw, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def figure_diagram():
    fig = plt.figure(figsize=(18, 9.5), dpi=130)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.015, 0.030, 0.970, 0.890])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); bare(ax)
    fig.text(0.5, 0.962, "GeoBayes v2 — pipeline schematic  (implementing module "
             "under each box)", ha="center", va="top", fontsize=17.5,
             fontweight="bold", color=INK)

    W, H = 0.205, 0.185
    GAPX = 0.052
    xs = [0.020 + i * (W + GAPX) for i in range(3)]
    r1, r2, r3 = 0.735, 0.430, 0.125
    lb, lg, lo, lr = "#E3F2FD", "#E8F5E9", "#FFF3E0", "#FFEBEE"

    # ---- row 1 : left -> right ----
    box(ax, xs[0], r1, W, H, "1. Cue naming", "cue_extract/   (GPT-4o)", lb, BLUE)
    box(ax, xs[1], r1, W, H, "2. Cue masks", "sam3_seg.py   (SAM 3)", lb, BLUE)
    box(ax, xs[2], r1, W, H, "3. Geometric de-dup", "dedup_cues.py   IoU ≥ 0.9", lb, BLUE)
    arrow(ax, (xs[0] + W, r1 + H / 2), (xs[1], r1 + H / 2))
    arrow(ax, (xs[1] + W, r1 + H / 2), (xs[2], r1 + H / 2))

    # ---- row 2 : right -> left (serpentine) ----
    box(ax, xs[2], r2, W, H, "4. Remove subset S",
        "inpaint_ops.py  (LaMa)\ngray fill = robustness arm", lg, GREEN)
    box(ax, xs[1], r2, W, H, "5. Adversary belief",
        "georanker_belief.py\nrun_georanker_inpaint.py", lg, GREEN)
    box(ax, xs[0], r2, W, H, "6. mPL for that subset",
        "run_georanker_check.py::mpl\n138 labels · 2 km alias merge", lg, GREEN)
    arrow(ax, (xs[2] + W / 2, r1), (xs[2] + W / 2, r2 + H))
    arrow(ax, (xs[2], r2 + H / 2), (xs[1] + W, r2 + H / 2))
    arrow(ax, (xs[1], r2 + H / 2), (xs[0] + W, r2 + H / 2))
    ax.text(xs[2] + W / 2 + 0.010, (r1 + r2 + H) / 2, "de-duplicated cue set  {1..m}",
            fontsize=9, color="#546E7A", ha="left", va="center", style="italic")
    ax.text(xs[0] + W / 2 + 0.016, (r2 + r3 + H) / 2,
            "steps 4–6 run once for every scored subset S ⊆ {1..m}\n"
            "(inpaint arm: singles, pairs, all  →  11 scorings when m = 4)",
            fontsize=9, color="#546E7A", ha="left", va="center", style="italic",
            linespacing=1.5)

    # ---- row 3 : left -> right ----
    box(ax, xs[0], r3, W, H, "7. Set function v(S)",
        "run_georanker_lattice.py\nexact 2ᵐ   |   order2_shapley.py", lo, ORANGE)
    box(ax, xs[1], r3, W, H, "8. Shapley φ + interaction",
        "shapley_v3.py   (φ, SII, d_kl)", lo, ORANGE)
    box(ax, xs[2], r3, W, H, "9. Artifact null",
        "run_georanker_control.py\nprecompute_inpaint.py  (ck-j)", lo, ORANGE)
    arrow(ax, (xs[0] + W / 2, r2), (xs[0] + W / 2, r3 + H))
    arrow(ax, (xs[0] + W, r3 + H / 2), (xs[1], r3 + H / 2))
    arrow(ax, (xs[1] + W, r3 + H / 2), (xs[2], r3 + H / 2))

    fx, fw = xs[2] + W + GAPX, 0.190
    box(ax, fx, r3, fw, H, "10. Sensitive cue set",
        "keep cue k  iff  φ_k > c_img / m", lr, RED)
    arrow(ax, (xs[2] + W, r3 + H / 2), (fx, r3 + H / 2), color=RED)

    # ---- side box : changed vs v1 ----
    bx, bw = fx, 0.190
    by, bh = r2, r1 + H - r2
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#FAFAFA", ec="#90A4AE", lw=1.5, ls="--", zorder=2))
    ax.text(bx + bw / 2, by + bh - 0.035, "changed vs v1", ha="center", va="center",
            fontsize=12, fontweight="bold", color=INK, zorder=3)
    items = ["gray fill  →  LaMa inpainting",
             "no de-dup  →  IoU ≥ 0.9 de-dup",
             "exact 2ᵐ only  →  order-2 option",
             "gray null  →  inpainted null",
             "optional vocabulary inventory\n(extract_vocab.py — ablation only)"]
    for i, t in enumerate(items):
        ax.text(bx + 0.014, by + bh - 0.090 - i * 0.062, "• " + t, ha="left", va="top",
                fontsize=9, color="#37474F", linespacing=1.4, zorder=3)

    fig.text(0.020, 0.038,
             "Blue = what the attacker can point at.     "
             "Green = the scoring loop, run once per subset S.     "
             "Orange = the set-function / attribution layer.     "
             "Red = the deliverable.",
             fontsize=9.6, color="#455A64", ha="left")

    out = os.path.join(FIGDIR, "pipeline_v2_diagram.png")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print("saved", out)
    return out


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    D = load_case()
    print(f"image {D['iid']}  m={D['m']}  cues={D['cues']}")
    print("singles", [round(x, 4) for x in D["singles"]])
    print("phi    ", [round(x, 4) for x in D["o2"]["phi"]])
    print("c_img", round(D["c_img"], 4), "c/m", round(D["c_img"] / D["m"], 4))
    figure_walkthrough(D)
    figure_diagram()


if __name__ == "__main__":
    main()
