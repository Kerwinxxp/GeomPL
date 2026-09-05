"""Framework figure for the note: per-cue location-leakage measurement + attribution.

Draws ``paper/assets/fig_framework.png`` (300 dpi) and ``paper/assets/fig_framework.pdf``
(vector) from the real New York case study (image id prefix ``158307292``, m = 4 cues).

Panel (a) walks one subset S through the measurement pipeline
(cue proposal & grounding -> removal -> frozen adversary -> mPL), panel (b) shows what
happens once every subset has been scored (subset lattice -> Shapley attribution ->
equal-area artifact control -> sensitive cue set).

Every number and every thumbnail on the canvas is read from the checked-in results;
nothing is invented.  Run:

    python paper/fig_framework.py
"""
import itertools
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

from belief_elicit.attribution import order2_shapley
from belief_elicit.cues import cue_masks_of
from belief_elicit.results import (INPAINT, INPAINT_CACHE, INPAINT_CONTROLS, SAM3_DIR,
                                   SWEEP, image_path, load_inpaint, load_subsets,
                                   load_sweep)

OUT_DIR = os.path.join(ROOT, "paper", "assets")
IID_PREFIX = "158307292"

# ------------------------------------------------------------------ style ----
W_IN, H_IN = 16.0, 7.5

INK = "#1A1A1A"
MUTED = "#4A5A66"
FRAME = "#7A8A96"
GOLD = "#B8860B"
LEAF = "#2E7D32"
MAGENTA = "#AD1457"
BLUEBOX = "#1565C0"

CUE_COLORS = ["#1E88E5", "#E53935", "#43A047", "#FB8C00"]   # statue, flag, building, trees
CUE_SHORT = ["Equestrian statue", "American flag", "Historic building", "Autumn trees"]
SHORT_TAG = ["statue", "flag", "building", "trees"]

# gradient pairs (light -> dark) for the "network" blocks
G_ADV = ("#E8F1FB", "#8FB8E4")      # adversary: blue
G_SEG = ("#F0F1F2", "#BFC7CC")      # segmentation: grey
G_INP = ("#FFF3E3", "#F2BE7E")      # inpainting: orange
G_LLM = ("#F4ECF7", "#C8A6D6")      # cue-naming LLM: violet
G_ATT = ("#EAF5EC", "#A3CFAC")      # attribution: green

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": INK,
    "axes.linewidth": 0.7,
})


# ------------------------------------------------------------------- data ----

def load_case():
    """Everything the figure prints or draws, straight out of the results files."""
    rec = [r for r in load_sweep(SWEEP) if r["image_id"].startswith(IID_PREFIX)][0]
    iid = rec["image_id"]

    inp = load_inpaint(INPAINT)[iid]
    ctl = load_inpaint(INPAINT_CONTROLS)[iid]
    V = {v["spec"]: v for v in inp["variants"]}
    C = {v["spec"]: v for v in ctl["variants"]}

    cues, cats, masks, (W, H) = cue_masks_of(SAM3_DIR, iid)
    m = len(cues)
    img = Image.open(image_path(load_subsets()[iid])).resize((W, H)).convert("RGB")

    cache = os.path.join(INPAINT_CACHE, iid)
    thumbs = {f: Image.open(os.path.join(cache, f + ".png")).convert("RGB")
              for f in ("s2", "p0-2", "all", "c2-0")}

    singles = [V[f"s{k}"]["mpl"] for k in range(m)]
    pairs = {(k, l): V[f"p{k}-{l}"]["mpl"] for k, l in itertools.combinations(range(m), 2)}
    vN = V["all"]["mpl"]
    o2 = order2_shapley(singles, pairs, vN)

    # order-2 anchored phi is what shapley_v3 stores as phi_inpaint; prefer the stored value
    import json
    sh = json.load(open(os.path.join(ROOT, "belief_elicit", "shapley_v3_results.json"),
                        encoding="utf-8"))
    shr = [x for x in sh if x["image_id"] == iid][0]
    phi = [c.get("phi_inpaint", c["phi"]) for c in shr["cues"]]
    sii = shr["sii"]

    ctrl_mpl = {k: v["mpl"] for k, v in C.items()}
    c_img = float(np.mean(list(ctrl_mpl.values())))
    ctrl_max = {k: max(ctrl_mpl[f"c{k}-{j}"] for j in (0, 1)) for k in range(m)}

    post = rec["posterior"]
    top5 = [l for l, _ in sorted(post.items(), key=lambda kv: -kv[1])[:5]]
    prior_s2 = V["s2"]["prior"]

    return dict(iid=iid, m=m, cues=cues, cats=cats, masks=masks, size=(W, H), img=img,
                thumbs=thumbs, singles=singles, pairs=pairs, vN=vN, o2=o2, phi=phi,
                sii=sii, c_img=c_img, ctrl_max=ctrl_max, ctrl_mpl=ctrl_mpl,
                post=post, prior_s2=prior_s2, top5=top5,
                true_label=rec["true_label"], n_cand=len(post),
                residual=shr["inpaint"]["residual_share"], V=V)


def report(D):
    m = D["m"]
    print(f"image            {D['iid']}")
    print(f"true label       {D['true_label']}   candidates {D['n_cand']}")
    print("beliefs (top-5 by full-image posterior):")
    for l in D["top5"]:
        print(f"   {l:34s} Pr(.|I)={D['post'][l]:.4f}   Pr(.|I-S3)={D['prior_s2'].get(l, 0):.4f}")
    print("lattice v(S) [nats/1000 km]:")
    print("   empty 0.0000  " + "  ".join(f"{{{k+1}}}={D['singles'][k]:.4f}" for k in range(m)))
    print("   " + "  ".join(f"{{{k+1},{l+1}}}={v:.4f}" for (k, l), v in D["pairs"].items()))
    print(f"   N={D['vN']:.4f}   order-2 residual share={D['residual']:.3f}")
    print("phi (order-2 anchored, inpainting arm): "
          + "  ".join(f"phi{k+1}={D['phi'][k]:.4f}" for k in range(m))
          + f"   sum={sum(D['phi']):.4f}")
    print("SII: " + "  ".join(f"{k}={v:+.4f}" for k, v in D["sii"].items()))
    print(f"c_img={D['c_img']:.4f}  c_img/m={D['c_img']/m:.4f}")
    keep = [k for k in range(m) if D["phi"][k] > D["c_img"] / m]
    print("sensitive cue set: " + ", ".join(CUE_SHORT[k] for k in keep))
    return keep


# ---------------------------------------------------------------- helpers ----

def fx(x):
    return x / W_IN


def fy(y):
    return y / H_IN


def inset(fig, x0, y0, w, h, z=6):
    ax = fig.add_axes([fx(x0), fy(y0), fx(w), fy(h)], zorder=z)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def thumb(fig, im, x0, y0, h, ec="#37474F", lw=0.9, z=6):
    """Draw a PIL image with height `h` inches, keeping its aspect; returns (ax, width)."""
    w = h * im.size[0] / im.size[1]
    ax = inset(fig, x0, y0, w, h, z=z)
    ax.imshow(im)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(ec); s.set_linewidth(lw)
    return ax, w


def stage(ax, x0, y0, x1, y1, step, title, color, fs=8.0, bar_w=0.34):
    """Dashed rounded stage frame with a vertical coloured title bar at the left edge."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                fc="white", ec=FRAME, lw=1.0, ls=(0, (5, 3)), zorder=1))
    ax.add_patch(FancyBboxPatch((x0 + 0.055, y0 + 0.055), bar_w, (y1 - y0) - 0.11,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc=color, ec="none", alpha=0.92, zorder=2))
    ax.text(x0 + 0.055 + bar_w / 2, (y0 + y1) / 2, f"Step {step}\n{title}", rotation=90,
            ha="center", va="center", rotation_mode="anchor", linespacing=1.15,
            fontsize=fs, fontweight="bold", style="italic", color="white",
            multialignment="center", zorder=3)
    return x0 + 0.055 + bar_w + 0.17      # left edge of the usable interior


def grad_block(ax, x0, x1, yc, h, colors, label, sub=None, taper=0.62, ec="#455A64",
               fs=9.5, subfs=7.4, layers=3, frozen=False):
    """Trapezoidal 'network' block with a light horizontal gradient and stacked layers."""
    hh, ht = h / 2, h * taper / 2
    pts = [(x0, yc - hh), (x0, yc + hh), (x1, yc + ht), (x1, yc - ht)]
    poly = Polygon(pts, closed=True, fc="none", ec=ec, lw=1.1, zorder=4,
                   joinstyle="round")
    ax.add_patch(poly)
    cmap = LinearSegmentedColormap.from_list("g", list(colors))
    im = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, origin="lower",
                   extent=[x0, x1, yc - hh, yc + hh], zorder=3, interpolation="bilinear")
    im.set_clip_path(poly)
    for i in range(1, layers + 1):                      # stacked-layer hint
        t = i / (layers + 1)
        xx = x0 + t * (x1 - x0)
        hy = hh + t * (ht - hh)
        ax.plot([xx, xx], [yc - hy * 0.86, yc + hy * 0.86], color=ec, lw=0.5,
                alpha=0.45, zorder=5)
    dy = 0.0 if sub is None else 0.085
    ax.text((x0 + x1) / 2, yc + dy, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", style="italic", color=INK, zorder=6)
    if sub:
        ax.text((x0 + x1) / 2, yc + dy - 0.115, sub, ha="center", va="top",
                fontsize=subfs, color=MUTED, zorder=6, linespacing=1.35)
    if frozen:
        snowflake(ax, x0 + 0.15, yc + hh - 0.16, 0.072, "#1565C0")
    return (x0 + x1) / 2


def snowflake(ax, x, y, r, color="#1565C0", lw=1.0, z=7):
    """Small ❄ drawn from primitives (no glyph dependency)."""
    ax.add_patch(Circle((x, y), r * 1.55, fc="white", ec=color, lw=0.7, zorder=z))
    for a in (90, 30, 150):
        t = np.deg2rad(a)
        dx, dy = r * np.cos(t), r * np.sin(t)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=lw, zorder=z + 1,
                solid_capstyle="round")
        for s in (-1, 1):
            bx, by = x + s * dx, y + s * dy
            for b in (a + 140, a - 140):
                tb = np.deg2rad(b)
                ax.plot([bx, bx + s * r * 0.42 * np.cos(tb)],
                        [by, by + s * r * 0.42 * np.sin(tb)],
                        color=color, lw=lw * 0.8, zorder=z + 1, solid_capstyle="round")


def labelbox(ax, x, y, text, color, fs=7.6, pad=0.30, fc=None, ha="center", va="center",
             weight="normal", z=6, ls="-", lw=0.9, style="normal", linespacing=1.35):
    """Rounded label box with a thin coloured border (named-tensor style)."""
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=INK, zorder=z,
                   fontweight=weight, style=style, linespacing=linespacing,
                   bbox=dict(boxstyle=f"round,pad={pad}",
                             fc=(fc if fc else to_rgba(color, 0.10)), ec=color,
                             lw=lw, ls=ls))


def arrow(ax, p0, p1, color=INK, lw=1.2, rad=0.0, ls="-", z=5, mut=8.0, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                                 lw=lw, color=color, zorder=z, alpha=alpha,
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def mask_rgba(mask, color, alpha=0.55):
    r, g, b, _ = to_rgba(color)
    out = np.zeros(mask.shape + (4,))
    out[mask] = (r, g, b, alpha)
    return out


def sil(mask, color, bg="#FFFFFF"):
    """Binary silhouette of one mask on a plain background (for the stacked M_k thumbs)."""
    r, g, b, _ = to_rgba(color)
    br, bg_, bb, _ = to_rgba(bg)
    out = np.zeros(mask.shape + (3,))
    out[...] = (br, bg_, bb)
    out[mask] = (r, g, b)
    return out


def barpanel(fig, x0, y0, w, h, labels, vals, hi, color, title, xmax, fs=6.6):
    """Tiny horizontal bar chart of a belief; `hi` is the index to highlight."""
    ax = inset(fig, x0, y0, w, h, z=7)
    y = np.arange(len(vals))[::-1]
    cols = [color if i == hi else "#C3CDD4" for i in range(len(vals))]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="#5A6B75", linewidth=0.45)
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.035, y[i], f"{v:.3f}", va="center", ha="left", fontsize=fs,
                color=(INK if i == hi else MUTED),
                fontweight=("bold" if i == hi else "normal"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs, color=INK)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlim(0, xmax * 1.48)
    ax.set_ylim(-0.7, len(vals) - 0.3)
    ax.set_xticks([])
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#90A4AE")
    ax.spines["left"].set_linewidth(0.7)
    ax.set_title(title, fontsize=7.2, color=INK, pad=2.5, loc="left")
    return ax


def shortlab(l):
    a = l.split(",")[0]
    return a.replace("Town of ", "")


# ------------------------------------------------------------------- draw ----

def build(D, keep):
    m = D["m"]
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_xlim(0, W_IN); ax.set_ylim(0, H_IN)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_facecolor("white")

    AX0, AX1 = 0.22, 8.53          # panel (a) horizontal span
    BX0, BX1 = 9.32, 15.85         # panel (b) horizontal span

    fig.text(fx(0.22), fy(7.33),
             "Per-cue location leakage: (a) measuring one subset, (b) attributing over the subset lattice",
             fontsize=12.0, fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(fx(15.85), fy(7.33),
             f"case study: {D['true_label']}  ·  m = {m} cues  ·  "
             f"{D['n_cand']} candidate GPS labels",
             fontsize=8.2, ha="right", va="bottom", color=MUTED, style="italic")

    # =================================================== (a) step 1 ==========
    ix = stage(ax, AX0, 5.15, AX1, 7.22, 1, "Cue proposal & grounding", "#7E57C2")
    yc, TH = 6.02, 0.95

    _, w = thumb(fig, D["img"], ix + 0.04, yc - TH / 2, TH)
    xI0 = ix + 0.04
    ax.text(xI0 + w / 2, yc - TH / 2 - 0.05, r"image $I$", ha="center", va="top",
            fontsize=8.0, color=INK)
    labelbox(ax, xI0 + w / 2, 6.78, "Input", MAGENTA, fs=7.4, pad=0.24)

    x = xI0 + w + 0.02
    arrow(ax, (x, yc), (x + 0.18, yc))
    grad_block(ax, x + 0.20, x + 0.96, yc, 0.98, G_LLM, "GPT-4o",
               "cue naming\n(text only)", frozen=True, fs=9.0, subfs=7.0)
    x = x + 0.96

    arrow(ax, (x, yc), (x + 0.18, yc))
    lst = "\n".join(f"{k+1}.  {CUE_SHORT[k]}" for k in range(m))
    bx0, bx1 = x + 0.20, x + 1.46
    ax.add_patch(FancyBboxPatch((bx0, yc - 0.50), bx1 - bx0, 1.00,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc=to_rgba(MAGENTA, 0.07), ec=MAGENTA, lw=0.9, zorder=4))
    ax.text(bx0 + 0.24, yc + 0.324, lst, ha="left", va="top", fontsize=7.3, color=INK,
            linespacing=1.60, zorder=6)
    for k in range(m):
        ax.add_patch(FancyBboxPatch((bx0 + 0.10, yc + 0.202 - k * 0.1622), 0.082, 0.082,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=CUE_COLORS[k], ec="none", zorder=6))
    ax.text((bx0 + bx1) / 2, 6.78, "cue names + segment queries", ha="center",
            va="center", fontsize=7.2, color=MUTED, style="italic")
    x = bx1

    arrow(ax, (x, yc), (x + 0.18, yc))
    grad_block(ax, x + 0.20, x + 0.96, yc, 0.98, G_SEG, "SAM 3",
               "text $\\rightarrow$ mask", frozen=True, fs=9.0, subfs=7.0)
    x = x + 0.96

    arrow(ax, (x, yc), (x + 0.16, yc))
    # stacked, offset silhouettes M_1..M_4
    sh = 0.80
    sw = sh * D["size"][0] / D["size"][1]
    sx, sy = x + 0.20, yc - sh / 2 + 0.06
    for k in (3, 2, 1, 0):
        off = 0.050 * k
        a2 = inset(fig, sx + off, sy - off, sw, sh, z=6 + (4 - k))
        a2.imshow(sil(D["masks"][k], CUE_COLORS[k]))
        for s in a2.spines.values():
            s.set_visible(True); s.set_color(CUE_COLORS[k]); s.set_linewidth(0.9)
    ax.text(sx + sw / 2 + 0.075, sy - 0.20, r"$M_1 \ldots M_4$", ha="center", va="top",
            fontsize=8.0, color=INK)
    x = sx + sw + 0.15

    ovw = TH * D["size"][0] / D["size"][1]
    ov = inset(fig, x + 0.20, yc - TH / 2, ovw, TH, z=6)
    ov.imshow(D["img"])
    for k in range(m):
        ov.imshow(mask_rgba(D["masks"][k], CUE_COLORS[k], 0.45))
        ov.contour(D["masks"][k].astype(float), levels=[0.5], colors=[CUE_COLORS[k]],
                   linewidths=0.8)
    for s in ov.spines.values():
        s.set_visible(True); s.set_color("#37474F"); s.set_linewidth(0.9)
    ax.text(x + 0.20 + ovw / 2, yc - TH / 2 - 0.05, "masks on $I$", ha="center",
            va="top", fontsize=8.0, color=INK)
    ax.text(x + 0.20 + ovw / 2, 6.78, "grounded cue masks", ha="center", va="center",
            fontsize=7.2, color=MUTED, style="italic")
    x = x + 0.20 + ovw

    arrow(ax, (x + 0.04, yc), (x + 0.24, yc))
    dcx = 7.68
    labelbox(ax, dcx, 6.42,
             "geometric de-dup\nmerge if mask IoU $\\geq 0.9$\n(0 merges here)",
             "#546E7A", fs=7.2, pad=0.32)
    arrow(ax, (dcx, 6.08), (dcx, 5.88))
    labelbox(ax, dcx, 5.64, "players  $N=\\{1,2,3,4\\}$", BLUEBOX, fs=8.0, pad=0.30,
             weight="bold")

    # =================================================== (a) step 2 ==========
    ix = stage(ax, AX0, 2.75, 3.28, 4.85, 2, "Removal  $\\ominus$", "#EF6C00")
    _, wi = thumb(fig, D["img"], ix + 0.02, 3.80, 0.60)
    cI = ix + 0.02 + wi / 2
    ax.text(cI - 0.04, 4.44, "$I$", ha="center", va="bottom", fontsize=8.2)
    ms = inset(fig, ix + 0.02, 3.00, wi, 0.60, z=6)
    ms.imshow(D["img"])
    ms.imshow(mask_rgba(D["masks"][2], CUE_COLORS[2], 0.72))
    for s in ms.spines.values():
        s.set_visible(True); s.set_color(CUE_COLORS[2]); s.set_linewidth(1.0)
    ax.text(cI, 2.95, "$M_S,\\; S=\\{3\\}$", ha="center", va="top", fontsize=7.4,
            color=INK)

    bx = ix + 0.02 + wi
    arrow(ax, (bx + 0.02, 4.08), (bx + 0.26, 3.92), rad=-0.12)
    arrow(ax, (bx + 0.02, 3.32), (bx + 0.26, 3.46), rad=0.12)
    grad_block(ax, bx + 0.28, bx + 1.06, 3.68, 1.00, G_INP, "LaMa",
               "inpainting\nremoval", frozen=True, fs=9.0, subfs=7.0)

    arrow(ax, (bx + 1.08, 3.68), (bx + 1.28, 3.68))
    wo = 0.0
    for j, key in ((2, "all"), (1, "p0-2"), (0, "s2")):
        _, wo = thumb(fig, D["thumbs"][key], bx + 1.30 + 0.055 * j, 3.34 - 0.055 * j,
                      0.86, z=6 + (2 - j))
    cS = bx + 1.30 + 0.055 + wo / 2
    ax.text(cS, 3.20, "$I \\ominus S$", ha="center", va="top", fontsize=8.2, color=INK)
    ax.text(cS, 3.05, "front $S=\\{3\\}$;\nbehind $\\{1,3\\}$, $N$", ha="center",
            va="top", fontsize=6.3, color=MUTED, style="italic", linespacing=1.35)
    ax.text(1.90, 2.99, "gray fill:\nrobustness variant", ha="center", va="top",
            fontsize=6.6, color=MUTED, style="italic", linespacing=1.35)

    # =================================================== (a) step 3 ==========
    ix = stage(ax, 3.46, 2.75, AX1, 4.85, 3, "Frozen adversary $\\mathcal{A}$", "#1565C0")
    arrow(ax, (3.22, 3.62), (3.43, 3.62), lw=1.3)
    grad_block(ax, ix + 0.04, ix + 1.14, 3.78, 1.20, G_ADV, "Adversary $\\mathcal{A}$",
               "GeoRanker\nQwen2-VL-7B + LoRA\n+ value head", frozen=True, fs=9.2,
               subfs=6.9, layers=4)
    # green secondary flow: the full-image posterior is computed once and reused
    arrow(ax, (cI + wi / 2, 4.40), (3.43, 4.10), color=LEAF, lw=1.2, ls=(0, (4, 3)),
          rad=-0.13, mut=9)
    ax.text(2.36, 4.54, "reuse $\\Pr_\\theta(\\cdot\\,|\\,I)$ for every $S$",
            ha="center", va="bottom", fontsize=7.0, color=LEAF, style="italic")

    x = ix + 1.14
    arrow(ax, (x + 0.02, 3.78), (x + 0.20, 3.78))
    labelbox(ax, x + 0.78, 3.78,
             "scores $s_\\theta(x;\\cdot)$ on\n$\\mathcal{X}$: 138 GPS labels\n"
             "$\\downarrow$ softmax $(\\tau\\!=\\!1)$\n$\\downarrow$ merge $<2$ km",
             BLUEBOX, fs=7.3, pad=0.30)
    x = x + 1.38
    arrow(ax, (x + 0.02, 3.78), (x + 0.22, 3.78))

    labs = [shortlab(l) for l in D["top5"]]
    pv = [D["post"][l] for l in D["top5"]]
    qv = [D["prior_s2"].get(l, 0.0) for l in D["top5"]]
    xmax = max(max(pv), max(qv))
    barpanel(fig, AX1 - 1.16, 3.92, 1.10, 0.62, labs, pv, 0, "#1565C0",
             "$\\Pr_\\theta(\\cdot\\,|\\,I)$", xmax)
    barpanel(fig, AX1 - 1.16, 3.00, 1.10, 0.62, labs, qv, 0, "#EF6C00",
             "$\\Pr_\\theta(\\cdot\\,|\\,I\\ominus S)$", xmax)
    ax.text(AX1 - 0.08, 2.90, "true label in colour; identical label order",
            ha="right", va="top", fontsize=6.8, color=MUTED, style="italic")

    # =================================================== (a) step 4 ==========
    ix = stage(ax, AX0, 0.90, AX1, 2.15, 4, "mPL $\\rightarrow v(S)$", "#00838F")
    ax.add_patch(FancyBboxPatch((ix + 0.02, 1.18), 5.28, 0.84,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc="#F4FAFB", ec="#00838F", lw=0.9, zorder=4))
    ax.text(ix + 0.18, 1.92,
            r"$v(S)\;=\;\mathrm{mPL}(S)\;=\;\underset{i<j}{\mathrm{mean}}\;"
            r"\frac{\left|\,\lambda_i-\lambda_j\,\right|}{d(x_i,x_j)}\times 1000$"
            r"$\qquad$"
            r"$\lambda_x=\log\frac{\mathrm{Pr}_\theta(x\,|\,I)}"
            r"{\mathrm{Pr}_\theta(x\,|\,I\ominus S)}$",
            ha="left", va="top", fontsize=10.0, color=INK, zorder=6)
    ax.text(ix + 0.18, 1.50,
            "units nats / 1000 km  ·  $d$ = haversine distance between candidates  ·  "
            "geo-indistinguishability: each pairwise term is an empirical $\\varepsilon$",
            ha="left", va="top", fontsize=7.4, color=MUTED, zorder=6)

    labelbox(ax, 7.24, 1.66, "$v(S) = 0.4287$   for $S=\\{3\\}$", LEAF, fs=9.2,
             pad=0.32, weight="bold")
    labelbox(ax, 7.24, 1.22, f"$v(N) = {D['vN']:.4f}$   (all 4 cues)", LEAF, fs=8.0,
             pad=0.28)

    # gold feedback loop: repeat for every scored subset
    arrow(ax, (6.05, 2.72), (1.70, 2.72), color=GOLD, lw=1.4, ls=(0, (5, 3)), rad=-0.14,
          mut=10)
    ax.text(3.88, 2.26,
            "repeat for every scored subset $S$: singles $s_k$, pairs $p_{k,l}$, all "
            "($1+m+\\binom{m}{2}+1 = 12$ forward passes)",
            ha="center", va="bottom", fontsize=7.4, color=GOLD, style="italic")

    # (a) -> (b) corridor
    arrow(ax, (8.98, 1.85), (8.98, 4.45), color=LEAF, lw=1.4, ls=(0, (4, 3)), mut=10)
    ax.text(8.82, 3.15, "each $v(S)$ = one lattice node", rotation=90, ha="center",
            va="center", rotation_mode="anchor", fontsize=7.2, color=LEAF,
            style="italic")

    # =================================================== (b) step 5 ==========
    ix = stage(ax, BX0, 4.28, BX1, 7.22, 5, "Subset lattice $v(S)$", "#00695C")
    LX0, LX1 = ix + 0.02, BX1 - 0.12
    nodes = {}
    NW, NH = 0.94, 0.34

    def node(key, cx, cy, top, val, color, ls="-", ghost=False):
        fc = "white" if ghost else to_rgba(color, 0.11)
        ax.add_patch(FancyBboxPatch((cx - NW / 2, cy - NH / 2), NW, NH,
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=fc, ec=color, lw=0.9, ls=ls, zorder=5))
        ax.text(cx, cy + 0.075, top, ha="center", va="center", fontsize=7.4,
                color=(MUTED if ghost else INK), zorder=6,
                fontweight=("normal" if ghost else "bold"))
        ax.text(cx, cy - 0.085, val, ha="center", va="center", fontsize=7.0,
                color=(MUTED if ghost else INK), zorder=6,
                style=("italic" if ghost else "normal"))
        nodes[key] = (cx, cy)

    y_e, y_s, y_p, y_t, y_n = 4.62, 5.16, 5.72, 6.28, 6.84
    xs_single = [10.34, 11.92, 13.50, 15.08]
    xs_pair = [LX0 + NW / 2 + i * ((LX1 - LX0 - NW) / 5) for i in range(6)]

    node("e", 12.71, y_e, "$\\emptyset$", "0.000", "#90A4AE")
    for k in range(m):
        node(("s", k), xs_single[k], y_s, "$\\{%d\\}$" % (k + 1),
             f"{D['singles'][k]:.3f}", CUE_COLORS[k])
    for i, ((k, l), v) in enumerate(D["pairs"].items()):
        node(("p", k, l), xs_pair[i], y_p, "$\\{%d,%d\\}$" % (k + 1, l + 1),
             f"{v:.3f}", "#455A64")
    for i, T in enumerate(itertools.combinations(range(m), 3)):
        node(("t",) + T, xs_single[i], y_t,
             "$\\{%s\\}$" % ",".join(str(t + 1) for t in T), "not scored",
             "#B0BEC5", ls=(0, (2, 2)), ghost=True)
    node("N", 12.71, y_n, "$N=\\{1,2,3,4\\}$", f"{D['vN']:.3f}", LEAF)

    def edge(a, b, ghost=False):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.plot([x0, x1], [y0 + NH / 2, y1 - NH / 2],
                color=("#CFD8DC" if ghost else "#B0BEC5"), lw=0.45, zorder=2,
                ls=((0, (2, 2)) if ghost else "-"), alpha=0.9)

    for k in range(m):
        edge("e", ("s", k))
    for (k, l) in D["pairs"]:
        edge(("s", k), ("p", k, l)); edge(("s", l), ("p", k, l))
    for T in itertools.combinations(range(m), 3):
        for (k, l) in itertools.combinations(T, 2):
            edge(("p", k, l), ("t",) + T, ghost=True)
        edge(("t",) + T, "N", ghost=True)

    ax.text(LX0 + 0.02, 4.48, "$v(S)$: mPL from Step 4", ha="left", va="center",
            fontsize=7.2, color=MUTED, style="italic")
    ax.text(LX1, 4.48, "triples skipped (GPU budget)\n$\\rightarrow$ order-2 anchored",
            ha="right", va="center", fontsize=7.2, color=MUTED, style="italic",
            linespacing=1.35)

    # =================================================== (b) step 6 ==========
    ix = stage(ax, BX0, 2.30, BX1, 4.15, 6, "Shapley attribution", "#2E7D32")
    grad_block(ax, ix + 0.00, ix + 0.94, 3.28, 1.00, G_ATT, "$\\phi_k$",
               "Shapley\nvalue", fs=10.5, subfs=7.2, layers=3, taper=0.72)
    ax.text(ix + 0.47, 2.62, "order-2 anchored", ha="center", va="top", fontsize=7.0,
            color=MUTED, style="italic")
    arrow(ax, (ix + 0.96, 3.28), (ix + 1.12, 3.28))

    px0, pw = ix + 1.80, 1.45
    axp = inset(fig, px0, 2.68, pw, 1.14, z=7)
    yy = np.arange(m)[::-1]
    axp.barh(yy, D["phi"], height=0.62, color=CUE_COLORS, edgecolor="#37474F", lw=0.5)
    thr = D["c_img"] / m
    axp.axvline(thr, color=MAGENTA, lw=1.1, ls=(0, (3, 2)), zorder=5)
    for k in range(m):
        axp.text(D["phi"][k] + 0.012, yy[k], f"{D['phi'][k]:.3f}", va="center",
                 ha="left", fontsize=6.8, color=INK)
    axp.set_yticks(yy)
    axp.set_yticklabels([f"{k+1} {t}" for k, t in enumerate(SHORT_TAG)], fontsize=6.9)
    axp.tick_params(axis="y", length=0, pad=1.5)
    axp.set_xlim(0, max(D["phi"]) * 1.30)
    axp.set_ylim(-1.05, m - 0.30)
    axp.set_xticks([])
    axp.spines["left"].set_visible(True); axp.spines["left"].set_color("#90A4AE")
    axp.spines["left"].set_linewidth(0.7)
    axp.set_title("$\\phi_k$  [nats / 1000 km]", fontsize=7.2, loc="left", pad=2.5)
    axp.text(thr, -0.74, "  artifact null $c_{img}/m$", ha="left", va="center",
             fontsize=6.6, color=MAGENTA)
    ax.text(px0 - 0.62, 2.58, f"$\\sum_k \\phi_k = v(N) = {D['vN']:.4f}$  (efficiency)",
            ha="left", va="top", fontsize=7.4, color=INK)

    sx0, swid = 13.78, 1.86
    axs = inset(fig, sx0, 2.68, swid, 1.14, z=7)
    keys = list(D["sii"].keys())
    sv = [D["sii"][k] for k in keys]
    yy2 = np.arange(len(keys))[::-1]
    axs.barh(yy2, sv, height=0.62,
             color=["#C62828" if v < 0 else "#1565C0" for v in sv],
             edgecolor="#37474F", lw=0.5)
    axs.axvline(0, color="#546E7A", lw=0.7)
    lim = max(abs(min(sv)), abs(max(sv))) * 1.75
    for i, v in enumerate(sv):
        axs.text(v + (0.03 * lim if v >= 0 else -0.03 * lim), yy2[i], f"{v:+.3f}",
                 va="center", ha=("left" if v >= 0 else "right"), fontsize=6.5,
                 color=INK)
    axs.set_yticks(yy2)
    axs.set_yticklabels(["$\\{%d,%d\\}$" % (int(k.split(",")[0]) + 1,
                                            int(k.split(",")[1]) + 1) for k in keys],
                        fontsize=6.6)
    axs.tick_params(axis="y", length=0, pad=1.5)
    axs.set_xlim(-lim, lim)
    axs.set_ylim(-0.7, len(keys) - 0.3)
    axs.set_xticks([])
    axs.spines["left"].set_visible(False)
    axs.set_title("$I^{\\mathrm{SII}}_{k\\ell}$: overlap $<0$ · backup $>0$",
                  fontsize=7.2, loc="left", pad=2.5)
    ax.text(sx0 - 0.40, 2.58,
            "$\\{3,4\\}=-0.360$: building and trees cover\n"
            "the same evidence (redundant, not additive)",
            ha="left", va="top", fontsize=7.0, color=MUTED, linespacing=1.4)

    # =================================================== (b) step 7 ==========
    ix = stage(ax, BX0, 0.90, BX1, 2.15, 7, "Equal-area control", "#AD1457")
    _, wc = thumb(fig, D["thumbs"]["c2-0"], ix + 0.02, 1.02, 0.80)
    ax.text(ix + 0.02 + wc / 2, 1.84, "control, cue 3", ha="center", va="bottom",
            fontsize=7.0, color=INK)
    arrow(ax, (ix + 0.04 + wc, 1.52), (ix + 0.26 + wc, 1.52))
    labelbox(ax, 11.82, 1.52,
             "equal-area control: inpaint a\nshifted copy of each mask\n"
             f"$c_{{img}}$ = mean of $2m=8$ = ${D['c_img']:.4f}$\n"
             f"artifact null $c_{{img}}/m = {D['c_img']/m:.4f}$",
             "#AD1457", fs=7.2, pad=0.30)
    arrow(ax, (12.80, 1.62), (13.24, 1.62))
    labelbox(ax, 14.36, 1.63,
             "sensitive cue set  $\\{k:\\phi_k > c_{img}/m\\}$\n"
             "= { " + ", ".join(SHORT_TAG[k] for k in keep) + " }\n"
             f"all {len(keep)} of {m} cues clear the artifact null",
             LEAF, fs=7.8, pad=0.30, weight="bold")
    ax.text(14.36, 1.07,
            "exact over $2^m$ when the lattice is complete ($m\\leq 5$);\n"
            "order-2 anchored otherwise (residual share "
            f"{D['residual']:.2f})",
            ha="center", va="center", fontsize=6.7, color=MUTED, style="italic",
            linespacing=1.4)

    # ================================================= legend + captions =====
    fig.text(fx((AX0 + AX1) / 2), fy(0.63),
             "(a)  Measurement — one subset $S$", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color=INK)
    fig.text(fx((BX0 + BX1) / 2), fy(0.63),
             "(b)  Attribution — over the subset lattice", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color=INK)

    ax.add_patch(FancyBboxPatch((AX0, 0.12), AX1 - AX0, 0.40,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#FAFAFA", ec="#CFD8DC", lw=0.8, zorder=1))
    lx = AX0 + 0.22
    snowflake(ax, lx, 0.32, 0.075)
    ax.text(lx + 0.16, 0.32, "frozen model (no training anywhere)", ha="left",
            va="center", fontsize=7.6, color=INK)
    lx += 2.34
    ax.add_patch(FancyBboxPatch((lx - 0.07, 0.25), 0.15, 0.15,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="#1565C0", ec="#1565C0", lw=0.8, zorder=3))
    ax.text(lx + 0.16, 0.32, "scored by adversary $\\mathcal{A}$", ha="left",
            va="center", fontsize=7.6, color=INK)
    lx += 1.62
    ax.add_patch(FancyBboxPatch((lx - 0.07, 0.25), 0.15, 0.15,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec="#78909C", lw=0.8, zorder=3))
    ax.text(lx + 0.16, 0.32, "post-processing (not scored)", ha="left", va="center",
            fontsize=7.6, color=INK)
    lx += 1.92
    ax.text(lx, 0.32, "$\\ominus$", ha="center", va="center", fontsize=12,
            color="#EF6C00")
    ax.text(lx + 0.16, 0.32, "removal operator", ha="left", va="center", fontsize=7.6,
            color=INK)
    lx += 1.32
    ax.add_patch(Circle((lx, 0.32), 0.085, fc="white", ec="#546E7A", lw=0.9, zorder=3))
    ax.text(lx, 0.315, "$\\cup$", ha="center", va="center", fontsize=7.0,
            color="#546E7A", zorder=4)
    ax.text(lx + 0.16, 0.32, "union of the masks in $S$", ha="left", va="center",
            fontsize=7.6, color=INK)

    return fig


def main():
    D = load_case()
    keep = report(D)
    fig = build(D, keep)
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "fig_framework.png")
    pdf = os.path.join(OUT_DIR, "fig_framework.pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print("saved", png)
    print("saved", pdf)


if __name__ == "__main__":
    main()
