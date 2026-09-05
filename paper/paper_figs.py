"""Figure assets for the technical note `mPL_to_Shapley.pdf`.

Two jobs:
  1. draw Figure 1 (the v2 pipeline diagram) from scratch with matplotlib;
  2. copy the result figures produced by the `belief_elicit` plotting scripts
     into `paper/assets/` so that `build_pdf.py` has a self-contained asset dir.

Run:  python paper/paper_figs.py
"""
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "assets")
SRC_FIGS = os.path.join(ROOT, "belief_elicit", "figures")

# result figures pulled into paper/assets (dest name -> candidate sources, first hit wins)
COPY = {
    "fig2_sweep.png": ["georanker_sweep.png"],
    "fig3_overview.png": ["georanker_overview_dedup.png", "georanker_overview.png"],
    "fig4_dedup.png": ["dedup_before_after.png"],
    "fig5_vocab.png": ["vocab_vs_gpt4o_leakage.png"],
    "fig6_inpaint.png": ["inpaint_vs_gray.png"],
    "fig7_case_newyork.png": ["case_newyork_158307292.png"],
    "fig8_case_slovenia.png": ["case_slovenia_754780171.png"],
}

INK = "#1a1a1a"
EDGE = "#3b4a5a"
FILLS = {
    "data": "#eef1f4",
    "inventory": "#e3ecf7",
    "removal": "#fdeee2",
    "belief": "#e6f2ea",
    "attrib": "#efe8f5",
    "out": "#f7f1dd",
}


def _box(ax, x, y, w, h, title, lines, fill, title_size=10.2, body_size=8.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.1, edgecolor=EDGE, facecolor=fill, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.36, ln, ha="center", va="center",
                fontsize=body_size, color=INK, zorder=3)


def _arrow(ax, x0, y0, x1, y1, label=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
            color=EDGE, shrinkA=0, shrinkB=0, zorder=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.16, label, ha="center", va="bottom",
                fontsize=8.0, color=EDGE, zorder=4)


def pipeline_figure(out_path):
    """Figure 1 — the v2 cue-attribution pipeline."""
    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    y, h = 1.05, 2.75
    w = 2.30
    gap = 0.24
    xs = [0.16 + i * (w + gap) for i in range(5)]

    _box(ax, xs[0], y, w, h, "Cue inventory",
         ["GPT-4o cue naming", "(name, category,", "segment query)",
          "SAM 3 text-to-mask", "geometric de-dup", "IoU $\\geq$ 0.9: 244 $\\rightarrow$ 223"],
         FILLS["inventory"])

    _box(ax, xs[1], y, w, h, "Removal  $I \\ominus S$",
         ["LaMa inpainting", "(primary operator)", "gray fill",
          "(robustness variant)", "equal-area control", "placements (null)"],
         FILLS["removal"])

    _box(ax, xs[2], y, w, h, "Adversary belief",
         ["GeoRanker reward", "Qwen2-VL-7B + LoRA", "softmax $\\tau = 1$ over",
          "138 labels, 2 km", "alias dedup (137 pts)"],
         FILLS["belief"])

    _box(ax, xs[3], y, w, h, "Leakage set function",
         ["mPL (nats / 1000 km)", "over admissible pairs",
          "$v(S)$ on the subset", "lattice: exact $2^m$", "($m \\leq 5$); order-2",
          "anchored otherwise"],
         FILLS["attrib"])

    _box(ax, xs[4], y, w, h, "Attribution",
         ["Shapley value $\\varphi_k$", "with $\\Sigma_k \\varphi_k = v(N)$",
          "interaction index $I^{SII}$", "compare against the",
          "artifact floor $\\rightarrow$", "sensitive cue set"],
         FILLS["out"])

    for i in range(4):
        _arrow(ax, xs[i] + w + 0.02, y + h / 2, xs[i + 1] - 0.02, y + h / 2)

    # input / output caps
    ax.text(0.16 + w / 2, y + h + 0.42, "im2gps3k image  $I$  (100 hi-res)",
            ha="center", va="center", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=FILLS["data"],
                      edgecolor=EDGE, linewidth=1.0))
    _arrow(ax, 0.16 + w / 2, y + h + 0.20, 0.16 + w / 2, y + h + 0.03)

    ax.text(xs[4] + w / 2, y - 0.52,
            "per-cue leakage share  $\\varphi_k / v(N)$",
            ha="center", va="center", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=FILLS["data"],
                      edgecolor=EDGE, linewidth=1.0))
    _arrow(ax, xs[4] + w / 2, y - 0.04, xs[4] + w / 2, y - 0.30)

    # feedback: every subset S selected on the lattice is re-masked and re-scored
    fb = y - 0.42
    xa, xb = xs[3] + w / 2, xs[1] + w / 2
    ax.plot([xa, xa], [y - 0.02, fb], color="#7a8899", lw=1.2,
            ls=(0, (5, 3)), zorder=1)
    ax.plot([xa, xb], [fb, fb], color="#7a8899", lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.add_patch(
        FancyArrowPatch((xb, fb), (xb, y - 0.04),
                        arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                        linestyle=(0, (5, 3)), color="#7a8899",
                        shrinkA=0, shrinkB=0, zorder=1)
    )
    ax.text((xa + xb) / 2, fb - 0.24,
            "every subset $S \\subseteq N$ is masked and re-scored",
            ha="center", va="center", fontsize=8.6, color="#5c6b7a")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor="white")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(ASSETS, exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    out = pipeline_figure(os.path.join(ASSETS, "fig1_pipeline.png"))
    print("drew", os.path.relpath(out, ROOT))

    missing = []
    for dest, cands in COPY.items():
        for c in cands:
            src = os.path.join(SRC_FIGS, c)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(ASSETS, dest))
                print(f"copied {c} -> assets/{dest}")
                break
        else:
            missing.append(dest)
    if missing:
        print("MISSING (regenerate with the belief_elicit plot_* scripts):", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
