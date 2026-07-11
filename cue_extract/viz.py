"""⑥ 可视化 QA:原图 + SAM mask 半透明叠加(颜色=线索序号),编号+图例(PLAN §2⑥)。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rle import rle_to_mask

COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
          "#00ACC1", "#F06292", "#7CB342", "#5E57C2", "#8D6E63"]


def render(image, cues: list, out_path: str, title: str = ""):
    """cues: 最终 JSON 条目(instances 内含 bbox / 可选 mask_rle)。"""
    img = np.asarray(image.convert("RGB"))
    h, w = img.shape[:2]
    overlay = img.astype(float).copy()
    fig, ax = plt.subplots(figsize=(11, 11 * h / w * 0.62 + 2.2))
    for k, c in enumerate(cues):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        for inst in c.get("instances", []):
            if inst.get("mask_rle"):
                m = rle_to_mask(inst["mask_rle"])
                overlay[m] = 0.55 * overlay[m] + 0.45 * col
    ax.imshow(overlay.astype(np.uint8))
    for k, c in enumerate(cues):
        col = COLORS[k % len(COLORS)]
        for inst in c.get("instances", []):
            x1, y1, x2, y2 = inst["bbox"]
            ls = "--" if inst.get("degenerate", c.get("degenerate")) else "-"
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=col, linewidth=1.6, linestyle=ls))
        if c.get("instances"):
            x1, y1 = c["instances"][0]["bbox"][:2]
            ax.text(x1 + 2, y1 + 2, str(k + 1), color="white", fontsize=10, fontweight="bold",
                    va="top", bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    lines = []
    for k, c in enumerate(cues):
        txt = next((i.get("text") for i in c.get("instances", []) if i.get("text")), None)
        bits = [f"{k+1}. {c['cue']}", f"[{c.get('category','?')}]",
                f"risk={c.get('risk_level','?')}", f"spec={c.get('geo_specificity','?')}"]
        if txt:
            bits.append(f"text=“{txt}”")
        if not c.get("maskable", True):
            bits.append("GLOBAL(not maskable)")
        if c.get("degenerate"):
            bits.append("DEGENERATE box")
        if not c.get("instances"):
            bits.append("NOT GROUNDED")
        lines.append("  ".join(bits))
    fig.text(0.02, 0.02, "\n".join(lines), ha="left", va="bottom", fontsize=8.2,
             family="DejaVu Sans")
    fig.subplots_adjust(bottom=0.04 + 0.028 * len(lines))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
