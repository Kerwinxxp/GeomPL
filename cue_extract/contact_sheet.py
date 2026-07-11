"""把 cue_extract/results 里所有图渲成紧凑缩略图(mask 叠加+编号),拼成 contact sheet。
每格标题 = id前缀 · #cues(g=grounded) · 最高风险。用法(需图片,建议 venv 也可主 env):
  python -m cue_extract.contact_sheet
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cue_extract.rle import rle_to_mask
from cue_extract.viz import COLORS

INDIR = os.path.join(os.path.dirname(__file__), "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
SUBSET = os.path.join(ROOT, "data", "subset100.jsonl")
RISK_RANK = {"high": 3, "medium": 2, "low": 1}


def thumb(ax, img, cues):
    overlay = np.asarray(img.convert("RGB")).astype(float).copy()
    for k, c in enumerate(cues):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        for inst in c.get("instances", []):
            if inst.get("mask_rle"):
                m = rle_to_mask(inst["mask_rle"])
                if m.shape == overlay.shape[:2]:
                    overlay[m] = 0.5 * overlay[m] + 0.5 * col
    ax.imshow(overlay.astype(np.uint8))
    for k, c in enumerate(cues):
        col = COLORS[k % len(COLORS)]
        for inst in c.get("instances", []):
            if inst.get("degenerate"):
                continue
            x1, y1, x2, y2 = inst["bbox"]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=col, linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])


def main():
    subset = {json.loads(l)["image_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")}
    files = sorted(f for f in os.listdir(INDIR) if f.endswith(".json"))
    recs = [json.load(open(os.path.join(INDIR, f), encoding="utf-8")) for f in files]
    n = len(recs)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_2d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, r in zip(axes, recs):
        iid = r["image_id"]
        img = Image.open(subset[iid]["path"])
        # 需与提取时同尺寸:结果里存了 image_size
        if r.get("image_size") and tuple(img.size) != tuple(r["image_size"]):
            img = img.resize(tuple(r["image_size"]))
        cues = r["geo_privacy_cues"]
        thumb(ax, img, cues)
        ng = sum(1 for c in cues if any(not i.get("degenerate") for i in c.get("instances", [])))
        risks = [c.get("risk_level", "low") for c in cues]
        top = max(risks, key=lambda x: RISK_RANK.get(x, 0)) if risks else "-"
        gt = subset[iid].get("gt_country", "")
        ax.set_title(f"{iid.split('_')[0]} · {gt[:16]}\n{len(cues)} cues (g={ng}) · risk:{top}",
                     fontsize=7.5, pad=3)
    fig.suptitle(f"Cue-extraction contact sheet — {n} images (mask overlay + grounded boxes)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "contact_sheet.png")
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"saved {out} ({n} imgs)")


if __name__ == "__main__":
    main()
