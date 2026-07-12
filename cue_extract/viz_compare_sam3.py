"""五图掩码质量对比:每行一张图,左=旧 5 阶段(results/),右=SAM3(results_sam3/)。
左右都把该管线的线索掩码叠到(高清)原图上,标题列出线索名。给导师看质量差。
用法：python -m cue_extract.viz_compare_sam3
"""
import glob
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

OLD = os.path.join(os.path.dirname(__file__), "results")
NEW = os.path.join(os.path.dirname(__file__), "results_sam3")
OUT = os.path.join(os.path.dirname(__file__), "figures", "compare_sam3_vs_old.png")

PILOT = [("181848051", "New York"), ("261517384", "Okazaki"), ("311344213", "New Delhi"),
         ("370717727", "Cuba"), ("847733166", "Venice")]


def load_subsets():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); out.setdefault(it["image_id"], it)
    return out


def overlay(ax, img, cues, title):
    base = np.asarray(img.convert("RGB")).astype(float).copy()
    H, W = base.shape[:2]
    names = []
    for k, c in enumerate(cues):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        painted = False
        for inst in c.get("instances", []):
            if inst.get("degenerate") or not inst.get("mask_rle"):
                continue
            m = rle_to_mask(inst["mask_rle"])
            if m.shape == (H, W):
                base[m] = 0.5 * base[m] + 0.5 * col
                painted = True
        tag = c["cue"][:22] + ("" if painted else " (∅/degen)")
        names.append(f"{k+1}. {tag}")
    ax.imshow(base.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=3)
    ax.text(0.01, -0.02, "\n".join(names), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.2, family="DejaVu Sans")


def main():
    subset = load_subsets()
    fig, axes = plt.subplots(len(PILOT), 2, figsize=(11, 4.3 * len(PILOT)))
    for row, (pref, place) in enumerate(PILOT):
        iid = next(k for k in subset if k.startswith(pref))
        img = Image.open(subset[iid]["path"] if os.path.isabs(subset[iid]["path"])
                         else os.path.join(ROOT, subset[iid]["path"]))
        for col, (d, tag) in enumerate([(OLD, "OLD 5-stage"), (NEW, "SAM3")]):
            f = os.path.join(d, iid + ".json")
            ax = axes[row, col]
            if not os.path.exists(f):
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{place} — {tag} (missing)")
                continue
            rec = json.load(open(f, encoding="utf-8"))
            im = img.resize(tuple(rec["image_size"])) if rec.get("image_size") else img
            overlay(ax, im, rec["geo_privacy_cues"],
                    f"{place} — {tag}  ({len(rec['geo_privacy_cues'])} cues)")
    fig.suptitle("Mask quality: old 5-stage pipeline  vs  route-B + SAM3 (hi-res)", fontsize=13, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=115)
    print("saved", OUT)


if __name__ == "__main__":
    main()
