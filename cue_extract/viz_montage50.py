"""50 图线索标注质量拼图:抽 N 张,掩码叠回高清原图 + 每图线索名。
用法：python -m cue_extract.viz_montage50 [N]  (默认12,均匀抽样覆盖不同国家)
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

RES = os.path.join(os.path.dirname(__file__), "results_sam3")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    src = sys.argv[2] if len(sys.argv) > 2 else "data/subset50_hires.jsonl"
    rows = [json.loads(l) for l in open(os.path.join(ROOT, src), encoding="utf-8")]
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)      # 均匀抽样
    picks = [rows[i] for i in idx]
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.9 * rows_n))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, it in zip(axes, picks):
        f = os.path.join(RES, it["image_id"] + ".json")
        rec = json.load(open(f, encoding="utf-8"))
        p = it["path"] if os.path.isabs(it["path"]) else os.path.join(ROOT, it["path"])
        img = Image.open(p).resize(tuple(rec["image_size"])).convert("RGB")
        base = np.asarray(img).astype(float).copy()
        H, W = base.shape[:2]
        names = []
        for k, c in enumerate(rec["geo_privacy_cues"]):
            col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
            painted = False
            for ins in c.get("instances", []):
                if ins.get("degenerate") or not ins.get("mask_rle"):
                    continue
                m = rle_to_mask(ins["mask_rle"])
                if m.shape == (H, W):
                    base[m] = 0.45 * base[m] + 0.55 * col
                    painted = True
            mark = "" if painted else " (∅/degen)"
            names.append(f"{k+1}. {c['cue'][:22]}{mark}")
        ax.imshow(base.astype(np.uint8))
        ax.set_title(f"{it['gt_country']} — guess: {rec['location_guess'][:20]}", fontsize=8.5)
        ax.text(0.01, -0.01, "\n".join(names), transform=ax.transAxes, va="top", ha="left",
                fontsize=6.6, family="DejaVu Sans")
    fig.suptitle(f"SAM3 cue masks on hi-res subset (showing {n} of {len(rows)})", fontsize=13, y=1.002)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUT = os.path.join(os.path.dirname(__file__), "figures", f"montage{len(rows)}_sam3.png")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
