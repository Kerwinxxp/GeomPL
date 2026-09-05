"""LaMa 修复算子的目检图:原图 | 灰块遮蔽 | 修复 | 修复版等面积对照(各取一条线索)。

看两件事:(1) 线索区确实被移除,(2) 修复内容与周围连贯、无明显平铺/糊块伪影。
用法:cue_extract/.venv/Scripts/python.exe -m belief_elicit.plot_inpaint_check
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

from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.precompute_inpaint import cue_masks_of

plt.rcParams.update({"font.family": "DejaVu Sans"})
CACHE = os.path.join(os.path.dirname(__file__), "inpaint_cache")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
PREFIXES = ["158307292", "754780171", "370717727"]      # NY / Bled / Cuba 案例图
CUE_K = 0                                               # 每图看第 0 条线索


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ids = sorted(os.listdir(CACHE))
    picks = [next(i for i in ids if i.startswith(p)) for p in PREFIXES]
    fig, axes = plt.subplots(len(picks), 4, figsize=(15, 3.6 * len(picks)))
    for r, iid in enumerate(picks):
        d = os.path.join(CACHE, iid)
        mf = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        W, H = mf["image_size"]
        _, _, masks, _ = cue_masks_of(os.path.join(ROOT, mf["src"]), iid)
        img = Image.open(os.path.join(ROOT, mf["image_path"])).resize((W, H)).convert("RGB")
        gray = mask_solid_from_masks(img, [masks[CUE_K]])
        panels = [(img, "original"),
                  (gray, "gray-fill (current)"),
                  (Image.open(os.path.join(d, f"s{CUE_K}.png")), "LaMa inpaint"),
                  (Image.open(os.path.join(d, f"c{CUE_K}-0.png")),
                   "inpaint, equal-area control")]
        cue = mf["cues"][CUE_K]
        for c, (im, ttl) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{iid[:9]}\n{cue['cue'][:22]}\n"
                              f"area={cue['area_frac']*100:.1f}%", fontsize=8)
    fig.suptitle("LaMa inpainting sanity check (cue 0 of each image)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGDIR, "inpaint_check.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
