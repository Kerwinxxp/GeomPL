"""Visual sanity check of the LaMa operator: original | gray fill | inpaint | control.

Two things to look for: (1) the cue region really is gone, and (2) the fill is coherent with
its surroundings, with no obvious tiling or smearing artifacts. One cue per image.

Run: cue_extract/.venv/Scripts/python.exe -m belief_elicit.plot_inpaint_check
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

from belief_elicit.cues import cue_masks_of
from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.plotstyle import apply_style, save
from belief_elicit.results import FIGDIR, INPAINT_CACHE as CACHE

apply_style(hide_spines=False)     # image panels keep their frames
PREFIXES = ["158307292", "754780171", "370717727"]      # the NY / Bled / Cuba case images
CUE_K = 0                                               # look at cue 0 of each image


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
    save(fig, os.path.join(FIGDIR, "inpaint_check.png"), dpi=110)


if __name__ == "__main__":
    main()
