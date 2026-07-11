"""三联对比:原图 / 方框涂灰(旧) / 不规则 SAM 掩码涂灰(新)。
说明"遮方框 vs 遮线索本体"的区别。用法：python -m cue_extract.viz_mask_compare <id_prefix>
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
from PIL import Image

from clue_leak.masking import mask_solid_from_masks, mask_solid_regions
from cue_extract.rle import rle_to_mask

INDIR = os.path.join(os.path.dirname(__file__), "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
SUBSET = os.path.join(ROOT, "data", "subset100.jsonl")


def main():
    pref = sys.argv[1] if len(sys.argv) > 1 else "158307292"
    rec = json.load(open(glob.glob(os.path.join(INDIR, pref + "*"))[0], encoding="utf-8"))
    subset = {json.loads(l)["image_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")}
    iid = rec["image_id"]
    img = Image.open(subset[iid]["path"]).resize(tuple(rec["image_size"])).convert("RGB")

    boxes, masks, names = [], [], []
    for c in rec["geo_privacy_cues"]:
        for inst in c.get("instances", []):
            if inst.get("degenerate") or not inst.get("mask_rle"):
                continue
            boxes.append(inst["bbox"])
            masks.append(rle_to_mask(inst["mask_rle"]))
            names.append(c["cue"])
    box_masked = mask_solid_regions(img, boxes)
    irr_masked = mask_solid_from_masks(img, masks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, im, t in zip(axes, [img, box_masked, irr_masked],
                         ["original", "bbox masked (rectangles)", "SAM-mask masked (irregular)"]):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=12)
    fig.suptitle(f"{iid.split('_')[0]} — masking {len(masks)} cue(s): {', '.join(names)}",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"mask_compare_{pref}.png")
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("saved", out, "| masks:", names)


if __name__ == "__main__":
    main()
