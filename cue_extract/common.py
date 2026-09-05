"""Shared helpers for the cue-extraction stack: image manifests and cue-mask reading.

`load_subsets` is used by every extractor (run_extract_sam3 / extract_vocab); `cue_masks`
is the single reader for the `<image_id>.json` cue records both extractors write, and is
re-exported to the analysis layer as `belief_elicit.cues.cue_masks_of`.

Library module — no CLI.  SAM 3 model loading lives in `cue_extract.sam3_seg`.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_sam3")
VOCABDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_vocab")


def load_subsets():
    """data/subset*.jsonl -> {image_id: item}.

    Sorted glob, later files overwrite earlier ones (so the hi-res `subset_sample`
    entries win) — every caller depends on exactly this order.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            out[it["image_id"]] = it
    return out


def cue_masks(path):
    """Read one cue-record JSON -> (cues, categories, masks, (W, H)); None if missing.

    Filtering rule (the one every consumer of these records uses): keep cues marked
    `maskable`, drop instances flagged `degenerate` or lacking a `mask_rle`, union the
    remaining instance masks that match the record's image size, and drop the cue when
    that union is empty.  Categories are returned raw (callers apply their own default).
    """
    from cue_extract.rle import rle_to_mask
    if not os.path.exists(path):
        return None
    rec = json.load(open(path, encoding="utf-8"))
    W, H = rec["image_size"]
    cues, cats, masks = [], [], []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        good = [i for i in c["instances"] if not i.get("degenerate") and i.get("mask_rle")]
        if not good:
            continue
        u = np.zeros((H, W), bool)
        for i in good:
            m = rle_to_mask(i["mask_rle"])
            if m.shape == (H, W):
                u |= m
        if not u.any():
            continue
        cues.append(c["cue"]); cats.append(c.get("category")); masks.append(u)
    return cues, cats, masks, (W, H)
