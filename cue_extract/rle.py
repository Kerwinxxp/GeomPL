"""Minimal RLE (row-major boolean mask <-> {size, counts}), no pycocotools dependency.

counts = alternating run lengths starting from a "False" run (COCO uncompressed RLE
convention); the leading run may be 0 when the mask starts True.
"""
import numpy as np


def mask_to_rle(mask) -> dict:
    """Encode a boolean mask. Vectorised: a megapixel mask takes milliseconds, not 0.4 s."""
    m = np.asarray(mask, dtype=bool)
    flat = m.ravel(order="C")
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {"size": list(m.shape), "counts": [int(c) for c in counts]}


def rle_to_mask(rle) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos: pos + c] = True
        pos += c
        val = not val
    return flat.reshape(h, w)
