"""The two ablation primitives: solid-fill an irregular mask, and enumerate subsets.

- mask_solid_from_masks: replace the cue's own pixels with neutral gray (information
  removal with no hyper-parameter);
- nonempty_subsets: all 2^m - 1 non-empty subsets of m cues, ordered by (size,
  lexicographic) so the output order is deterministic and runs can resume per subset.

Library module — no CLI.
"""
from itertools import combinations

import numpy as np
from PIL import Image


def mask_solid_from_masks(image, masks, color=(128, 128, 128)):
    """Solid-fill the union of **irregular boolean masks** (not boxes) — the main ablation.

    masks: [np.bool_ (H, W), ...] at the image's size; masks of other shapes are ignored.
    Returns a new image; the input is not modified.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for m in masks or []:
        m = np.asarray(m, dtype=bool)
        if m.shape == (h, w):
            union |= m
    arr[union] = color
    return Image.fromarray(arr, mode="RGB")


def nonempty_subsets(m: int) -> list:
    """[(0,), (1,), ..., (0,1), ..., (0,...,m-1)]: ascending size, lexicographic within size."""
    return [s for size in range(1, m + 1) for s in combinations(range(m), size)]
