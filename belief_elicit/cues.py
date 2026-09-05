"""Cue masks and the equal-area control placement — the pixel-level primitives.

`cue_masks_of` is the single reader for the SAM 3 cue records (it delegates to
`cue_extract.common.cue_masks`, so extraction and analysis can never drift apart);
`translate_mask` / `sample_control` build the equal-area control that measures the
masking-artifact floor; `mask_to_rle` is re-exported from `cue_extract.rle`.

Library module — no CLI.
"""
import os

import numpy as np

from cue_extract.common import cue_masks
from cue_extract.rle import mask_to_rle          # noqa: F401  (re-exported encoder)


def cue_masks_of(src, image_id):
    """`<src>/<image_id>.json` -> (cues, categories, masks, (W, H)); None if missing.

    Filtering rule in `cue_extract.common.cue_masks`: maskable cues only, non-degenerate
    instances with a mask_rle, union of the size-matching masks, non-empty union.
    """
    return cue_masks(os.path.join(src, image_id + ".json"))


def cue_unions(src, image_id):
    """Same source, reduced to (masks, (W, H)) — for overlays that only need the pixels."""
    got = cue_masks_of(src, image_id)
    if got is None:
        return None
    _, _, masks, size = got
    return masks, size


def translate_mask(mask, dx, dy):
    """Shift a boolean mask by (dx, dy); pixels leaving the frame are dropped."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    out[ny[ok], nx[ok]] = True
    return out


def sample_control(cue_mask, cue_union, rng, tries=40):
    """Random translation minimising overlap with the cue union, losing <10% of the area.

    Returns (mask, overlap_frac); (None, None) if no placement kept enough area.
    """
    H, W = cue_mask.shape
    ys, xs = np.nonzero(cue_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    area = int(cue_mask.sum())
    best, best_bad = None, 10 ** 18
    for _ in range(tries):
        dx = int(rng.integers(-x0, W - 1 - x1)) if x1 - x0 < W - 1 else 0
        dy = int(rng.integers(-y0, H - 1 - y1)) if y1 - y0 < H - 1 else 0
        t = translate_mask(cue_mask, dx, dy)
        if t.sum() < 0.9 * area:
            continue
        ov = int((t & cue_union).sum())
        if ov < best_bad:
            best_bad, best = ov, t
        if ov == 0:
            break
    return best, (best_bad / max(area, 1) if best is not None else None)
