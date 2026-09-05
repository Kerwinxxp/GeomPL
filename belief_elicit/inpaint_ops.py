"""LaMa inpainting as the "information removal" operator, in place of solid gray blocks.

Gray fill introduces a large tampering artifact: the equal-area control shows that graying
out an irrelevant region also moves the posterior a lot. LaMa fills the removed region with
plausible image content, leaving almost no visible tampering trace, so it is closer to the
counterfactual "this cue was never in the picture".

The interface mirrors belief_elicit.masking.mask_solid_from_masks:
    inpaint_from_masks(img, [mask, ...]) -> PIL.Image (RGB, same size)

Library module — no CLI.
"""
import numpy as np
from PIL import Image

DILATE_PX = 5          # mask dilation: give LaMa a clean border (no cue edge left behind)
_LAMA = None           # module-level lazy singleton


def get_lama():
    """Load and cache LaMa (torchscript big-lama; weights come from the torch.hub cache)."""
    global _LAMA
    if _LAMA is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LAMA = SimpleLama(device=dev)
    return _LAMA


def union_of(masks, shape):
    """Union several boolean masks into one (H, W) mask; masks of other sizes are ignored."""
    h, w = shape
    u = np.zeros((h, w), dtype=bool)
    for m in masks or []:
        m = np.asarray(m, dtype=bool)
        if m.shape == (h, w):
            u |= m
    return u


def dilate(mask, px=DILATE_PX):
    """Morphological dilation by px pixels (elliptical kernel) so the fill covers cue edges."""
    if px <= 0:
        return np.asarray(mask, dtype=bool)
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    d = cv2.dilate(np.asarray(mask, dtype=np.uint8), k, iterations=1)
    return d.astype(bool)


def inpaint_from_masks(image, masks, dilate_px=DILATE_PX):
    """Inpaint the union of **irregular boolean masks**: union -> dilate -> LaMa.

    masks: [np.bool_ (H, W), ...] at the image's size. An empty union returns an RGB copy.
    Returns a new image; the input is not modified.
    """
    img = image.convert("RGB")
    w, h = img.size
    u = dilate(union_of(masks, (h, w)), dilate_px)
    if not u.any():
        return img.copy()
    m = Image.fromarray((u.astype(np.uint8) * 255), mode="L")   # 255 = region to repair
    out = get_lama()(img, m)
    # LaMa pads its input to a multiple of 8, so the output can be larger -> crop back
    if out.size != (w, h):
        out = out.crop((0, 0, w, h))
    return out.convert("RGB")
