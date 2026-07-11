"""Inpainting 移除线索(替代灰块):LaMa 大掩码修复,把线索区域填成合理中性内容,
让图保持在分布内 → 消除灰块 OOD 伪影。懒加载单例(需 GPU/torch)。
"""
import numpy as np

_LAMA = None


def _load(device="cuda"):
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama(device=device)
    return _LAMA


def inpaint_masks(image, masks, dilate: int = 7, device="cuda"):
    """image: PIL RGB;masks: [np.bool_(H,W),...] → 并集(可膨胀 dilate 像素)→ LaMa 修复。
    空掩码返回原图副本。"""
    from PIL import Image
    h, w = image.height, image.width
    union = np.zeros((h, w), dtype=bool)
    for m in masks or []:
        m = np.asarray(m, dtype=bool)
        if m.shape == (h, w):
            union |= m
    if not union.any():
        return image.convert("RGB").copy()
    if dilate > 0:
        from scipy.ndimage import binary_dilation
        union = binary_dilation(union, iterations=dilate)
    mask_img = Image.fromarray((union * 255).astype(np.uint8), mode="L")
    result = _load(device).__call__(image.convert("RGB"), mask_img)
    return result.convert("RGB").resize((w, h))     # LaMa 可能改尺寸,还原
