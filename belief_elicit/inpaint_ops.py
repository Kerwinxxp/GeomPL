"""LaMa 修复(inpainting)作为"信息删除"算子,替代纯灰块遮蔽。

灰块遮蔽会引入巨大的"篡改伪影":等面积对照实验显示,把一块无关区域涂灰
也能大幅改变后验。LaMa 用图像内容填补被删区域,几乎不留可见篡改痕迹,
因而更接近"该线索从未出现过"的反事实。

接口与 clue_leak.masking.mask_solid_from_masks 对齐:
    inpaint_from_masks(img, [mask, ...]) -> PIL.Image (RGB, 同尺寸)
"""
import numpy as np
from PIL import Image

DILATE_PX = 5          # 掩码外扩像素数:给 LaMa 一个干净的边界(避免线索边缘残留)
_LAMA = None           # 模块级懒加载单例


def get_lama():
    """加载并缓存 LaMa(torchscript big-lama,权重走 torch.hub 缓存)。"""
    global _LAMA
    if _LAMA is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LAMA = SimpleLama(device=dev)
    return _LAMA


def union_of(masks, shape):
    """把若干布尔掩码并成一个 (H, W) 布尔掩码;尺寸不符的忽略。"""
    h, w = shape
    u = np.zeros((h, w), dtype=bool)
    for m in masks or []:
        m = np.asarray(m, dtype=bool)
        if m.shape == (h, w):
            u |= m
    return u


def dilate(mask, px=DILATE_PX):
    """形态学膨胀 px 像素(椭圆核),让修复区完整盖住线索边缘。"""
    if px <= 0:
        return np.asarray(mask, dtype=bool)
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    d = cv2.dilate(np.asarray(mask, dtype=np.uint8), k, iterations=1)
    return d.astype(bool)


def inpaint_from_masks(image, masks, dilate_px=DILATE_PX):
    """按**不规则布尔掩码**修复:并集 → 膨胀 → LaMa 填补。返回新图,不就地修改。

    masks: [np.bool_ (H,W), ...],须与图同尺寸。空掩码时原样返回 RGB 副本。
    """
    img = image.convert("RGB")
    w, h = img.size
    u = dilate(union_of(masks, (h, w)), dilate_px)
    if not u.any():
        return img.copy()
    m = Image.fromarray((u.astype(np.uint8) * 255), mode="L")   # 255 = 待修复
    out = get_lama()(img, m)
    # LaMa 内部把输入 pad 到 8 的倍数,输出可能比原图大 → 裁回原尺寸
    if out.size != (w, h):
        out = out.crop((0, 0, w, h))
    return out.convert("RGB")
