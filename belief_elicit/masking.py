"""消融所需的两个基元:不规则掩码涂纯色 + 非空子集枚举。

- mask_solid_from_masks:把线索本体像素替换为中性灰(信息删除,无超参);
- nonempty_subsets:对 m 条线索枚举全部 2^m-1 个非空子集,按 (大小, 字典序) 稳定排序 →
  输出顺序确定,结果可按子集缓存/续跑。
"""
from itertools import combinations

import numpy as np
from PIL import Image


def mask_solid_from_masks(image, masks, color=(128, 128, 128)):
    """按**不规则布尔掩码**(而非方框)涂纯色:只覆盖线索本体像素(主消融用)。

    masks: [np.bool_ (H,W), ...],须与图同尺寸;逐掩码取并集后填色。返回新图,不就地修改。
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
    """[(0,), (1,), ..., (0,1), ..., (0,...,m-1)]:大小升序、同大小字典序。"""
    return [s for size in range(1, m + 1) for s in combinations(range(m), size)]
