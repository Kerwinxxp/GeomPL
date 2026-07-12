"""纯色块遮蔽:把线索区域替换为中性灰(信息删除,无超参),供逐线索 mPL 消融。"""
import numpy as np
from PIL import Image


def mask_solid_regions(image, boxes, color=(128, 128, 128)):
    """按**方框**涂纯色(默认中性灰)。返回新图,不就地修改。boxes: [[x1,y1,x2,y2],...]。"""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    for box in boxes or []:
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, int(round(x1)))); x2 = max(0, min(w, int(round(x2))))
        y1 = max(0, min(h, int(round(y1)))); y2 = max(0, min(h, int(round(y2))))
        if x2 > x1 and y2 > y1:
            arr[y1:y2, x1:x2, :] = color
    return Image.fromarray(arr, mode="RGB")


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
