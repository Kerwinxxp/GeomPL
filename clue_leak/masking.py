"""噪声遮挡：对图像指定 bbox 区域加高斯噪声,破坏其中可定位的视觉线索。

框法②(clue-leak study)的核心操作:把识别出的线索区域用噪声覆盖 → 得到"去线索图"。
sigma(0–255 尺度)控制噪声强度,越大越彻底破坏内容。加噪后钳制到 [0,255]。
"""
import numpy as np


def mask_coverage(boxes, size) -> float:
    """被遮挡像素占全图比例(重叠只算一次)。size=(w,h)。"""
    w, h = size
    if w <= 0 or h <= 0 or not boxes:
        return 0.0
    covered = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, int(round(x1)))); x2 = max(0, min(w, int(round(x2))))
        y1 = max(0, min(h, int(round(y1)))); y2 = max(0, min(h, int(round(y2))))
        if x2 > x1 and y2 > y1:
            covered[y1:y2, x1:x2] = True
    return float(covered.sum()) / (w * h)


def add_noise_to_regions(image, boxes, sigma: float, seed: int):
    """返回新图(不就地修改)。boxes: [[x1,y1,x2,y2], ...]，越界自动裁剪。"""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64).copy()
    h, w = arr.shape[:2]
    if sigma > 0:
        rng = np.random.default_rng(seed)
        for box in boxes or []:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w, int(round(x1)))); x2 = max(0, min(w, int(round(x2))))
            y1 = max(0, min(h, int(round(y1)))); y2 = max(0, min(h, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            noise = rng.normal(0.0, sigma, size=(y2 - y1, x2 - x1, arr.shape[2]))
            arr[y1:y2, x1:x2, :] += noise
    from PIL import Image
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def mask_solid_regions(image, boxes, color=(128, 128, 128)):
    """纯色块屏蔽:bbox 内像素替换为纯色(默认中性灰),信息删除、无超参。

    子集消融实验(逐线索控制变量)的屏蔽操作。返回新图,不就地修改。
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    for box in boxes or []:
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, int(round(x1)))); x2 = max(0, min(w, int(round(x2))))
        y1 = max(0, min(h, int(round(y1)))); y2 = max(0, min(h, int(round(y2))))
        if x2 > x1 and y2 > y1:
            arr[y1:y2, x1:x2, :] = color
    from PIL import Image
    return Image.fromarray(arr, mode="RGB")


def mask_solid_from_masks(image, masks, color=(128, 128, 128)):
    """按**不规则布尔掩码**(而非方框)涂纯色:只覆盖线索本体像素。

    masks: [np.bool_ (H,W), ...],须与图同尺寸;逐掩码取并集后填色。返回新图,不就地修改。
    """
    import numpy as np
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for m in masks or []:
        m = np.asarray(m, dtype=bool)
        if m.shape == (h, w):
            union |= m
    arr[union] = color
    from PIL import Image
    return Image.fromarray(arr, mode="RGB")


def add_laplace_noise_to_regions(image, boxes, epsilon: float, seed: int,
                                 sensitivity: float = 255.0):
    """拉普拉斯机制(像素级 ε-DP):对区域每个像素通道值加 Lap(0, Δ/ε) 并钳制 [0,255]。

    ε 越小 → 尺度 b=Δ/ε 越大 → 噪声越强 → 越私密。Δ=255(8-bit 像素敏感度)。
    对"越近越好"效用,指数机制等价于此离散拉普拉斯。
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float64).copy()
    h, w = arr.shape[:2]
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    b = sensitivity / epsilon
    rng = np.random.default_rng(seed)
    for box in boxes or []:
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, int(round(x1)))); x2 = max(0, min(w, int(round(x2))))
        y1 = max(0, min(h, int(round(y1)))); y2 = max(0, min(h, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        noise = rng.laplace(0.0, b, size=(y2 - y1, x2 - x1, arr.shape[2]))
        arr[y1:y2, x1:x2, :] += noise
    from PIL import Image
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
