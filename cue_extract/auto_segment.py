"""Set-of-Mark 步骤一:SAM 自动分割整图 → 若干候选区域(过滤噪声/背景)。

用 transformers 的 mask-generation pipeline(SAM)。返回按面积降序的区域列表,
每个 {mask(bool HxW), area_frac, bbox, point(内部代表点)}。懒加载单例。
"""
import numpy as np

_GEN = None
MODEL_ID = "facebook/sam-vit-base"


def _load(device=0):
    global _GEN
    if _GEN is None:
        from transformers import pipeline
        _GEN = pipeline("mask-generation", model=MODEL_ID, device=device)
    return _GEN


def _interior_point(mask):
    """区域内一个稳健代表点(距离变换峰值 → 一定在区域内部,适合放编号)。"""
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(mask)
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return int(x), int(y)


def segment_regions(image, min_area=0.008, max_area=0.45, max_regions=16,
                    iou_dedup=0.7, device=0):
    """→ [{mask, area_frac, bbox:[x1,y1,x2,y2], point:[x,y]}],按面积降序。
    过滤:丢 <min_area(噪声)与 >max_area(背景/天空);重叠 IoU>iou_dedup 的去重(留大的);
    最多 max_regions 个(编号太多会干扰 VLM)。"""
    gen = _load(device)
    out = gen(image, points_per_side=24, pred_iou_thresh=0.86, stability_score_thresh=0.9)
    H, W = image.height, image.width
    tot = float(H * W)
    cand = []
    for m in out["masks"]:
        m = np.asarray(m, dtype=bool)
        a = m.sum() / tot
        if a < min_area or a > max_area:
            continue
        ys, xs = np.nonzero(m)
        cand.append({"mask": m, "area_frac": float(a),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    cand.sort(key=lambda c: -c["area_frac"])
    # IoU 去重:大的优先,压掉与它高度重叠的小的
    kept = []
    for c in cand:
        dup = False
        for k in kept:
            inter = (c["mask"] & k["mask"]).sum()
            if inter and inter / float(c["mask"].sum()) > iou_dedup:
                dup = True
                break
        if not dup:
            kept.append(c)
        if len(kept) >= max_regions:
            break
    for c in kept:
        c["point"] = list(_interior_point(c["mask"]))
    return kept
