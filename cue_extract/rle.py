"""极简 RLE(行优先布尔掩码 ↔ {size, counts}),无 pycocotools 依赖。

counts = 交替段长,从"False 段"开始(与 COCO uncompressed RLE 同约定)。
"""
import numpy as np


def mask_to_rle(mask) -> dict:
    m = np.asarray(mask, dtype=bool).ravel(order="C")
    counts, prev, run = [], False, 0
    for v in m:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = v, 1
    counts.append(run)
    return {"size": list(np.asarray(mask).shape), "counts": counts}


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
