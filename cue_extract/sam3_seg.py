"""SAM 3(facebook/sam3):文字短语 → 干净实例掩码。

给一个概念短语(如 "red sandstone minaret"),分割出图中该概念的所有实例。
用来替换旧的 Grounding DINO + SAM 定位步骤:route B 的 GPT-4o 出线索**语义名**,
SAM3 按名字出**尽可能完美的区域**。懒加载单例,需 GPU/torch。
"""
import numpy as np

_MODEL = None
_PROC = None
MODEL_ID = "facebook/sam3"


def _load(device="cuda"):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import Sam3Model, Sam3Processor
        _PROC = Sam3Processor.from_pretrained(MODEL_ID)
        _MODEL = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    return _MODEL, _PROC


def segment_phrase(image, phrase, threshold=0.5, max_instances=8, device="cuda"):
    """image: PIL RGB;phrase: 概念短语 → [{mask(bool HxW), score, bbox[x1,y1,x2,y2]}]。
    按分数降序;无命中返回 []。"""
    import torch
    model, proc = _load(device)
    H, W = image.height, image.width
    inp = proc(images=image, text=phrase, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_instance_segmentation(
        out, threshold=threshold, target_sizes=[(H, W)])[0]
    masks = res.get("masks")
    scores = res.get("scores")
    boxes = res.get("boxes")
    if masks is None or len(masks) == 0:
        return []
    scores = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
    boxes = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else list(boxes)
    items = []
    for i in range(len(masks)):
        m = masks[i]
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        m = np.asarray(m).astype(bool)
        if m.shape != (H, W):                       # 保险:尺寸对齐
            from PIL import Image as _I
            m = np.asarray(_I.fromarray(m).resize((W, H))).astype(bool)
        items.append({"mask": m, "score": float(scores[i]),
                      "bbox": [float(v) for v in boxes[i]]})
    items.sort(key=lambda d: -d["score"])
    return items[:max_instances]


def _variants(*phrases):
    """按优先级生成候选查询:给定短语 + 轻量形态变体(去冠词、单数、去尾复数),去重保序。"""
    import re
    seen, out = set(), []
    for p in phrases:
        p = (p or "").strip()
        if not p:
            continue
        cands = [p, re.sub(r"^(a |an |the )", "", p, flags=re.I).strip()]
        if p.lower().endswith("s"):
            cands.append(p[:-1])                       # 复数→单数粗略
        cands.append(p.split()[-1])                    # 末词(核心名词兜底)
        for c in cands:
            c = c.strip()
            if c and c.lower() not in seen:
                seen.add(c.lower()); out.append(c)
    return out


def segment_with_fallback(image, segment_query, name, threshold=0.3, device="cuda"):
    """召回增强:依次试 segment_query → name → 形态变体,首个非空命中即返回。
    返回 (instances, used_query);全空返回 ([], None)。"""
    for q in _variants(segment_query, name):
        r = segment_phrase(image, q, threshold=threshold, device=device)
        if r:
            return r, q
    return [], None
