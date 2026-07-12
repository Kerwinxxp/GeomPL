"""路线 B:攻击者自报线索。一次 GPT-4o grounded 定位推理 →
{location_guess, cues:[{name, category, bbox, reasoning, confidence}]}。

cue 来自模型**推理时实际使用**的证据(不是旁路检测器提议),天然有语义、对准隐私课题。
bbox 在 client.prepare 后的像素空间(与 vision_json 的 {width}/{height} 一致)。
"""
import json

from . import prompts


def locate_and_ground(client, image) -> dict:
    """调 GPT-4o → {"location_guess": str, "cues": [解析后的线索]}。"""
    prompt = prompts.GROUNDED_LOCATE.replace(
        "{categories}", json.dumps(prompts.CATEGORIES, ensure_ascii=False))
    raw = client.vision_json(prompt, image)          # 内部 prepare + 填 {width}/{height}
    return {"location_guess": raw.get("location_guess", ""),
            "cues": parse_cues(raw.get("cues", []))}


def parse_cues(raw_cues) -> list:
    """容错解析 → [{cue, category, is_text, bbox, reasoning, confidence}]。丢弃无 name/bbox 的。"""
    text_cats = {"text/signage"}
    out = []
    for e in raw_cues or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or e.get("cue")
        bbox = e.get("bbox")
        if not name or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            bbox = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        cat = str(e.get("category", "other"))
        sq = e.get("segment_query")
        out.append({
            "cue": str(name), "category": cat,
            "is_text": cat in text_cats,
            "segment_query": str(sq).strip() if sq else "",   # SAM3 用的物体名词
            "bbox": bbox,
            "reasoning": str(e.get("reasoning", "")),
            "confidence": float(e.get("confidence", 0.0) or 0.0),
        })
    return out
