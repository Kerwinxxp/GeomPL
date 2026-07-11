"""① Cue Proposal:VLM 看整图列线索,不画框(PLAN §2①)。"""
from . import prompts


def parse_proposal(raw: dict) -> list:
    """LLM 输出 → 扁平线索列表 [{cue, category, grounding_phrase, is_text}]。

    容错:cues 可为 {category: [entries]}(checklist 形态,首选)或扁平 list
    (entry 自带 category);缺 cue/grounding_phrase 的条目跳过;is_text 默认 False。
    """
    cues = raw.get("cues", {})
    out = []
    if isinstance(cues, dict):
        pairs = [(cat, e) for cat, entries in cues.items()
                 for e in (entries or []) if isinstance(e, dict)]
    else:
        pairs = [(e.get("category", "other"), e) for e in (cues or []) if isinstance(e, dict)]
    for cat, e in pairs:
        cue, phrase = e.get("cue"), e.get("grounding_phrase")
        if not cue or not phrase:
            continue
        out.append({"cue": str(cue), "category": str(cat),
                    "grounding_phrase": str(phrase), "is_text": bool(e.get("is_text", False))})
    return out


def propose_cues(client, image) -> list:
    """调 VLM(client.vision_json)→ 解析后的线索列表。"""
    return parse_proposal(client.vision_json(prompts.PROPOSAL, image))
