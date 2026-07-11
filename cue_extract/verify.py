"""⑤ Verifier:VLM 审计每条线索的风险/特异性/可遮蔽性(PLAN §2⑤)。"""
import json

from . import prompts


def verify_cues(client, image, cues: list) -> list:
    """给每条线索附加 {risk_level, geo_specificity, searchability, maskable, explanation}。

    LLM 按序返回 assessments;数量不匹配时尽量按序对齐,缺的给保守默认。
    """
    brief = [{"cue": c["cue"], "category": c["category"],
              "text": c.get("text") or next((i.get("text") for i in c.get("instances", [])
                                             if i.get("text")), None),
              "n_instances": len(c.get("instances", []))} for c in cues]
    raw = client.vision_json(
        prompts.VERIFIER.replace("{cues}", json.dumps(brief, ensure_ascii=False)), image)
    assess = raw.get("assessments", [])
    out = []
    defaults = {"risk_level": "medium", "geo_specificity": "regional",
                "searchability": "medium", "maskable": True,
                "explanation": "(verifier missing — defaults)"}
    for k, c in enumerate(cues):
        a = assess[k] if k < len(assess) and isinstance(assess[k], dict) else {}
        out.append({**c, **{key: a.get(key, defaults[key]) for key in defaults}})
    return out
