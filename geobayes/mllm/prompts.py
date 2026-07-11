"""v0 草稿 prompt（全部 [my proposed assumption]，论文未公开任何 prompt）。

设计要点见 docs/paper_to_code_map.md §3：
- si ∈ [0,1] 独立打分 + 语言锚点（跨越 0.6 截断线）
- judge 只见假设标签，不见概率；中性路径 c=3；抗确认偏误措辞
- bbox [x1,y1,x2,y2] 绝对像素（图像已按 smart_resize 预缩放）
"""

HYPOTHESIZE = """You are a geo-localization analyst. Study the whole image and reason about WHERE it was taken.

Return ONLY a JSON object:
{{
  "level": "{level}",
  "scene_summary": "<one-sentence scene description>",
  "candidates": [
    {{"location": "<{unit}>", "confidence": <float 0-1>, "reason": "<short>"}}
  ],
  "verification_tasks": [
    {{"desc": "<what to examine, e.g. 'Examine the sign'>", "reason": "<what clue it may reveal>", "bbox": [x1, y1, x2, y2]}}
  ]
}}

Rules:
- List exactly {k} {unit_plural}, most plausible first (e.g. "{example}").
- Give the finest-grained candidates you can reasonably justify at the {level} level; use your full
  world knowledge of specific places, do NOT retreat to a coarser guess than asked.
- "confidence" is YOUR OWN independent belief for each candidate (do NOT normalize across candidates).
  Anchors: 0.9 = near-certain, 0.6 = clearly plausible, 0.3 = a weak guess, 0.1 = a long shot.
- Propose 3-{max_tasks} verification_tasks targeting DISCRIMINATIVE visual objects
  (signs, license plates, vehicles, vegetation, architecture, road markings, utility infrastructure).
- bbox uses absolute pixel coordinates [x1, y1, x2, y2] on this image ({width}x{height}).
"""

# 粒度规格 [assumption]：单层可配，粒度与层级解耦（研究目标 = 同支撑先验↔后验）
GRANULARITY = {
    "country": {"unit": "country name in English",
                "unit_plural": "candidate countries", "example": "France"},
    "city": {"unit": "city or town, with its country",
             "unit_plural": "candidate cities/towns",
             "example": "San Francisco, United States"},
    "place": {"unit": "specific place (neighborhood, landmark, or street area), with city and country",
              "unit_plural": "candidate specific places",
              "example": "Marina Piccola, Capri, Italy"},
}

VERIFY = """You are examining a cropped region of a photo as part of a geo-localization investigation.
Task: {desc}
Why: {reason}

Describe what you actually see and any geographic clues it carries (language/script, brand, style, side of traffic, climate hints).
Do NOT guess the final location; just report the evidence.

Return ONLY a JSON object:
{{"observation": "<what you see and what it implies>", "geo_clues": ["<clue 1>", "<clue 2>"]}}
"""

JUDGE = """You are scoring one piece of evidence against COMPETING location hypotheses for geo-localization.

Evidence: {evidence}

Competing hypotheses: {labels}

For EACH hypothesis, rate how this evidence DISCRIMINATES it from the alternatives:
- "c": integer 1-5. 5 = evidence genuinely favors THIS hypothesis over the alternatives, 4 = weakly favors it over alternatives, 3 = carries no discriminating information for it, 2 = weakly argues against it relative to alternatives, 1 = strongly argues against it.
- "alpha": float 0-1, how specific and unambiguous the discrimination is. Anchors: 0.9 = distinctive, near-diagnostic detail (a script, a plate format, a unique object), 0.5 = a meaningful but imperfect regional cue, 0.2 = vague.

CRITICAL rule — discrimination, not compatibility:
- If the evidence is EQUALLY CONSISTENT with several hypotheses (e.g. "Mediterranean coastline" when the candidates are Italy/Greece/Spain), give ALL of those hypotheses c=3. Compatibility with everything = information about nothing.
- Reserve c=4/5 for hypotheses this evidence favors OVER the other candidates in this specific list.
- Deliberately look for ways the evidence ARGUES AGAINST currently plausible-looking hypotheses.
- If the evidence is unreadable or uninformative, use c=3 for every hypothesis.

Return ONLY a JSON object:
{{"ratings": {{"<hypothesis>": {{"c": <int>, "alpha": <float>}}, ...}}}}
Include every hypothesis exactly once, using the exact names given.
"""

SCORE_CANDIDATES = """You are a geo-localization expert scoring an image against a CLOSED SET of candidate locations.
Exactly one of these candidates is the true location of the image.

Candidates:
{candidates}

For EACH candidate, output a score in [0,1] = how well the image's visual evidence matches that candidate
(architecture, vegetation, signage/script, vehicles, climate, terrain). Score candidates INDEPENDENTLY;
they need NOT sum to 1. Use the full range: a strongly-matching candidate ~0.9, a clearly-wrong one ~0.05.
Do not default to uniform — discriminate based on what you actually see.

Return ONLY a JSON object:
{{"scores": {{"<candidate exactly as given>": <float 0-1>, ...}}}}
Include every candidate exactly once, using the exact strings given.
"""

PLAN_VERIFICATION = """You are planning what to inspect in an image to distinguish these candidate locations:
{candidates}

Propose 3-{max_tasks} verification tasks, each targeting a DISCRIMINATIVE visual object whose appearance
would differ across these specific candidates (signs/script, license plates, vehicles, vegetation,
architecture, road markings, utility infrastructure).

Return ONLY a JSON object:
{{"verification_tasks": [{{"desc": "<what to examine>", "reason": "<which candidates it separates>", "bbox": [x1, y1, x2, y2]}}]}}
bbox uses absolute pixel coordinates on this image ({width}x{height}).
"""

ZERO_SHOT = """You are a geo-localization expert. Look at this image and give your single best guess
for where it was taken, as specifically as you can (down to the street if possible).

This is a direct one-shot guess — no tools, no step-by-step verification. Just your best estimate.

Return ONLY a JSON object:
{{"country": "<country in English, or empty>",
  "city": "<city/town, or empty if unsure>",
  "street": "<street or specific place/landmark, or empty if unsure>"}}

Be as specific as your confidence allows; leave a field empty ("") rather than guessing wildly,
but DO provide the finest level you can reasonably justify.
"""

SUBHYPOTHESES = """You are narrowing a geo-localization from {parent} down to the {level} level.

You have committed to: {parent}.
Key visual evidence collected so far: {objects}
Web search results about this evidence:
{snippets}

Based ONLY on {parent} and the evidence/search results, propose candidate {level}-level locations WITHIN {parent}.

Return ONLY a JSON object:
{{"level": "{level}",
  "candidates": [{{"location": "<{level} name>", "confidence": <0-1 independent belief>}}],
  "verification_tasks": [{{"desc": "...", "reason": "...", "bbox": [x1,y1,x2,y2]}}]}}

Rules:
- List 2-5 candidates, most plausible first, each a real {level} inside {parent}.
- "confidence" in [0,1], judged independently (do NOT normalize).
- Propose verification_tasks targeting objects that would distinguish these {level} candidates.
"""

REPLACE = """You are revising a geo-localization analysis. All previously proposed candidates were verified and none is supported by the evidence.

Scene: {scene_summary}
Rejected candidates: {failed}
Key evidence collected so far: {memory}

Propose a FRESH set of candidates (different from the rejected ones) consistent with the key evidence, plus new verification tasks.

Return ONLY a JSON object with the same schema as before:
{{"level": "{level}", "scene_summary": "...", "candidates": [{{"location": "...", "confidence": <0-1>}}], "verification_tasks": [{{"desc": "...", "reason": "...", "bbox": [x1,y1,x2,y2]}}]}}
"""
