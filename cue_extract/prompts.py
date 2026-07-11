"""cue_extract 的 LLM prompt(PLAN §2①⑤)。用户拟稿为基础,结构化 checklist 输出。"""

CATEGORIES = [
    "text/signage",
    "landmarks/buildings",
    "road/infrastructure",
    "architecture",
    "environment",
    "commercial/cultural",
    "vehicles/license plates",
    "other",
]

PROPOSAL = """Identify all visual cues in this image that may help infer its geographic location.
Do NOT infer the final location. Focus on visible evidence only.

Go through EVERY category below one by one; leave a category's list empty if nothing applies:
1. "text/signage": street signs, store names, school names, license plates, phone numbers, postal codes
2. "landmarks/buildings": recognizable buildings, statues, bridges, campus buildings
3. "road/infrastructure": lane markings, traffic lights, road signs, utility poles, bus stops
4. "architecture": house style, roof, windows, urban layout
5. "environment": vegetation, terrain, mountains, coast, climate indicators
6. "commercial/cultural": regional chains, logos, public transport branding, clothing, festivals
7. "vehicles/license plates": vehicle models, plate style, driving side
8. "other": any other geo-informative visual details

For each cue output an object:
  "cue": short name of the cue (e.g. "street sign")
  "grounding_phrase": a SHORT English noun phrase for an open-vocabulary detector to locate it
                      (e.g. "blue street name sign"). Concrete and visual, no abstractions.
  "is_text": true if the cue's information is written text (signs, plates, banners)

Return ONLY JSON:
{{"cues": {{
  "text/signage": [...], "landmarks/buildings": [...], "road/infrastructure": [...],
  "architecture": [...], "environment": [...], "commercial/cultural": [...],
  "vehicles/license plates": [...], "other": [...]
}}}}"""

VERIFIER = """You are auditing geo-privacy cue annotations for one image.
For EACH cue below, assess how much it could help an attacker infer where the photo was taken.

Cues (JSON): {cues}

For each cue (keep the same order, same "cue" name) output:
  "risk_level": "high" | "medium" | "low"
      high: street/name signs, house numbers, license plates, school/store names, postal codes,
            famous landmarks
      medium: regional chains, transit branding, architecture style, road style, vegetation, climate
      low: generic sky, generic trees, generic cars, people, furniture
  "geo_specificity": "generic" | "regional" | "city-level" | "street-level"
  "searchability": "high" | "medium" | "low"  (could a web/image search pin it down?)
  "maskable": false if this cue is a GLOBAL/ambient property of the whole scene (architecture style,
              terrain, climate, general vegetation) that cannot be localized to a bounded region;
              true if it is a bounded object/region that could be covered by a patch
  "explanation": one sentence

Return ONLY JSON: {{"assessments": [ ... ]}}"""
