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

GROUNDED_LOCATE = """You are a geolocation analyst. Infer WHERE this photo was taken, and — most
importantly — expose the visual evidence you actually reason from.

Think evidence-first: list EVERY visual cue you genuinely use to narrow down the location.
Be thorough — include weak or partial cues, not only the decisive ones. Do NOT list objects
that are irrelevant to location (generic sky, plain people, furniture) unless they actually
inform your guess.

For each cue output an object:
  "name": concrete semantic name of the cue, specific
          (e.g. "blue street-name sign", "Orthodox church onion dome", "Cyrillic shopfront text",
           "left-hand-drive car", "Mediterranean pantile roof")
  "segment_query": a SHORT, canonical OBJECT NOUN that an open-vocabulary segmenter can find —
          the concrete physical thing, NOT a description. Prefer a common singular noun.
          e.g. name "Japanese text on festival banners" -> segment_query "banner";
               name "Cyrillic shopfront text" -> "storefront sign";
               name "Samurai armor on a person" -> "samurai warrior";
               name "Mediterranean pantile roof" -> "tiled roof".
          If the cue is a diffuse property (architecture style, climate) with no boundable object,
          set segment_query to the best physical proxy object or "" if none.
  "category": one of {categories}
  "bbox": [x1, y1, x2, y2] in pixels of a {width}x{height} image, tight around the cue
  "reasoning": what this cue tells you about the location and in which direction it points
               (continent / country / region / city) and why
  "confidence": 0.0-1.0 — how much your location guess relies on this cue

Then give your overall location guess.

Return ONLY JSON:
{{"location_guess": "your best guess (city, country or region)",
  "cues": [ {{"name": "...", "segment_query": "...", "category": "...", "bbox": [x1,y1,x2,y2],
              "reasoning": "...", "confidence": 0.0}} ]}}"""

SOM_LOCATE = """You are a geolocation analyst. The image has been pre-segmented into numbered,
colored regions. Infer WHERE this photo was taken, and expose the evidence you reason from by
REFERENCING THE NUMBERED REGIONS — do not invent coordinates.

Think evidence-first: go through the numbered regions and list every one you genuinely use as a
cue to narrow down the location. Be thorough — include weak or partial cues. Ignore regions that
carry no location information (plain sky, generic ground, anonymous people).

For each cue output an object:
  "region_id": the integer label of the region (must be one shown in the image)
  "name": concrete semantic name of what that region is
          (e.g. "Orthodox church onion dome", "Cyrillic shopfront text", "red sandstone minaret")
  "category": one of {categories}
  "reasoning": what this region tells you about the location and in which direction it points
  "confidence": 0.0-1.0 — how much your location guess relies on this region

If a single decisive cue spans two regions, list both. Then give your overall location guess.

Return ONLY JSON:
{{"location_guess": "city, country or region",
  "cues": [ {{"region_id": 1, "name": "...", "category": "...",
              "reasoning": "...", "confidence": 0.0}} ]}}"""

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
