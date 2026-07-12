"""cue_extract 的 LLM prompt(route-B + SAM3 管线)。"""

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
