# cue_extract — Geolocation Cue Extraction Pipeline

Extracts "visual cues that may leak geographic location" from an image and produces **pixel-level masks**. Feeds clean cue regions to the downstream per-cue mPL ablation (`clue_leak/`).

## Pipeline: route-B + SAM 3 (`run_extract_sam3.py`)

Two models, each doing what it is good at:

```
① GPT-4o (clean image)  → geo-reasoning: names the location cues it actually uses,
                            + segment_query (a concrete object noun), reasoning, confidence
② SAM 3 (facebook/sam3) → segments each cue by its text phrase → precise instance masks
③ flag_degenerate + assign_maskable
```
GPT-4o does the semantics ("what / why"); SAM 3 does the localization ("where", pixel-precise). `grounded.py` parses the VLM output; `sam3_seg.segment_with_fallback` tries `segment_query → name → morphological variants` for high recall.

**Prerequisites:**
- Use **high-resolution images** (see root README — the sample images ship at 1024px; low-res 500px badly hurts recall). `client.prepare` keeps up to ~1MP.
- **SAM 3 is a gated model.** Accept the license once at <https://huggingface.co/facebook/sam3> and `huggingface-cli login`; then `transformers>=5.13` loads it natively (`Sam3Model`, ~840M params, ~3.4 GB VRAM).

```bash
cue_extract/.venv/Scripts/python -m cue_extract.run_extract_sam3 --ids <id1,id2,...>
#   → cue_extract/results_sam3/<id>.json  (used by clue_leak.run_combo2 --cue_dir)
cue_extract/.venv/Scripts/python -m cue_extract.contact_sheet     # batch QA overview
```

## Modules

| file | role |
|---|---|
| `run_extract_sam3.py` | orchestrator (GPT-4o → SAM 3 → degenerate/maskable) |
| `grounded.py` | parse the VLM geo-reasoning output into cues |
| `sam3_seg.py` | SAM 3 text→mask + `segment_with_fallback` recall chain |
| `merge.py` | `flag_degenerate` (bbox > 40% img → non-maskable) · `assign_maskable` (evidence-based) |
| `rle.py` | minimal mask RLE encode/decode (no pycocotools) |
| `viz.py` · `contact_sheet.py` | annotation overlay · batch QA sheet |
| `viz_compare_sam3.py` | old-pipeline vs SAM 3 mask-quality comparison figure |
| `inpaint.py` · `viz_mask_compare.py` | gray-mask vs LaMa-inpaint robustness check |

## Output JSON (`cue_extract/results_sam3/<id>.json`)

```json
{
  "image_id": "...", "image_size": [W, H], "location_guess": "City, Country",
  "geo_privacy_cues": [{
    "cue": "Qutub Minar", "segment_query": "tower", "category": "landmarks/buildings",
    "is_text": false, "reasoning": "...", "confidence": 0.9,
    "used_query": "tower",
    "instances": [{"bbox": [x1,y1,x2,y2], "score": 0.97, "source": "sam3",
                   "mask_rle": {"size": [H,W], "counts": [...]}, "degenerate": false}],
    "maskable": true, "degenerate": false
  }]
}
```
A cue GPT-4o reports but SAM 3 cannot localize is kept with `instances: []` and `maskable: false` (honest — reported but not localized), not silently dropped.

## Known limitations
- SAM 3 is sensitive to phrasing; `segment_query` + the fallback chain mitigate but rare cues can still miss (kept as un-localized);
- diffuse whole-scene properties (architecture style, climate) have no boundable object → segmented as a large region and flagged `degenerate` (non-maskable), which is correct;
- an image that is "one landmark filling the whole frame" yields no maskable cue and is excluded from the ablation.
