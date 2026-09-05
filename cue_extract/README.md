# cue_extract — Geolocation Cue Extraction Pipeline

Extracts "visual cues that may leak geographic location" from an image and produces **pixel-level masks**. Feeds clean cue regions to the downstream per-cue mPL ablation and Shapley attribution (`belief_elicit/`).

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
- `OPENAI_API_KEY` must be in the environment — the client never reads a key from code or config. Everything else (model, endpoint, `max_pixels`, cache dir) comes from the repo-root `config.yaml`.

```bash
cue_extract/.venv/Scripts/python -m cue_extract.run_extract_sam3 --ids <id1,id2,...>
#   → cue_extract/results_sam3/<id>.json  (consumed by belief_elicit/run_georanker_*.py)
cue_extract/.venv/Scripts/python -m cue_extract.viz_montage50 12    # batch QA montage
```

## Bottom-up variant: fixed vocabulary (`extract_vocab.py`)

The same SAM 3 segmentation driven by a **fixed ~12-concept vocabulary** instead of GPT-4o's per-image proposals: deterministic, free, and identical across images, so masks are directly comparable image-to-image. `compare_vocab.py` measures how much of the GPT-4o cue list the vocabulary covers (IoU / pixel recall); the leakage-side comparison lives in `belief_elicit/vocab_vs_gpt4o.py`.

```bash
cue_extract/.venv/Scripts/python -m cue_extract.extract_vocab   # → cue_extract/results_vocab/<id>.json
cue_extract/.venv/Scripts/python -m cue_extract.compare_vocab
cue_extract/.venv/Scripts/python -m cue_extract.viz_vocab_vs_gpt4o
```

## Modules

**Library** (no CLI; imported by the runners and by `belief_elicit/`)

| file | role |
|---|---|
| `common.py` | `load_subsets` (the `data/subset*.jsonl` manifest) · `cue_masks` — **the single reader** for the cue-record JSON below (maskable cue, union of non-degenerate instances, non-empty union). Re-exported to the analysis layer as `belief_elicit.cues.cue_masks_of`. |
| `rle.py` | minimal mask RLE encode/decode (no pycocotools) — the single `mask_to_rle`, also used by `belief_elicit/` |
| `prompts.py` | the geo-reasoning prompt and the cue category list |
| `grounded.py` | parse the VLM geo-reasoning output into cues |
| `sam3_seg.py` | SAM 3 text→mask + `segment_with_fallback` recall chain (the only place SAM 3 is loaded) |
| `merge.py` | `flag_degenerate` (bbox > 40% img → non-maskable) · `assign_maskable` (evidence-based) |
| `mllm.py` | GPT-4o client (OpenAI-compatible), disk-cached responses, key from `OPENAI_API_KEY` |
| `imaging.py` | `smart_resize_dims` — model coordinate space == our pixel space |
| `viz.py` | `render` — annotation overlay for a single image |

**Runners and reports** (each is a `python -m cue_extract.<name>` entry point)

| file | role |
|---|---|
| `run_extract_sam3.py` | orchestrator (GPT-4o → SAM 3 → degenerate/maskable) + config loader; `extract_one` |
| `extract_vocab.py` | fixed-vocabulary arm; `extract_vocab_cues` |
| `compare_vocab.py` | how much of the GPT-4o cue list the vocabulary covers (IoU / pixel recall) |
| `stats_cues.py` | dataset-wide cue statistics figure (categories, counts, localization outcomes) |
| `viz_montage50.py` | multi-image annotation-quality montage |
| `viz_vocab_vs_gpt4o.py` | side-by-side GPT-4o vs vocabulary mask overlays |

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
