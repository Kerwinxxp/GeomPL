# cue_extract — Geolocation Cue Extraction Pipeline

Extracts "visual cues that may leak geographic location" from an image and produces **pixel-level masks**. Feeds clean cue regions to the downstream per-cue mPL ablation (`clue_leak/`).

## ⭐ Recommended pipeline: route-B + SAM 3 (`run_extract_sam3.py`)

The current best extractor. Two models, each doing what it is good at:

```
① GPT-4o (clean image)  → geo-reasoning: names the location cues it actually uses,
                            + segment_query (a concrete object noun), reasoning, confidence
② SAM 3 (facebook/sam3) → segments each cue by its text phrase → precise instance masks
③ flag_degenerate + assign_maskable
```
GPT-4o does the semantics ("what / why"); SAM 3 does the localization ("where", pixel-precise). This fixed the failure modes of the earlier detector-based pipeline (mis-localized boxes, blurry masks, missed small cues). `grounded.py` parses the VLM output; `sam3_seg.segment_with_fallback` tries `segment_query → name → morphological variants` for high recall.

**Prerequisites:**
- Use **high-resolution images** (see root README — the sample images ship at 1024px; low-res 500px badly hurts recall). `client.prepare` keeps up to ~1MP.
- **SAM 3 is a gated model.** Accept the license once at <https://huggingface.co/facebook/sam3> and `huggingface-cli login`; then `transformers>=5.13` loads it natively (`Sam3Model`, ~840M params, ~3.4 GB VRAM).

```bash
cue_extract/.venv/Scripts/python -m cue_extract.run_extract_sam3 --ids <id1,id2,...>
#   → cue_extract/results_sam3/<id>.json  (same schema as below; used by clue_leak.run_combo2 --cue_dir)
```

`run_extract_som.py` (Set-of-Mark: SAM auto-seg + numbered marks) and `run_extract_b.py` (VLM draws boxes) are earlier variants kept for reference — SAM 3 supersedes both.

---

## Legacy pipeline: detector-based 5-stage (`run_extract.py`)

The original approach (kept for the archived comparison). Extracts cues and masks via a fixed 5-stage detector chain.

## Five stages (`run_extract.extract_one`)

```
input image
 ├─① proposal.py   GPT-4o lists candidate cues by 8-category checklist (says "what", no boxes)
 ├─② grounding.py  Grounding DINO turns each cue phrase → bbox (multiple instances allowed)
 ├─③ ocr.py        RapidOCR (PP-OCRv5 server) reads text → {text, bbox}, merged into cues by IoU
 ├─④ sam_mask.py   SAM 2.1: bbox → pixel-level mask (RLE)
 └─⑤ verify.py     GPT-4o audit: risk / geo_specificity / maskable / degenerate
```

`merge.py` handles the merge logic:
- `prune_uncorroborated_text_boxes` — a text cue's box must be corroborated by OCR (else it is a grounding hallucination and is dropped)
- `flag_degenerate` — an instance bbox > 40% of the image is flagged non-maskable (fixes "cue fills the whole frame")
- `assign_maskable` — maskability is decided by evidence: only cues with a non-degenerate real box are maskable

`rle.py` is a minimal RLE encode/decode (no pycocotools dependency). `viz.py` renders annotation overlays, `contact_sheet.py` produces a batch QA overview, and `inpaint.py` + `viz_mask_compare.py` are the gray-mask vs LaMa-inpaint robustness checks.

## Output JSON (`cue_extract/results/<id>.json`)

```json
{
  "image_id": "...", "image_size": [W, H],
  "geo_privacy_cues": [{
    "cue": "street sign", "category": "text/signage", "is_text": true,
    "instances": [{"bbox": [x1,y1,x2,y2], "score": 0.7, "source": "grounding|ocr",
                   "mask_rle": {"size": [H,W], "counts": [...]}, "degenerate": false,
                   "text": "Denton Square"}],
    "risk_level": "high", "geo_specificity": "street-level",
    "searchability": "high", "maskable": true, "degenerate": false,
    "reason": "..."
  }]
}
```

## Run (needs the GPU venv, see root README)

```bash
cue_extract/.venv/Scripts/python -m cue_extract.run_extract --ids <id1,id2,...>
cue_extract/.venv/Scripts/python -m cue_extract.contact_sheet   # QA overview
```

## Known limitations
- Vertical CJK text (e.g. Japanese festival banners) is not read by line-based OCR — the semantic cue is still captured by the proposal step, and masking is unaffected, only the transcription is missing;
- abstract / non-visual phrases occasionally cause grounding to mis-box (mostly filtered by the OCR-corroboration rule);
- an image that is "one landmark filling the whole frame" yields no maskable cue (g=0) and is excluded from the ablation.
