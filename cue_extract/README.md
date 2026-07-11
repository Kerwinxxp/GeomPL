# cue_extract — Geolocation Cue Extraction Pipeline

Extracts "visual cues that may leak geographic location" from an image and produces **pixel-level masks**. Feeds clean cue regions to the downstream per-cue mPL ablation (`clue_leak/`).

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
