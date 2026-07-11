# GeoBayes — MLLM Geolocation Attacker Modeling & Per-Cue Location-Privacy Leakage

Using a multimodal LLM (GPT-4o) as a **location-privacy attacker** to study *which visual cues in an image leak geographic location, and how much each one leaks*.

The repo contains two independent lines of work:

| Line | Directory | Status | Description |
|---|---|---|---|
| **① Per-cue mPL leakage study** (current focus) | `cue_extract/` + `clue_leak/` | active | Extract location cues → SAM masks → per-cue / combination ablation → quantify each cue's leakage with mPL |
| **② GeoBayes paper reproduction** (archived) | `geobayes/` + `scripts/` | frozen | Reproduction of GeoBayes (AAAI-26), a training-free Bayesian geolocation method; see [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) |

> **Metric — mPL** (metric-normalized posterior leakage, Chen et al. 2026) = per candidate pair, `|Δln posterior-odds − Δln prior-odds| / geographic distance`. Here **prior = image with the cue masked out**, **posterior = full image**; larger mPL ⇒ the cue carries more location information.

---

## Quick start (no dataset / API / GPU needed)

The repo ships **5 sample images** (`data/sample_images/`) with **precomputed results** (cue annotations + ablation + posteriors). After cloning, install `matplotlib` and reproduce the per-image mPL figures directly:

```bash
pip install pillow matplotlib pyyaml       # plotting only needs these
python -m clue_leak.plot_one_mpl cuba      # also: newyork / okazaki / newdelhi / venice
# → clue_leak/figures/per_image_mpl/cuba_370717727_mpl.png
python -m clue_leak.plot_clue_mpl          # cross-sample per-cue mPL ranking
```
The 5 samples cover typical patterns: Cuba (single decisive cue), New York (strong text / redundancy), Okazaki (cultural cue), New Delhi (perfect redundancy), Venice (architecture + inscription). To re-run the ablation (needs `OPENAI_API_KEY`): `python -m clue_leak.run_combo2 --ids <see data/subset_sample.jsonl>` — the candidate gallery is frozen in `data/gallery_labels.json` so results stay comparable to the full run.

## Environment

Two environments (main logic is pure Python; cue extraction needs GPU deep models):

```bash
# Main environment (scoring / analysis / plotting)
pip install -r requirements.txt      # pillow, pyyaml, openai, pytest
export OPENAI_API_KEY=sk-...          # attacker model (GPT-4o); read only from the environment

# cue_extract GPU stack (Grounding DINO + SAM 2.1 + RapidOCR + LaMa, needs CUDA)
python -m venv cue_extract/.venv
cue_extract/.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cue_extract/.venv/Scripts/pip install transformers accelerate rapidocr onnxruntime \
    simple-lama-inpainting scipy pillow matplotlib numpy openai pyyaml
```
Model weights (Grounding DINO ~700MB, SAM 2.1, PP-OCRv5, LaMa ~200MB) download to the HuggingFace cache on first run. Tested GPU: RTX 5080 16GB.

## Data

Committed data: `data/subset_sample.jsonl` (5 samples), `data/sample_images/` (sample images), `data/gallery_labels.json` (frozen 77-label gallery), `data/*_cache.json` (geocoding caches).

Full reproduction needs the **IM2GPS3k** source dataset (not committed, see `.gitignore`). Once `data/im2gps3k.csv` is in place:

```bash
python scripts/build_subset.py --n 100    # fetch images from Flickr + reverse-geocode → data/subset100.jsonl
```

---

## Line ① — run order

```bash
# 1) Build the gallery geometry cache (gallery labels → coords, for the mPL distance matrix)
python -m clue_leak.prep_geo100

# 2) Cue extraction (venv): VLM proposal → Grounding DINO → OCR → SAM masks → VLM risk audit
cue_extract/.venv/Scripts/python -m cue_extract.run_extract --ids <id1,id2,...>
#   output: cue_extract/results/<id>.json (each cue carries a mask_rle)
#   QA overview: python -m cue_extract.contact_sheet → cue_extract/figures/contact_sheet.png

# 3) Per-cue mPL ablation (main env): prior = image with subset S masked, posterior = full image
python -m clue_leak.run_combo2 --ids <id1,id2,...>
#   or batch to N images: python -m clue_leak.run_50 --target 50
#   output: clue_leak/combo2_results/<id>.json

# 4) Figures
python -m clue_leak.plot_one_mpl <id-prefix>   # single sample: image + masks | per-cue & combination mPL
python -m clue_leak.plot_clue_mpl              # across samples: sorted per-cue mPL
```
Figures land in `clue_leak/figures/per_image_mpl/` (one per sample).

## Repository layout

```
cue_extract/        Cue-extraction pipeline (GPU)
  proposal / grounding / ocr / sam_mask / verify / merge / rle / prompts / viz
  run_extract.py    orchestrator  ·  contact_sheet.py QA overview  ·  inpaint.py + viz_mask_compare.py robustness checks
clue_leak/          Per-cue mPL ablation
  combo.py masking.py           subset enumeration + solid masking
  run_combo2.py run_50.py       ablation runner / batch driver
  plot_one_mpl.py plot_clue_mpl.py plot_combo2.py   plotting
  prep_geo100.py                gallery geometry prep
  cache_fullimage_posterior/    full-image posterior cache (optional; run_combo2 recomputes if absent)
  combo2_results/               ablation results  ·  figures/  output figures
geobayes/           [archived] GeoBayes reproduction: core (Bayesian loop) / search / eval / mllm / analysis
scripts/            [archived] reproduction batch jobs + data building (build_subset.py still used)
data/               subset*.jsonl + geocoding caches (source images not committed)
tests/              pytest (main line + reproduction, 190 passing)
```

## Tests

```bash
python -m pytest tests/ -q          # all (incl. reproduction archive)
python -m pytest tests/test_clueleak_combo.py tests/test_cue_extract.py -q   # main line only
```

## Known limitations (relevant for writing up)

- mPL is **marginal** leakage (leave-one-out), affected by cue redundancy; absolute values are small (mean over all pairs + distance normalization);
- the masking-combination value function is **non-monotonic** and violates Shapley additivity — report single-cue mPL only; treat combinations qualitatively;
- conclusions are specific to **this attacker (GPT-4o) + this candidate gallery**, not universal.

## References

- GeoBayes: Shi et al., AAAI-26
- mPL: Chen et al., 2026, *Metric-Normalized Posterior Leakage*
- Grounding DINO · SAM 2.1 · PaddleOCR/RapidOCR · LaMa · IM2GPS3k
