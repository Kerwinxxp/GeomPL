# GeoBayes — MLLM Geolocation Attacker Modeling & Per-Cue Location-Privacy Leakage

Using a multimodal LLM (GPT-4o) as a **location-privacy attacker** to study *which visual cues in an image leak geographic location, and how much each one leaks*.

The repo contains two independent lines of work:

| Line | Directory | Status | Description |
|---|---|---|---|
| **① Per-cue leakage: mPL + Shapley attribution** (current focus) | `cue_extract/` + `belief_elicit/` | active | GPT-4o cue naming → SAM 3 masks → subset ablation under a frozen adversary → mPL as a set function → Shapley attribution |
| **② Earlier per-cue ablation** (superseded) | `clue_leak/` | archived | First-generation ablation with GPT-4o verbalized scores; kept for reference, its mPL values are superseded (see below) |
| **③ GeoBayes paper reproduction** (archived) | `geobayes/` + `scripts/` | frozen | Reproduction of GeoBayes (AAAI-26), a training-free Bayesian geolocation method; see [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) |

> ### Current main line → [`belief_elicit/`](belief_elicit/README.md)
> Write-up: **[`mPL_to_Shapley.pdf`](mPL_to_Shapley.pdf)** (English) · **[`mPL_to_Shapley_zh.pdf`](mPL_to_Shapley_zh.pdf)** (Chinese)
>
> Key findings on 100 hi-res im2gps3k images (200 images have cue annotations):
> - Masking effects are **non-additive**: of 239 cue pairs, 182 overlap (sub-additive) and 23 back each other up — so single-cue mPL double-counts shared leakage and is a biased per-cue attribution.
> - **Shapley attribution** fixes this with an exact `Σφ_k = v(N)` on all 80 multi-cue images (full `2^m` lattice, m ≤ 5), roughly halving per-category medians and reordering the top categories.
> - An **equal-area control** (95 images / 244 cues / 516 random placements) quantifies the masking-artifact floor: only 49 % of single-cue effects exceed their own control, while 70 % of `φ` values exceed the pure-artifact null.
> - The belief meter is now **GeoRanker** (Qwen2-VL-7B + LoRA), running fully locally with **no API cost**; the geometry uses a 2 km alias dedup so genuinely nearby places stay distinguishable.

> ⚠️ **Superseded results.** Figures and JSONs under `clue_leak/` come from the first-generation setup (GPT-4o verbalized scoring, 25 km cluster merge). That elicitation was later found to be **blind to masking** (0.1-quantized scores with a 0.05 floor), so those mPL numbers should not be used. Use `belief_elicit/` instead.

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

# cue_extract GPU stack (SAM 3 + LaMa, needs CUDA)
python -m venv cue_extract/.venv
cue_extract/.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cue_extract/.venv/Scripts/pip install "transformers>=5.13" accelerate \
    simple-lama-inpainting scipy "pillow>=10.4" matplotlib numpy openai pyyaml
```
`transformers>=5.13` ships SAM 3 support natively (`Sam3Model`). `pillow>=10.4` is required — older Pillow (9.x) renders digit glyphs as boxes, breaking numbered-badge overlays. Model weights (SAM 3 ~840MB, LaMa ~200MB) download to the HuggingFace cache on first run; **SAM 3 is gated** — accept the license at <https://huggingface.co/facebook/sam3> and `huggingface-cli login` first. Tested GPU: RTX 5080 16GB (SAM 3 uses ~3.4 GB).

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

# 2) Cue extraction (venv). RECOMMENDED: route-B + SAM 3 (see cue_extract/README).
#    GPT-4o names the cues + reasoning → SAM 3 segments each by text → precise masks.
#    Needs: accept the gated model at https://huggingface.co/facebook/sam3 + huggingface-cli login.
cue_extract/.venv/Scripts/python -m cue_extract.run_extract_sam3 --ids <id1,id2,...>
#   output: cue_extract/results_sam3/<id>.json (each cue carries a mask_rle)

# 3) Per-cue mPL ablation (main env): prior = image with subset S masked, posterior = full image
python -m clue_leak.run_combo2 --ids <id1,id2,...> \
       --cue_dir cue_extract/results_sam3 --out_dir clue_leak/combo2_sam3_results \
       --post_dir clue_leak/cache_post_hires   # separate posterior cache per resolution
#   or batch to N images: python -m clue_leak.run_50 --target 50
#   output: clue_leak/combo2_sam3_results/<id>.json

# 4) Figures
python -m clue_leak.plot_one_mpl cuba          # single sample (place name or image-id prefix): image + masks | per-cue & combination mPL
python -m clue_leak.plot_clue_mpl              # across samples: sorted per-cue mPL
```
Figures land in `clue_leak/figures/per_image_mpl_sam3/` (one per sample).

## Repository layout

```
cue_extract/        Cue-extraction pipeline (GPU): route-B + SAM 3
  grounded / sam3_seg / merge / rle / prompts / viz
  run_extract_sam3.py  orchestrator  ·  contact_sheet.py QA overview
  viz_compare_sam3.py  old-vs-SAM3 comparison  ·  inpaint.py + viz_mask_compare.py robustness check
clue_leak/          Per-cue mPL ablation
  combo.py masking.py           subset enumeration + solid masking
  run_combo2.py run_50.py       ablation runner / batch driver
  plot_one_mpl.py plot_clue_mpl.py plot_combo2.py   plotting
  prep_geo100.py                gallery geometry prep
  cache_post_hires/             full-image posterior cache (optional; run_combo2 recomputes if absent)
  combo2_sam3_results/          ablation results  ·  figures/per_image_mpl_sam3/  output figures
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
- SAM 3 (`facebook/sam3`) · LaMa · IM2GPS3k
