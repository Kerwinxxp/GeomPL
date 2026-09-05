# GeoBayes — Per-Cue Location-Privacy Leakage from Images

Which visual cues in a photo leak geographic location, and how much does each one leak?
This repository holds a single pipeline (v2) that answers that question empirically.
GPT-4o names the location cues it actually reasons from; SAM 3 turns each name into a
pixel-precise mask; a **frozen adversary** (GeoRanker = Qwen2-VL-7B + LoRA, fully local,
no API cost) scores every candidate location in a fixed 138-place gallery, both on the
full image and on images with cue subsets removed. The distance-normalized belief shift
between the two is **mPL** (metric-normalized posterior leakage). Because masking effects
are strongly non-additive, mPL is treated as a set function `v(S)` over the cue subset
lattice and attributed to individual cues with **Shapley values** (`Σφ_k = v(N)`), with a
Shapley Interaction Index separating overlapping cues from mutually-backing ones. An
equal-area control quantifies the masking-artifact floor, LaMa inpainting is used as a
less artifact-prone removal operator, and a geometric de-duplication step merges cues
whose masks are the same pixels.

Write-up: **[`mPL_to_Shapley.pdf`](mPL_to_Shapley.pdf)**.

Headline results on 100 hi-res im2gps3k images (200 have cue annotations):

- Masking effects are **non-additive**: of 239 cue pairs, 182 overlap (sub-additive) and
  23 back each other up — so single-cue mPL double-counts shared leakage and is a biased
  per-cue attribution.
- **Shapley attribution** restores exactness (`Σφ_k = v(N)` on all 80 multi-cue images,
  full `2^m` lattice, m ≤ 5), roughly halving per-category medians and reordering the top
  categories.
- An **equal-area control** (95 images / 244 cues / 516 random placements) puts the
  masking-artifact floor at a median of 0.087 nats/1000 km: only 49 % of single-cue
  effects exceed their own control, while 70 % of `φ` values exceed the pure-artifact null.
- The geometry uses a **2 km alias dedup only**, so genuinely nearby places stay
  distinguishable.

> **Metric — mPL** (Chen et al. 2026) = per candidate pair,
> `|Δln posterior-odds − Δln prior-odds| / geographic distance`. Here **prior = image with
> the cue masked out**, **posterior = full image**; larger mPL ⇒ the cue carries more
> location information.

---

## Repository layout

```
cue_extract/        Cue extraction (GPU): GPT-4o names cues → SAM 3 masks
  run_extract_sam3.py  orchestrator  ·  grounded / sam3_seg / merge / rle / prompts / viz
  extract_vocab.py     fixed-vocabulary (bottom-up) variant  ·  compare_vocab.py
  mllm.py imaging.py   GPT-4o client + smart_resize (API key from OPENAI_API_KEY only)
  results_sam3/        per-image cue JSON (each cue carries a mask_rle)
belief_elicit/      Belief elicitation, mPL measurement, Shapley attribution  ← main line
  georanker_belief.py  frozen adversary  ·  geometry.py masking.py  shared primitives
  run_georanker_*.py   sweep / lattice / control / inpaint runs (GPU)
  shapley_v3.py dedup_cues.py alt_attribution.py control_report.py   analysis (CPU)
  figures/             publication figures
data/               subset*.jsonl, gallery_v2.json, geocoding caches (source images not committed)
scripts/            update_gallery.py, fetch_hires50.py, remote_control/, remote_setup/
tests/              pytest (masking primitives, cue extraction, distributed runs, remote control)
paper/              source material for the technical note
```

## Environments

Three environments — the analysis layer is pure Python, the two model stacks need GPUs:

```bash
# 1) Main environment (analysis / plotting / tests) — CPU only
pip install -r requirements.txt      # pillow, pyyaml, openai, pytest
pip install numpy matplotlib          # the CPU analyses and figures need these too
export OPENAI_API_KEY=sk-...          # cue naming (GPT-4o); read only from the environment

# 2) cue_extract GPU stack (SAM 3 + LaMa, needs CUDA)
python -m venv cue_extract/.venv
cue_extract/.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cue_extract/.venv/Scripts/pip install "transformers>=5.13" accelerate \
    simple-lama-inpainting scipy "pillow>=10.4" matplotlib numpy openai pyyaml

# 3) belief_elicit GPU stack (GeoRanker) — see belief_elicit/README.md §3
py -3.12 -m venv belief_elicit/.venv_gr
```

`transformers>=5.13` ships SAM 3 support natively (`Sam3Model`). `pillow>=10.4` is
required — older Pillow (9.x) renders digit glyphs as boxes, breaking numbered-badge
overlays. Model weights (SAM 3 ~840 MB, LaMa ~200 MB, Qwen2-VL-7B ~16.5 GB) download to
the HuggingFace cache on first run; **SAM 3 is gated** — accept the license at
<https://huggingface.co/facebook/sam3> and `huggingface-cli login` first. Tested GPU:
RTX 5080 16 GB (SAM 3 uses ~3.4 GB).

## Data

Committed: `data/subset*.jsonl` (image manifests, read via a glob),
`data/sample_images/` (5 sample images), `data/gallery_v2.json` (138 city-level GT labels
with GPS — the candidate gallery), `data/*_cache.json` (geocoding caches).

Full reproduction needs the **IM2GPS3k** source dataset (not committed, see `.gitignore`);
`scripts/fetch_hires50.py` re-fetches hi-res originals from Flickr for an existing subset,
and `scripts/update_gallery.py` rebuilds the gallery geometry (free, no API key).

## Run order

The end-to-end order — cue extraction, GPU scoring sweeps, then the CPU analyses and
figures — is documented step by step in **[`belief_elicit/README.md`](belief_elicit/README.md) §4**.
Cue extraction itself is documented in [`cue_extract/README.md`](cue_extract/README.md).

## Results and figures

Result JSONs live next to the code that produces them in `belief_elicit/`
(`shapley_v3_results.json` is the primary attribution; see that README §5 for the full
table). Figures land in `belief_elicit/figures/` and `cue_extract/figures*/`; the
annotated per-image cue overlays are in `cue_extract/figures_sam3/`.

## Tests

```bash
python -m pytest tests -q
```
`tests/test_cue_extract.py` needs numpy + Pillow; `tests/test_distributed_georanker.py`
and `tests/test_remote_control.py` are pure-Python and run anywhere.

## Known limitations (relevant for writing up)

- mPL is a **marginal** quantity; with redundant cues the single-cue value is biased, which
  is exactly why Shapley `φ` (not single-cue mPL) is the reported per-cue attribution;
- absolute values are small — mPL averages over all candidate pairs and normalizes by
  distance, so compare within an image, not across setups;
- gray-block masking introduces a measurable **artifact floor**; the equal-area control and
  the LaMa-inpaint variant bound it, but cues below that floor are not resolvable;
- conclusions are specific to **this attacker (GeoRanker) + this candidate gallery**, not
  universal.

## History

Two earlier lines of work were removed from the working tree at tag **`pre-cleanup`** and
remain in git history: the **GeoBayes paper reproduction** (`geobayes/` + its batch scripts
and reports) and the **first-generation per-cue ablation** (`clue_leak/`, GPT-4o verbalized
scoring with a 25 km cluster merge), whose elicitation turned out to be blind to masking.
Neither is part of the current pipeline; `git checkout pre-cleanup` to inspect them.

## References

- GeoBayes: Shi et al., AAAI-26
- mPL: Chen et al., 2026, *Metric-Normalized Posterior Leakage*
- GeoRanker (Qwen2-VL-7B + LoRA) · SAM 3 (`facebook/sam3`) · LaMa · IM2GPS3k
