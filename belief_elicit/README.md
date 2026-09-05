# belief_elicit — Belief elicitation, mPL measurement, and Shapley attribution

This module contains the main line of the project: how the *adversary's belief*
is elicited, how per-cue location leakage (mPL) is measured, and how the non-additive
leakage is attributed to individual cues via Shapley values.

Its only dependency on the rest of the repo is `cue_extract` (`rle` for mask decode,
`common` for the cue-record reader) plus the cue JSONs under `cue_extract/results_sam3/`
and `cue_extract/results_vocab/`.

---

## 0. Module map

The package is flat: a handful of **library** modules hold everything shared, and every
runner / report / plot script is a thin CLI over them.

**Library** (no CLI, no side effects at import)

| file | contents |
|---|---|
| `results.py` | where every result and data file lives (path constants) + the loaders: `load_json` · `load_sweep` · `load_lattice` · `load_controls` · `load_inpaint` · `load_shapley` · `load_gallery` · `load_subsets` · `image_path` · `load_manifest` / `manifest_index` · `index_variants` · `by_image`, plus the set function `build_v(record, lattice)`, its merged form `merged_v(v, groups)` and the belief lattice `build_q(record, lattice)` |
| `attribution.py` | the operators on v(S): `shapley` · `sii` · `empty_interaction` · `banzhaf` · `harsanyi` · `additive_fit` · `min_sufficient` · `order2_shapley` / `order2_from_v` (second-order anchored truncation) · `spearman` |
| `cues.py` | pixel primitives: `cue_masks_of` (the single cue-mask reader, delegating to `cue_extract.common.cue_masks`) · `cue_unions` · `translate_mask` · `sample_control` (equal-area control placement) · `mask_to_rle` |
| `geometry.py` | mPL geometry: `haversine_km` · `cluster_representatives` · `merge_distribution` · `build_geometry` · `mpl` |
| `risk.py` | **protection-oriented** (signed, forward-looking) risk on a released belief: `signed_evidence` · `p_true` · `rank_true` · `err_km` / `exp_err_km` · `mpl_vs` / `mpl_max_vs` · `eps_local` · `residual_allmask` · `risk_vector` · `gallery_coords`, plus the cached numpy evaluator `RiskGeometry` |
| `masking.py` | `mask_solid_from_masks` (gray fill from boolean masks) · `nonempty_subsets` |
| `inpaint_ops.py` | `inpaint_from_masks` — LaMa as the removal operator (GPU) |
| `georanker_belief.py` | the frozen adversary: model loading + `score_labels` (GPU) |
| `plotstyle.py` | figure palette, the shared `rcParams` (`apply_style`), `save(fig, path, dpi=...)`, `short`, `fmt` |

**GPU runners** (`python -m belief_elicit.<name>`)

| file | produces |
|---|---|
| `run_georanker_sweep.py` | `georanker_sweep_results.json` |
| `run_georanker_lattice.py` | `georanker_lattice_results.json` |
| `run_georanker_control.py` | `georanker_control_results.json` |
| `run_georanker_inpaint.py` | `georanker_inpaint[_control]_results.json` (also owns the cache-manifest parsing: `parse_stem`, `list_specs`) |
| `run_georanker_inpaint_vocab.py` | `georanker_inpaint_vocab_results.json` |
| `run_georanker_check.py` | instrument health check (stdout + `georanker_check_<tag>.json`) |
| `location_prior.py` | `location_prior.json` — the adversary's content-free prior pi (3 blank probes, ~3 min) |
| `precompute_inpaint.py` | the `inpaint_cache*/` pixel variants (LaMa, cue_extract venv) |
| `distributed_georanker.py` · `run_distributed_georanker.py` · `merge_distributed_georanker.py` | multi-machine sharding and merge |

**CPU analyses and reports**

| file | produces |
|---|---|
| `dedup_cues.py` | `cue_dedup_groups.json` |
| `shapley_v3.py` | `shapley_v3_results.json` — holds the single attribution implementation, `run(dedup=...)` |
| `shapley_v2.py` | `shapley_v2_results.json` — the same `run` with `dedup=False` (thin CLI) |
| `order2_shapley.py` | `order2_shapley_validation.json` (CLI over `attribution.order2_shapley`) |
| `alt_attribution.py` | `alt_attribution_results.json` |
| `calibrate_tau.py` | `calibrate_tau_results.json` |
| `dedup_report.py` | `dedup_report.md` + `.json` |
| `inpaint_report.py` | `[vocab_]inpaint_report.md` + `[vocab_]inpaint_summary.json` |
| `vocab_vs_gpt4o.py` | `vocab_vs_gpt4o.md` + `.json` |
| `control_report.py` | stdout only (artifact floor, resolvability, corrected phi) |
| `protection_set.py` | `protection_set_results.json` + `protection_set_report.md` — selection-strategy comparison under the risk metrics |

**Figures** — `plot_overview` · `plot_georanker_sweep` · `plot_shapley_v2` ·
`plot_control_v2` · `plot_dedup` · `plot_inpaint` · `plot_inpaint_check` ·
`plot_vocab_vs_gpt4o` · `plot_case_study` · `plot_pipeline_v2` · `plot_protection_set`, all writing into
`figures/` via `plotstyle.save`.

---

## 1. Pipeline in one paragraph

Cues are named by GPT-4o and localized by SAM 3 (`cue_extract/`). For a cue subset
`S`, the masked image `I ⊖ S` gray-fills the union of those cue masks (or LaMa-inpaints
it — see §4.5). A frozen adversary scores every candidate location in a fixed gallery;
the softmax over scores is the *normalized adversarial belief*. **mPL** measures the
distance-normalized shift between the full-view belief and the masked-view belief.
Because masking effects are non-additive, mPL is treated as a set function `v(S)` and
attributed to individual cues with **Shapley values** (`Σφ_k = v(N)`), with a **Shapley
Interaction Index** separating overlapping from mutually-backing cues.

Full write-up: [`../mPL_to_Shapley.pdf`](../mPL_to_Shapley.pdf).

## 2. Current configuration (as of 2026-07-18)

| Component | Choice |
|---|---|
| Adversary / belief meter | **GeoRanker** = Qwen2-VL-7B + official LoRA + value head, nf4-quantized, deterministic |
| Prompt (variant B) | query image + candidate GPS + candidate label text |
| Gallery | `data/gallery_v2.json` — 138 city-level GT labels with GPS |
| Geometry | **2 km alias dedup only** (keeps genuinely nearby distinct places; see PDF §2) |
| Removal operator | gray fill (main line) + LaMa inpainting (artifact-robustness check) |
| Metric | mPL in nats/1000 km, within-image comparable |
| Attribution | Shapley `φ` + Shapley Interaction Index, exact over the full `2^m` lattice (m ≤ 5); second-order anchored Shapley where the full lattice is unaffordable |
| Cue list | GPT-4o proposals, then **geometric de-duplication** at mask IoU ≥ 0.9 |
| Cost | fully local, **no API calls** |

Earlier belief meters (GPT-4o verbalized scores, MC-logprob, GeoCLIP) were found to be
blind to masking, saturated, or less accurate respectively; they were removed at tag
`pre-cleanup` and remain in git history.

## 3. Setup

```bash
# separate env: GeoRanker needs its own torch/transformers stack
py -3.12 -m venv belief_elicit/.venv_gr
belief_elicit/.venv_gr/Scripts/python.exe -m pip install \
    torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
belief_elicit/.venv_gr/Scripts/python.exe -m pip install \
    "transformers>=4.54,<5" peft accelerate "qwen-vl-utils==0.0.8" bitsandbytes pillow numpy
```

The Qwen2-VL-7B base model downloads automatically on first run (~16.5 GB).

**LoRA checkpoint** (not shipped, 13.8 MB): download `checkpoints/adapter_model.safetensors`
from [GeoRanker](https://github.com/Applied-Machine-Learning-Lab/GeoRanker) into
`belief_elicit/georanker_ckpt/`. The source commit is recorded in
`georanker_ckpt/SOURCE_COMMIT.txt`; `adapter_config.json` is included here.

Note: `flash-attn` and `deepspeed` from the upstream repo are **not** required — the
model loader in `georanker_belief.py` re-implements the reward head and uses `sdpa`
attention, so it runs on Windows.

## 4. Running the experiments

### 4.1 Gallery

```bash
python scripts/update_gallery.py     # reverse+forward geocoding, free, no API key
```

### 4.2 GPU scoring runs (gray-block operator)

```bash
# main sweep: posterior + per-single-cue + all-cues-masked, 100 images (~7 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_sweep

# complete the subset lattice for images with 3–5 cues (~5 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_lattice

# equal-area control: masking-artifact floor (~6 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_control --all --nctrl 2

# optional instrument health check on a few images (accuracy / masking response / mPL shape)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_check --images 261517384
```

For multi-machine runs, see [`DISTRIBUTED_RUNS.md`](DISTRIBUTED_RUNS.md)
(`run_distributed_georanker.py` shards, `merge_distributed_georanker.py` merges).

### 4.3 De-duplication (CPU, seconds)

```bash
python -m belief_elicit.dedup_cues        # → cue_dedup_groups.json (mask IoU ≥ 0.9 merge)
```

### 4.4 Attribution and reports (CPU, seconds)

```bash
python -m belief_elicit.shapley_v2        # φ + SII on the raw 244-cue list
python -m belief_elicit.shapley_v3        # φ + SII after de-dup  ← primary attribution
#   (both are the same implementation: shapley_v3.run(dedup=True|False))
python -m belief_elicit.dedup_report      # BEFORE/AFTER report (v2 vs v3) → dedup_report.md
python -m belief_elicit.control_report    # artifact floor, resolvability, corrected φ (stdout)
python -m belief_elicit.alt_attribution   # Banzhaf / additive surrogate / Harsanyi / minimal sufficient set
python -m belief_elicit.calibrate_tau     # temperature sensitivity, NLL/Brier/ECE, mPL ∝ 1/τ
```

### 4.5 Inpainting arm (LaMa instead of gray blocks)

```bash
# 1) precompute pixel variants (uses the cue_extract venv: torch + CUDA + simple_lama_inpainting)
cue_extract/.venv/Scripts/python.exe -m belief_elicit.precompute_inpaint
cue_extract/.venv/Scripts/python.exe -m belief_elicit.precompute_inpaint \
    --src cue_extract/results_vocab --cache belief_elicit/inpaint_cache_vocab

# 2) score them (GeoRanker venv)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint_vocab

# 3) reports (CPU) — second-order anchored Shapley, see order2_shapley.py
python -m belief_elicit.inpaint_report                       # → inpaint_report.md + inpaint_summary.json
python -m belief_elicit.vocab_vs_gpt4o                       # GPT-4o vs fixed-vocabulary cue lists
```

### 4.6 Protection-set selection (which cues to actually remove)

Attribution says *which cue explains the shift*; protection asks *what is left after
treating S*. The risk metrics live in `risk.py` and are measured against the adversary's
content-free prior π, elicited once on the GPU:

```bash
# 1) the reference prior π: three content-free probes (GeoRanker venv, ~3 min)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.location_prior

# 2) six selection strategies × every budget k, on the full gray lattice (CPU, ~2 s)
python -m belief_elicit.protection_set        # + a k ≤ 2 replicate on the inpaint arm
```

Without step 1 the analysis falls back to a uniform prior and says so loudly.

### 4.7 Figures

```bash
python -m belief_elicit.plot_overview                        # → figures/georanker_overview.png
python -m belief_elicit.plot_overview \
    --shapley belief_elicit/shapley_v3_results.json \
    --out belief_elicit/figures/georanker_overview_dedup.png # de-dup variant
python -m belief_elicit.plot_georanker_sweep
python -m belief_elicit.plot_shapley_v2
python -m belief_elicit.plot_control_v2
python -m belief_elicit.plot_dedup
python -m belief_elicit.plot_inpaint                         # add --out-prefix vocab_ for the vocab arm
python -m belief_elicit.plot_inpaint_check
python -m belief_elicit.plot_vocab_vs_gpt4o
python -m belief_elicit.plot_case_study 158307292 754780171
python -m belief_elicit.plot_pipeline_v2                     # explainer figures for the v2 pipeline
python -m belief_elicit.plot_protection_set                 # -> figures/protection_set.png
```

All long runs save incrementally and resume automatically if interrupted.
Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to make runs immune to network issues
once the weights are cached.

## 5. Result files

| File | Contents |
|---|---|
| `shapley_v3_results.json` | **primary attribution** — per-cue `φ` + SII after geometric de-dup (223 merged players) |
| `shapley_v2_results.json` | same on the raw 244-cue list (BEFORE side of the de-dup comparison) |
| `cue_dedup_groups.json` | which cues were merged, with pairwise IoU and containment reports |
| `georanker_sweep_results.json` | 95 images: posterior, per-cue priors and mPL, all-masked prior, accuracy |
| `georanker_lattice_results.json` | intermediate subsets for m = 3–5 (completes the `2^m` lattice) |
| `georanker_control_results.json` | equal-area control placements (95 images / 244 cues / 516 placements) |
| `georanker_inpaint_results.json` | LaMa-inpaint scores: singles, pairs, all, per-cue controls |
| `georanker_inpaint_control_results.json` | inpaint vs same-placement gray controls (`c*` / `cg*`) |
| `georanker_inpaint_vocab_results.json` | same, for the fixed-vocabulary cue list |
| `alt_attribution_results.json` | Banzhaf, additive-surrogate `R²`, Harsanyi dividends, minimal sufficient sets |
| `order2_shapley_validation.json` | accuracy of the second-order truncation against the exact gray-block lattice |
| `calibrate_tau_results.json` | τ sensitivity, NLL/Brier/ECE, `mPL(τ) ≈ mPL(1)/τ` |
| `location_prior.json` | the adversary's content-free prior π (gray / noise / blur probes + their average) and the uniform reference |
| `protection_set_results.json` | per-image risk cache over every subset, the six strategies' picks at every k, minimal-k/η, misleading-cue statistics, inpaint replicate |
| `dedup_report.md` · `inpaint_report.md` · `vocab_inpaint_report.md` · `vocab_vs_gpt4o.md` · `protection_set_report.md` | generated prose reports |

Full belief distributions are persisted, so any change of geometry (merge radius,
multi-resolution partitions) is pure post-processing — no re-scoring needed.

## 6. Key figures

| Figure | Shows |
|---|---|
| `figures/georanker_overview.png` | all 244 cues: raw mPL → Shapley φ, plus per-category resolvability |
| `figures/georanker_overview_dedup.png` | the same after geometric de-duplication (v3 cue list) |
| `figures/georanker_sweep.png` | adversary accuracy, confidence, per-category single-cue mPL |
| `figures/georanker_shapley.png` | per-category φ vs raw mPL; interaction-index distribution |
| `figures/georanker_control.png` | real cue vs equal-area control; masking-artifact floor |
| `figures/dedup_before_after.png` | de-dup effect: per-category φ, SII, merged-group credit check |
| `figures/inpaint_vs_gray.png` · `vocab_inpaint_vs_gray.png` | LaMa inpainting vs gray blocks (values, interactions, artifact floor) |
| `figures/inpaint_check.png` | visual sanity check of one inpainted variant and its control |
| `figures/vocab_vs_gpt4o_leakage.png` | GPT-4o (top-down) vs fixed-vocabulary (bottom-up) cue lists |
| `figures/case_newyork_*.png` | overlap case: `Σv({k}) > v(N)`, Shapley corrects downward |
| `figures/case_slovenia_*.png` | backup case: `Σv({k}) < v(N)`, Shapley corrects upward |
| `figures/pipeline_v2_walkthrough.png` · `pipeline_v2_diagram.png` | end-to-end explainer for the v2 pipeline |
| `figures/protection_set.png` | protection: residual risk vs budget k, regret per strategy, minimal-k/η curves, and φ vs the *signed* contribution |

## 7. Headline numbers (100 images, 95 with maskable cues)

- Adversary: 74.7 % country-scale accuracy (< 750 km), 45.3 % within 25 km, median error 59 km.
- Non-additivity: of 239 cue pairs, 182 overlap (sub-additive) and 23 back each other up.
- Efficiency `Σφ_k = v(N)` holds exactly on all 80 multi-cue images.
- Shapley correction roughly halves per-category medians and reorders the top categories.
- Equal-area control: artifact floor median 0.087 nats/1000 km; 49 % of single-cue
  effects exceed their own control, while 70 % of `φ` values exceed the pure-artifact null.
- Protection (76 images, `P ≥ 2`): π is diffuse (H = 4.84 vs 4.93 nats uniform). 38 % of
  de-duplicated cues are **misleading** — removing them alone *raises* the attacker's
  log-odds on the truth — and 18/76 Shapley top-1 picks are among them. Shapley top-k
  costs 0.23–0.25 nats of regret against the optimal set and is optimal on only 42–63 %
  of images; the signed criterion is exactly optimal at k = 1. Even with the full cue
  vocabulary, 72–84 % of images cannot be pushed to `R1 ≤ η` by any proper subset.
