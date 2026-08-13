# belief_elicit — Belief elicitation, mPL measurement, and Shapley attribution

This module contains the current main line of the project: how the *adversary's belief*
is elicited, how per-cue location leakage (mPL) is measured, and how the non-additive
leakage is attributed to individual cues via Shapley values.

Everything here is self-contained and can be deleted without affecting `cue_extract/`
(cue naming + SAM 3 masks) or `geobayes/` (the GeoBayes reproduction).

---

## 1. Pipeline in one paragraph

Cues are named by GPT-4o and localized by SAM 3 (`cue_extract/`). For a cue subset
`S`, the masked image `I ⊖ S` gray-fills the union of those cue masks. A frozen
adversary scores every candidate location in a fixed gallery; the softmax over scores
is the *normalized adversarial belief*. **mPL** measures the distance-normalized shift
between the full-view belief and the masked-view belief. Because masking effects are
non-additive, mPL is treated as a set function `v(S)` and attributed to individual
cues with **Shapley values** (`Σφ_k = v(N)`), with a **Shapley Interaction Index**
separating overlapping from mutually-backing cues.

Full write-up: [`../mPL_to_Shapley.pdf`](../mPL_to_Shapley.pdf).

## 2. Current configuration (as of 2026-07-18)

| Component | Choice |
|---|---|
| Adversary / belief meter | **GeoRanker** = Qwen2-VL-7B + official LoRA + value head, nf4-quantized, deterministic |
| Prompt (variant B) | query image + candidate GPS + candidate label text |
| Gallery | `data/gallery_v2.json` — 138 city-level GT labels with GPS |
| Geometry | **2 km alias dedup only** (keeps genuinely nearby distinct places; see PDF §2) |
| Metric | mPL in nats/1000 km, within-image comparable |
| Attribution | Shapley `φ` + Shapley Interaction Index, exact over the full `2^m` lattice (m ≤ 5) |
| Cost | fully local, **no API calls** |

Earlier belief meters (GPT-4o verbalized scores, MC-logprob, GeoCLIP) are kept in this
folder for reference — they were found to be blind to masking, saturated, or less
accurate respectively.

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

```bash
# 0) refresh the gallery (reverse+forward geocoding, free, no API key)
python scripts/update_gallery.py

# 1) main sweep: posterior + per-single-cue + all-cues-masked, 100 images (~7 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_sweep

# 2) complete the subset lattice for images with 3–5 cues (~5 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_lattice

# 3) equal-area control: masking-artifact floor (~6 h)
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_control --all --nctrl 2

# 4) attribution + figures (CPU only, seconds)
python -m belief_elicit.shapley_v2          # φ + Shapley Interaction Index
python -m belief_elicit.control_report      # artifact floor, resolvability, corrected φ
python -m belief_elicit.plot_overview       # global overview figure
python -m belief_elicit.plot_georanker_sweep
python -m belief_elicit.plot_shapley_full
python -m belief_elicit.plot_control_v2
python -m belief_elicit.plot_case_study 158307292 754780171
```

All long runs save incrementally and resume automatically if interrupted.
Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to make runs immune to network issues
once the weights are cached.

## 5. Result files

| File | Contents |
|---|---|
| `georanker_sweep_results.json` | 95 images: posterior, per-cue priors and mPL, all-masked prior, accuracy |
| `georanker_lattice_results.json` | intermediate subsets for m = 3–5 (completes the `2^m` lattice) |
| `shapley_v2_results.json` | per-cue `φ`, Shapley Interaction Index, empty-context interaction |
| `georanker_control_results.json` | equal-area control placements (95 images / 244 cues / 516 placements) |

Full belief distributions are persisted, so any change of geometry (merge radius,
multi-resolution partitions) is pure post-processing — no re-scoring needed.

## 6. Key figures

| Figure | Shows |
|---|---|
| `figures/georanker_overview.png` | all 244 cues: raw mPL → Shapley φ, plus per-category resolvability |
| `figures/georanker_sweep.png` | adversary accuracy, confidence, per-category single-cue mPL |
| `figures/georanker_shapley.png` | per-category φ vs raw mPL; interaction-index distribution |
| `figures/georanker_control.png` | real cue vs equal-area control; masking-artifact floor |
| `figures/case_newyork_*.png` | overlap case: `Σv({k}) > v(N)`, Shapley corrects downward |
| `figures/case_slovenia_*.png` | backup case: `Σv({k}) < v(N)`, Shapley corrects upward |

## 7. Headline numbers (100 images, 95 with maskable cues)

- Adversary: 74.7 % country-scale accuracy (< 750 km), 45.3 % within 25 km, median error 59 km.
- Non-additivity: of 239 cue pairs, 182 overlap (sub-additive) and 23 back each other up.
- Efficiency `Σφ_k = v(N)` holds exactly on all 80 multi-cue images.
- Shapley correction roughly halves per-category medians and reorders the top categories.
- Equal-area control: artifact floor median 0.087 nats/1000 km; 49 % of single-cue
  effects exceed their own control, while 70 % of `φ` values exceed the pure-artifact null.
