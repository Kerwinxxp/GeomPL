# clue_leak — Per-Cue mPL Location-Leakage Ablation

Quantifies "how much the attacker's belief moves, relative to the full image, when a cue (or a group of cues) is masked out." Prior = image with subset S masked, posterior = full image.

## Metric

Over a fixed candidate gallery (unique GT labels of subset100, clustered at 25 km), for a given image's prior/posterior belief distributions:

```
mPL(i,j) = |ln(post_i/prior_i) − ln(post_j/prior_j)| / d_ij × 1000     # nats / 1000 km
```
For each image we take the **mean** over all candidate pairs (primary, stable metric). The `plot_*` scripts compute this inline (geometry from `data/forward_geocode_cache.json`).

## Combinations and overlap

`combo.nonempty_subsets(m)` enumerates all non-empty subsets (for m > 6 it reduces to singletons + leave-one-out + full set). Combination masking uses the **union** of the cue masks (`masking.mask_solid_from_masks`; overlapping pixels are masked once).

## Files

| File | Role |
|---|---|
| `combo.py` | subset enumeration |
| `masking.py` | solid masking: `mask_solid_from_masks` (irregular SAM masks) / `mask_solid_regions` (bbox) |
| `run_combo2.py` | ablation runner: reads masks from `cue_extract/results_sam3`, scores each subset → `combo2_sam3_results/` (dirs overridable via `--cue_dir/--out_dir/--post_dir`) |
| `run_50.py` | batch driver: picks N images with maskable cues, runs ablation + figures |
| `plot_one_mpl.py` | single-sample figure: image + masks \| per-cue & combination mPL |
| `plot_clue_mpl.py` | across images: sorted per-cue mPL bar chart |
| `plot_combo2.py` | multi-sample multi-panel + redundancy/synergy printout |
| `prep_geo100.py` | reverse/forward geocoding, builds the gallery geometry cache |

## Output JSON (`combo2_sam3_results/<id>.json`)

```json
{"image_id":"...", "true_label":"City, Country", "n_maskable": 4,
 "cue_meta":[{"cue":"...","category":"...","risk":"..."}],
 "posterior": {"label": prob, ...}, "mask_type": "sam_irregular",
 "combos":[{"subset":[0,1], "cues":[...], "prior":{...},
            "kl_bits": 0.05, "prior_prob_true": 0.02}]}
```

## Run

```bash
python -m clue_leak.prep_geo100                       # one-time: geometry cache
python -m clue_leak.run_combo2 --ids <id1,id2,...>    # ablation (defaults to SAM3 cues; posterior recomputed if not cached)
python -m clue_leak.plot_one_mpl cuba                 # single-sample figure (place name or image-id prefix)
```

## Reading the numbers
- Single-cue mPL = **marginal** leakage (with the other cues present); redundancy flattens it;
- the combination value function is **non-monotonic** (masking a superset can score lower than a subset) → no Shapley / additivity attribution;
- small absolute mPL is expected (mean over all pairs + distance normalization); compare cues **relatively**.
