# `run_georanker_inpaint.py` on a vocabulary cache: what breaks, and the fix

Verified by **reading only** (`run_georanker_inpaint.py` was not run or edited — a scoring job
is live on it). Checked against `belief_elicit/inpaint_cache_vocab/*/manifest.json`
(10 images, produced from `cue_extract/results_vocab`).

## Verdict

**Not scorable as-is.** The cache *parsing* is fine, but every piece of cue metadata is taken
from the sweep, and the vocabulary cue list is different for all 10 images:

| image (prefix) | sweep `n_cues` | vocab `n_cues` | cue 0 in sweep → cue 0 in vocab |
|---|---:|---:|---|
| 123515458 | 3 | 7 | Thai text on storefront sign → text or signage |
| 158307292 | 4 | 6 | Equestrian statue → text or signage |
| 166869956 | 3 | 1 | Golden clock with ornate design → statue or monument |
| 169956693 | 1 | 2 | Wrigley Field text on cup → person's clothing |
| 261517384 | 4 | 5 | Japanese text on festival banner → text or signage |
| 312179336 | 3 | 4 | Cyrillic text on gravestone → text or signage |
| 370717727 | 2 | 3 | Stone fortress walls → person's clothing |
| 453636890 | 3 | 1 | Mayan architectural style → tree or vegetation |
| 486597679 | 3 | 1 | Golden religious statues → statue or monument |
| 754780171 | 2 | 3 | Church on an island → building facade |

## What goes wrong (all silent — no exception is raised)

1. **Per-image header mislabelled.** In `main()`, the record is built from the sweep record `r`:
   `"n_cues": r["n_cues"]`, `"cue_names": [pc["cue"] for pc in r["per_cue"]]`,
   `"cue_categories": [...]`, `"mpl_all_gray": r["mpl_all"]`. All of these describe the SAM3
   cue set, not the vocabulary one.
2. **Per-variant cue names mislabelled.**
   `"cue_names": [r["per_cue"][k]["cue"] for k in cues if 0 <= k < len(r["per_cue"])]`
   maps the manifest's cue index `k` onto a completely different cue.
3. **Wrong gray baseline written.** `rec["gray_mpl"] = r["per_cue"][cues[0]]["mpl"]` attaches
   the gray mPL of an unrelated cue. The vocabulary run has no gray baseline at all, so this
   field must simply not exist.
4. **Silent truncation when the vocabulary has more cues.** The `if 0 <= k < len(r["per_cue"])`
   guard means indices past the sweep's cue count are dropped from `cue_names` (giving `[]`)
   and `gray_mpl` is skipped — no warning. That is 3 of the 10 images (7 vs 3, 6 vs 4, 5 vs 4).
5. **Output-file collision.** The default `--out` is `georanker_inpaint_results.json` — the file
   the live SAM3 job is writing. Spec stems (`s0`, `p0-1`, `all`, `c0-0`, `cg0-0`) are identical
   between the two caches, so variants from the two cue vocabularies would be merged into the
   same per-image record and the spec-level resume would then consider them already done.

What is **correct** and must be kept: `parse_stem` / `list_specs` / `load_manifest` read the
vocabulary manifest fine (`files[].cues`, `files[].cov`, `files[].op`, `files[].placement`,
`files[].overlap_frac` are all picked up; `spec` being a *category* is already handled by using
the file stem as the unique id), and taking `posterior` from the sweep is right — the posterior
belongs to the unmodified image and is independent of which cue vocabulary was extracted.

## Fix as shipped

`belief_elicit/run_georanker_inpaint_vocab.py` — a wrapper that imports
`list_specs`, `PART_OF`, `MERGE_KM`, `VARIANT`, `BATCH_SIZE` from `run_georanker_inpaint`
plus `score_labels` / `build_geometry` / `mpl`, and changes only:

* cue names / categories / area fractions come from `<cache>/<image_id>/manifest.json` `cues[]`
  (`read_cue_table`), and `n_cues` is the manifest's;
* `gray_mpl` and `mpl_all_gray` are not written; records carry `cue_source: "manifest"`,
  `cache`, and `cue_area_fracs` instead, and each variant also stores `cue_categories`;
* cue indices are range-checked against the manifest and out-of-range specs are skipped **with a
  warning** instead of being silently truncated;
* images without a manifest, or absent from the sweep (no posterior), are skipped with a warning;
* default output is `georanker_inpaint_vocab_results.json` /
  `georanker_inpaint_vocab_control_results.json`, and passing `--out` pointing at either SAM3
  results file is refused outright.

Run it (GPU env), then report with the manifest cue source:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint_vocab --part main

python -m belief_elicit.inpaint_report \
  --results belief_elicit/georanker_inpaint_vocab_results.json \
  --controls belief_elicit/georanker_inpaint_vocab_control_results.json \
  --cues-from manifest --cache belief_elicit/inpaint_cache_vocab
```

## Equivalent in-place patch (if you prefer editing the original later — do NOT while it runs)

In `main()`, after resolving `d = os.path.join(args.cache, iid)`, load
`cues_meta = json.load(open(os.path.join(d, "manifest.json")))["cues"]` and replace every
`r["per_cue"][k][...]` / `r["n_cues"]` / `r["mpl_all"]` use with `cues_meta[k][...]` /
`len(cues_meta)` / (drop). Guard `k < len(cues_meta)` with a `print`+`continue` rather than a
silent filter, and make the default `--out` depend on `os.path.basename(args.cache)`.
Keep `r["posterior"]`, `r["true_label"]`, `r["country_hit"]`, `r["km_error"]` from the sweep.
