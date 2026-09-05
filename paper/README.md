# `paper/` — build scripts for `mPL_to_Shapley.pdf`

Regenerate the technical note (written to the repo root) with two commands:

```bash
python paper/paper_figs.py    # draws Figure 1 and copies the result figures into paper/assets/
python paper/build_pdf.py     # renders the equations and writes ../mPL_to_Shapley.pdf
```

`paper/build_pdf.py` refuses to run if `paper/assets/` is missing, so always run
`paper_figs.py` first (it is cheap; re-run it whenever a `belief_elicit` figure changes).

## What each script does

**`paper_figs.py`**
- draws Figure 1, the v2 pipeline diagram, from scratch with matplotlib boxes and arrows;
- copies the result figures produced by the `belief_elicit/plot_*.py` scripts into
  `paper/assets/` under stable names, so the PDF build has a self-contained asset directory.

| asset | source (`belief_elicit/figures/`) | produced by |
|---|---|---|
| `fig1_pipeline.png` | — (drawn here) | `paper_figs.py` |
| `fig2_sweep.png` | `georanker_sweep.png` | `plot_georanker_sweep.py` |
| `fig3_overview.png` | `georanker_overview_dedup.png`, else `georanker_overview.png` | `plot_overview.py --shapley belief_elicit/shapley_v3_results.json` |
| `fig4_dedup.png` | `dedup_before_after.png` | `plot_dedup.py` |
| `fig5_vocab.png` | `vocab_vs_gpt4o_leakage.png` | `plot_vocab_vs_gpt4o.py` |
| `fig6_inpaint.png` | `inpaint_vs_gray.png` | `plot_inpaint.py` |
| `fig7_case_newyork.png` | `case_newyork_158307292.png` | `plot_case_study.py` |
| `fig8_case_slovenia.png` | `case_slovenia_754780171.png` | `plot_case_study.py` |

**`build_pdf.py`**
- registers DejaVu Serif (all four faces) straight out of matplotlib's `mpl-data/fonts/ttf`,
  so no system font install is needed;
- renders every display equation to a tight 300 dpi PNG with matplotlib mathtext
  (`mathtext.fontset = dejavuserif`) into `paper/assets/eq/`, then scales each one into a
  reportlab table with a right-aligned equation number — no TeX required;
- lays the note out on A4 with `reportlab.platypus`, using `KeepTogether` so section
  headings are never orphaned and no figure is separated from its caption.

## Sources of the numbers

Every figure in the note is checked into `belief_elicit/figures/`; the numbers quoted in the
body come from these files and can be re-derived from them:

- `belief_elicit/shapley_v3_results.json` — the de-duplicated inventory (95 images, 223
  players): per-cue `phi`, `v_single`, `sii`, efficiency, resolvability flags. Primary result.
- `belief_elicit/dedup_report.md` / `.json` — before/after de-duplication tables (§6.1, Table 2).
- `belief_elicit/vocab_vs_gpt4o.md` / `.json` — the fixed-vocabulary ablation (§6.2).
- `belief_elicit/inpaint_report.md`, `inpaint_summary.json` — gray vs LaMa inpainting,
  the artifact floor and resolvability (§7, Table 3).
- `belief_elicit/order2_shapley_validation.json` — order-2 anchored Shapley validation (§4.1).
- `belief_elicit/alt_attribution_results.json` (regenerate the printed summary with
  `python -m belief_elicit.alt_attribution`) — Banzhaf, additive-surrogate R², Harsanyi
  order shares (§8, robustness paragraph).
- `belief_elicit/georanker_sweep_results.json` — adversary accuracy and `p_true` (§8).

## Conventions

- **English only.** The PDF is shared publicly; no CJK text anywhere, including in figures.
- Keep the notation of the note: belief `Pr_θ(·|I)` from scores `s_θ(x;I)` via a softmax with
  temperature `τ`; pairwise mPL as `|Δ log-odds| / d(x_i,x_j) × 1000` in nats/1000 km; set
  function `v(S)`; Shapley `φ_k`; Shapley interaction index `I^SII`.
- Do not nest a `KeepTogether` inside another one — reportlab's `KeepTogether.wrap()` reports
  height `0xffffff` to force a split, so the outer block always overflows its page and leaves a
  large hole. Use `figure_parts()` (plain flowables) when a figure has to go inside an outer
  `KeepTogether`, and `figure()` everywhere else.
- DejaVu Serif has no U+1D4B3 / U+1D4AB, so the script metric-space and pair-set symbols are
  written as plain italic `X` / `P` in body text (the equations use `\mathcal{}`). The
  `SUBST` table in `build_pdf.py` handles that plus thin-space normalization.

## Verifying a build

```bash
python -c "import pypdfium2 as p; d=p.PdfDocument('mPL_to_Shapley.pdf'); print(len(d)); [d[i].render(scale=2).to_pil().save(f'page{i+1}.png') for i in range(len(d))]"
```

Check: page count (currently 9), no missing glyphs, equations legible, figures inside the
margins, no orphan headings, and no page left mostly blank.
