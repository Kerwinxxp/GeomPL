# Inpainting vs gray masking: per-cue location leakage

- results file: `C:\Users\phdwf\OneDrive\Desktop\GeoBayes\belief_elicit\georanker_inpaint_results.json`
- cue metadata: `sweep`
- gray baselines: available
- images in results file: **95** (scored variants: 578)
- generated from partial data: sections report their own usable-image counts

## (a) Single-cue removal: gray vs inpaint

Usable images: **95**, cues: **244** (paired with a gray baseline: 244).

- Spearman rho(gray, inpaint) = **0.856** (Pearson r = 0.879)
- fraction with inpaint < gray = **57.4%**
- median mPL: gray 0.0982 → inpaint 0.0919 (median delta -0.0057, median ratio 0.939x)

| category | n | inpaint median | gray median |
|---|---:|---:|---:|
| text/signage | 29 | 0.1347 | 0.1355 |
| vehicles/license plates | 12 | 0.1076 | 0.1293 |
| landmarks/buildings | 31 | 0.1027 | 0.1228 |
| road/infrastructure | 10 | 0.0905 | 0.0796 |
| commercial/cultural | 42 | 0.0888 | 0.0882 |
| environment | 78 | 0.0823 | 0.0940 |
| architecture | 41 | 0.0810 | 0.0870 |
| other | 1 | 0.0451 | 0.0468 |

## (b) Order-2 anchored Shapley under inpainting

phi_k = d_k + 1/2 * sum_l d_kl, then anchored so that sum_k phi_k = v(N).

Usable images (all singles + all pairs + all): **95**, cues: **244**; paired with gray exact phi: 80 images / 244 cues.

- median truncation residual share |v(N)-sum phi^(2)|/|v(N)| = **0.031**
- global Spearman(phi_inpaint, phi_gray) = **0.750**
- within-image Spearman median = **1.000** (n=80 images)
- top-1 cue agreement = **72.5%** (n=80)
- median phi: gray 0.0552 → inpaint 0.0561

| category | n | phi_inpaint median | phi_gray median |
|---|---:|---:|---:|
| vehicles/license plates | 12 | 0.0750 | 0.0450 |
| landmarks/buildings | 31 | 0.0681 | 0.0718 |
| text/signage | 29 | 0.0678 | 0.0803 |
| environment | 78 | 0.0591 | 0.0604 |
| architecture | 41 | 0.0568 | 0.0418 |
| commercial/cultural | 42 | 0.0374 | 0.0372 |
| road/infrastructure | 10 | 0.0355 | 0.0337 |
| other | 1 | 0.0277 | 0.0312 |

## (c) Non-additivity under inpainting

Usable images (singles + all, m>=2): **80**; pairs with an interaction value: **239**.

- fraction sub-additive (v(N) < sum_k v({k})) = **83.8%**
- median v(N) / sum_k v({k}) = **0.621**

| interaction estimator | n | d < -0.01 | ~0 | d > 0.01 | median |
|---|---:|---:|---:|---:|---:|
| inpaint d_kl (empty context) | 239 | 204 | 18 | 17 | -0.048 |
| gray SII (same images) | 239 | 182 | 34 | 23 | -0.028 |
| gray d_kl empty context (same images) | 239 | 215 | 12 | 12 | -0.053 |
| gray SII (all 95 images) | 239 | 182 | 34 | 23 | -0.028 |

## (d) Artifact floor and resolvability (equal-area controls)

Usable images: **95**, cues: **244**; placements scored: 488 inpaint / 516 gray.

> **Unpaired gray comparison.** The inpaint control run contains no same-placement gray twins (`cg*`), so the gray side below comes from the earlier, independent gray control run (`georanker_control_results.json`: 95 images / 244 cues / 516 placements at **different random positions**). Gray vs inpaint is therefore a *distribution-level* comparison, not a per-placement one.

- inpaint control floor: median 0.0448, P90 0.1312, max 0.5952
- gray control floor (earlier run, different placements): median 0.0874, P90 0.1886, max 0.6734
- **floor gray vs inpaint**: median 0.0874 vs 0.0448 (-0.0427), P90 0.1886 vs 0.1312 (-0.0574)  _[unpaired]_
- **overall resolvability gray vs inpaint**: 48.8% (n=244) vs 76.6% (n=244)  _[unpaired]_
- resolvability (inpaint, real > max of own controls): **76.6%** (n=244)
- resolvability (gray, real > max of own controls): **48.8%** (n=244)

| category | n (inpaint) | resolvable inpaint | n (gray, earlier run) | resolvable gray |
|---|---:|---:|---:|---:|
| other | 1 | 100.0% | 1 | 0.0% |
| road/infrastructure | 10 | 100.0% | 10 | 40.0% |
| text/signage | 29 | 89.7% | 29 | 65.5% |
| landmarks/buildings | 31 | 83.9% | 31 | 51.6% |
| commercial/cultural | 42 | 78.6% | 42 | 54.8% |
| vehicles/license plates | 12 | 75.0% | 12 | 66.7% |
| environment | 78 | 69.2% | 78 | 39.7% |
| architecture | 41 | 68.3% | 41 | 43.9% |

