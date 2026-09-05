# Inpainting vs gray masking: per-cue location leakage

- results file: `C:\Users\phdwf\OneDrive\Desktop\GeoBayes\belief_elicit\georanker_inpaint_vocab_results.json`
- cue metadata: `manifest` (cache `C:\Users\phdwf\OneDrive\Desktop\GeoBayes\belief_elicit\inpaint_cache_vocab`)
- gray baselines: NOT used (vocabulary run)
- images in results file: **10** (scored variants: 102)
- generated from partial data: sections report their own usable-image counts

## (a) Single-cue removal: gray vs inpaint

Usable images: **10**, cues: **33** (paired with a gray baseline: 0).

- no gray baseline available; median inpaint mPL = 0.0775

| category | n | inpaint median | gray median |
|---|---:|---:|---:|
| architecture | 4 | 0.2604 | n/a |
| landmarks/buildings | 4 | 0.2574 | n/a |
| environment | 9 | 0.1678 | n/a |
| commercial/cultural | 7 | 0.0808 | n/a |
| text/signage | 4 | 0.0713 | n/a |
| vehicles/license plates | 2 | 0.0536 | n/a |
| road/infrastructure | 3 | 0.0360 | n/a |

## (b) Order-2 anchored Shapley under inpainting

phi_k = d_k + 1/2 * sum_l d_kl, then anchored so that sum_k phi_k = v(N).

Usable images (all singles + all pairs + all): **10**, cues: **33**; paired with gray exact phi: 0 images / 0 cues.

- median truncation residual share |v(N)-sum phi^(2)|/|v(N)| = **0.227**

| category | n | phi_inpaint median | phi_gray median |
|---|---:|---:|---:|
| architecture | 4 | 0.2259 | n/a |
| landmarks/buildings | 4 | 0.1715 | n/a |
| text/signage | 4 | 0.0399 | n/a |
| commercial/cultural | 7 | 0.0322 | n/a |
| environment | 9 | 0.0288 | n/a |
| road/infrastructure | 3 | 0.0231 | n/a |
| vehicles/license plates | 2 | 0.0093 | n/a |

## (c) Non-additivity under inpainting

Usable images (singles + all, m>=2): **7**; pairs with an interaction value: **59**.

- fraction sub-additive (v(N) < sum_k v({k})) = **71.4%**
- median v(N) / sum_k v({k}) = **0.394**

| interaction estimator | n | d < -0.01 | ~0 | d > 0.01 | median |
|---|---:|---:|---:|---:|---:|
| inpaint d_kl (empty context) | 59 | 50 | 1 | 8 | -0.034 |

## (d) Artifact floor and resolvability (equal-area controls)

Usable images: **10**, cues: **28**; placements scored: 56 inpaint / 0 gray.

- inpaint control floor: median 0.0487, P90 0.1211, max 0.2182
- resolvability (inpaint, real > max of own controls): **72.7%** (n=22)

| category | n (inpaint) | resolvable inpaint | n (gray) | resolvable gray |
|---|---:|---:|---:|---:|
| architecture | 4 | 100.0% | 0 | n/a |
| landmarks/buildings | 2 | 100.0% | 0 | n/a |
| vehicles/license plates | 2 | 100.0% | 0 | n/a |
| text/signage | 4 | 75.0% | 0 | n/a |
| commercial/cultural | 4 | 50.0% | 0 | n/a |
| environment | 6 | 50.0% | 0 | n/a |

