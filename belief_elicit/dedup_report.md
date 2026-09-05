# Geometric cue de-duplication: before/after

Merge rule: pairwise mask IoU >= **0.9** (union-find over the per-cue union masks built exactly as `precompute_inpaint.cue_masks_of`). Containment (recall >= 0.95 with IoU below the merge threshold) is **reported only, never merged**.

**No re-scoring.** For a merged player `A = {a1..aj}` the members' masks are (near-)identical pixels, so the masked image of a merged subset `S'` equals — up to a handful of boundary pixels — the masked image of the ORIGINAL subset `⋃_{A∈S'} members(A)`. The gray lattice is complete (all 2^m subsets), so `v'(S') := v(⋃ members)` is read straight from the existing data. This is the single approximation in this pipeline.

How big is that approximation? For each merged group, compare every member mask against the group union: **12 of 14 groups are pixel-exact** (0 differing pixels, so `v'` is literally the already-scored value). The remaining 2 differ by 7635 px (9.35% of the 81624 px union, IoU 0.903), 846 px (0.23% of the 372224 px union, IoU 0.998).

## 1. Scale

| | before | after |
|---|---|---|
| images | 95 | 95 |
| cues / players | 244 | 223 |

- images affected: **14** / 95
- merged groups: **14**, duplicates removed: **21**
- containment pairs reported (not merged): 4

## 2. Merged groups and credit consolidation

Expectation: merging duplicates should *consolidate* credit, i.e. `φ(merged) ≈ Σφ(members, before)`.

| image | m→M | max IoU | members | Σφ before | φ after | Δ | \|Δ\|/v(N) |
|---|---|---|---|---|---|---|---|
| Spa | 4→2 | 1.000 | BorgWarner Turbo & Emissions Syste; ET3 European Championship logo; Michelin tire logo | 0.1758 | 0.1401 | -0.0357 | 25.1% |
| Bardo | 4→3 | 1.000 | Train model EN57; Polish text on train | 0.1983 | 0.1671 | -0.0312 | 9.6% |
| Paris | 5→3 | 1.000 | Bus route number and destination; French text on bus; Green and cream color scheme of bu | 0.2222 | 0.2047 | -0.0175 | 4.8% |
| Anaheim | 4→3 | 1.000 | Teacup ride; Heart design on teacup | 0.0648 | 0.0481 | -0.0167 | 16.0% |
| New York | 3→2 | 1.000 | White House; Dense urban buildings | 0.0657 | 0.0531 | -0.0126 | 13.9% |
| Venice | 4→3 | 1.000 | Venetian Gothic architecture; Canal-side location | 0.0152 | 0.0074 | -0.0078 | 7.5% |
| Tinum | 3→2 | 0.998 | Mayan architectural style; Ruined stone structures | 0.4646 | 0.4581 | -0.0066 | 1.3% |
| Bangkok | 3→2 | 1.000 | Thai text on storefront signs; Fujifilm signage | 0.0512 | 0.0452 | -0.0060 | 8.4% |
| Chaoyang District | 3→2 | 1.000 | Chinese text on building; High-rise construction | 0.0682 | 0.0642 | -0.0040 | 2.9% |
| Scotland | 3→2 | 0.903 | Rocky coastline with grassy hills; Sandy beach with rocks | 0.1008 | 0.1044 | +0.0037 | 1.7% |
| Cambridge | 4→1 | 1.000 | shopcambridgeside.com URL; Customer Service sign; Payless ShoeSource store; Inside Sale sign | 0.1258 | 0.1258 | +0.0000 | 0.0% |
| Seville | 3→1 | 1.000 | Golden religious statues; Crown and scepter on statue; Religious iconography | 0.3969 | 0.3969 | +0.0000 | 0.0% |
| Spa | 4→1 | 1.000 | Corvette text on car; ADAC logo on car; GT3 logo on car; KW logo on car | 0.1744 | 0.1744 | +0.0000 | 0.0% |
| Toronto | 2→1 | 1.000 | distinctive skyscraper with steppe; golden reflective building | 0.1862 | 0.1862 | +0.0000 | 0.0% |

Max |Δ| = **0.0357** (at most **25.1%** of that image's v(N); the worst ratio *to Σφ itself* is 51%, but that is the Venice group whose Σφ is already near zero), median |Δ| = 0.0063.

Credit is consolidated as expected. Δ = 0 **exactly** in 4 groups — those where the group swallows every cue in the image, so efficiency forces φ(merged) = v(N) = Σφ(before). In 9 of the remaining groups Δ is slightly **negative**: this is the standard merging effect for near-duplicate players — before merging, each duplicate was credited for a marginal contribution the other could equally have supplied, and the single merged player is paid that shared contribution only once. The residual is small (median 0.0063 nats/1000 km).

## 3. Per-category medians and ranking

| category | n before | n after | φ median before | φ median after | v_single median before | v_single median after |
|---|---|---|---|---|---|---|
| text/signage | 29 | 24 | 0.0803 | 0.1167 | 0.1355 | 0.1326 |
| landmarks/buildings | 31 | 28 | 0.0718 | 0.0716 | 0.1228 | 0.1197 |
| environment | 78 | 76 | 0.0604 | 0.0612 | 0.0940 | 0.0958 |
| vehicles/license plates | 12 | 11 | 0.0450 | 0.0464 | 0.1293 | 0.1111 |
| architecture | 41 | 40 | 0.0418 | 0.0432 | 0.0870 | 0.0856 |
| commercial/cultural | 42 | 33 | 0.0372 | 0.0368 | 0.0882 | 0.0743 |
| road/infrastructure | 10 | 10 | 0.0337 | 0.0337 | 0.0796 | 0.0796 |
| other | 1 | 1 | 0.0312 | 0.0312 | 0.0468 | 0.0468 |

- largest median-φ shifts: **text/signage** 0.0803 → 0.1167 (+0.0364, +45%); **vehicles/license plates** 0.0450 → 0.0464 (+0.0014, +3%); **architecture** 0.0418 → 0.0432 (+0.0014, +3%)
  De-duplication mostly moves **text/signage**: several images (Cambridge, Spa, Bardo, Bangkok, Paris) had 2-4 separately named signs segmented onto the same pixels, so consolidating them turns several small φ into one large φ while dropping the cue count (29 → 24).
- φ ranking before: text/signage > landmarks/buildings > environment > vehicles/license plates > architecture > commercial/cultural > road/infrastructure > other
- φ ranking after: text/signage > landmarks/buildings > environment > vehicles/license plates > architecture > commercial/cultural > road/infrastructure > other
- **φ ranking unchanged: yes**
- v_single ranking unchanged: NO

## 4. Interactions (Shapley Interaction Index)

| | pairs | overlap (SII < −0.01) | ≈0 (\|SII\| ≤ 0.01) | backup (SII > 0.01) | median |
|---|---|---|---|---|---|
| before | 239 | 182 | 34 | 23 | -0.0276 |
| after | 192 | 138 | 35 | 19 | -0.0242 |

- pairs whose two members ended up in the **same** merged group (i.e. pure geometric duplicates): **30** of 239 before-pairs; of those, **29** had been counted as "overlap" (SII < −0.01), median SII -0.0581.
- overlap pairs removed by de-duplication: 44 (182 → 138).

## 5. Efficiency

`Σφ = v(N)` holds on all 95 images: max |Σφ − v(N)| = 1.11e-16 before, 1.11e-16 after.

## 6. Harsanyi dividends by order (median share of Σ|d|)

| order | before | after |
|---|---|---|
| 1 | 60.5% | 65.5% |
| 2 | 32.1% | 29.7% |
| 3 | 13.2% | 11.6% |
| 4 | 3.8% | 3.2% |
| 5 | 0.8% | – |

(before: 80 images with m≥2; after: 76 images with M≥2)

## 7. Minimal sufficient set (cues needed to reach 80% of v(N))

| | images | median \|S\| | median m | share needing a single cue |
|---|---|---|---|---|
| before | 80 | 1 | 3 | 63.7% |
| after | 76 | 1 | 3 | 61.8% |

## 8. Inpaint arm

| arm | before | after |
|---|---|---|
| gray, real > max of own controls | 48.8% (n=244) | 47.1% (n=223) |
| inpaint, real > max of own controls | 76.6% (n=244) | 76.0% (n=221) |

For a merged player, *real* is `v'({A})` (all members masked together) and the artifact floor is the **max over the members' own equal-area controls** — the strictest available comparison. The inpaint after-column has n=221 rather than 223: 2 merged players group 3 originals that do not span their image, so their union was never scored under inpaint (only singles, pairs and all exist there). Both arms are essentially unchanged by de-duplication — the merged players inherit their members' resolvability rather than creating or destroying it.

- deduped order-2 anchored φ under inpaint: computable on **90** / 95 images; **5** are inpaint-incomputable because a required merged subset (a group of ≥3 originals, or a merged pair whose union has ≥3 originals) was never scored — the inpaint arm only has singles, pairs and all.
  - Anaheim (4→3 cues)
  - Bardo (4→3 cues)
  - Spa (4→2 cues)
  - Paris (5→3 cues)
  - Venice (4→3 cues)

## 9. Containment pairs (reported, not merged)

| image | recall(inner ⊂ outer) | IoU | inner | outer |
|---|---|---|---|---|
| Paris | 0.998 | 0.318 | Classical paintings | Large ornate mirror |
| Italy | 0.997 | 0.766 | Limestone cliffs | Rocky shoreline |
| Tinum | 0.975 | 0.016 | Stone masks and carvings | Ruined stone structures |
| Tinum | 0.969 | 0.016 | Stone masks and carvings | Mayan architectural style |
