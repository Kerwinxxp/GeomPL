# GPT-4o-proposed cues vs a fixed vocabulary: which cue inventory captures the leakage?

Same 10 images, same removal operator (LaMa inpainting), same adversary (GeoRanker). Only the definition of "a cue" changes.

- **GPT-4o (top-down)**: masks `cue_extract/results_sam3/`, scores `belief_elicit/georanker_inpaint_results.json`
- **Vocabulary (bottom-up, 12 generic SAM 3 concepts)**: masks `cue_extract/results_vocab/`, scores `belief_elicit/georanker_inpaint_vocab_results.json`
- v(N) = mPL lost when the *whole* inventory is removed = total removable leakage of that inventory (nats / 1000 km). phi = order-2 anchored Shapley (`belief_elicit/order2_shapley.py`), anchored so sum_k phi_k = v(N).
- artifact null c_img = mean control-placement mPL; source per image: **inpaint** (10 images). A cue is *provisionally* sensitive if phi_k > c_img / m (pure-artifact null under which every phi equals c_img/m).
- **Caveat**: the order-2 anchored phi is exact for m <= 3 and ~8% off in magnitude at m = 4 (validated against the full 2^m gray lattice); for m >= 5 treat it as a ranking, not a magnitude. Affected here: Bangkok (vocab m=7), New York (vocab m=6), Okazaki (vocab m=5).

## Headline: total removable leakage v(N)

| image | true label | m_gpt | m_vocab | v(N) GPT-4o | v(N) vocab | winner | vocab/gpt |
|---|---|---:|---:|---:|---:|:--:|---:|
| `123515458_4aa6de7f6b` | Bangkok, Thailand | 3 | 7 | 0.0746 | 0.1291 | **vocab** | 1.73x |
| `312179336_340ace22c6` | Budapest, Hungary | 3 | 4 | 0.2929 | 0.2711 | **GPT-4o** | 0.93x |
| `169956693_5edea53724` | Chicago, United States | 1 | 2 | 0.4719 | 0.1751 | **GPT-4o** | 0.37x |
| `370717727_f9564e3587` | Cuba | 2 | 3 | 0.2937 | 0.1418 | **GPT-4o** | 0.48x |
| `158307292_d73d7226ef` | New York, United States | 4 | 6 | 1.0212 | 0.8181 | **GPT-4o** | 0.80x |
| `261517384_292417efcc` | Okazaki, Japan | 4 | 5 | 0.1779 | 0.0980 | **GPT-4o** | 0.55x |
| `166869956_84928fd11b` | Paris, France | 3 | 1 | 0.0541 | 0.0467 | **GPT-4o** | 0.86x |
| `486597679_e1a557718f` | Seville, Spain | 3 | 1 | 0.4608 | 0.4608 | **tie** | 1.00x |
| `754780171_13ffd51dec` | Slovenia | 2 | 3 | 1.3896 | 1.3335 | **GPT-4o** | 0.96x |
| `453636890_c425e657ab` | Tinum, Mexico | 3 | 1 | 0.6886 | 0.0288 | **GPT-4o** | 0.04x |
| **total** | 10 images | 28 | 33 | **4.9254** | **3.5030** | **GPT-4o** | 0.71x |

Per-image winner counts: GPT-4o **8**, vocabulary **1**, exact ties **1** (of 10). Median per-image ratio v(N)_vocab / v(N)_gpt = **0.83x**.

## Per-image cue tables

### Bangkok, Thailand — `123515458_4aa6de7f6b_41_18684820@N00.jpg`

v(N) GPT-4o **0.0746** (3 cues) vs vocab **0.1291** (7 cues); c_img = 0.0410 (inpaint control) → threshold 0.0137 (GPT-4o) / 0.0059 (vocab). Order-2 residual share 0.361 / 3.873.  _phi magnitudes approximate where m >= 4._

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Thai text on storefront signs | text/signage | 15.3% | 0.0676 | 0.0293 | YES | text or signage (IoU 0.59, rec 0.98) |
| 1 | Fujifilm signage | commercial/cultural | 15.3% | 0.0676 | 0.0293 | YES | text or signage (IoU 0.59, rec 0.98) |
| 2 | Left-hand traffic with vehicles | vehicles/license plates | 1.8% | 0.0339 | 0.0160 | YES | vehicle (IoU 0.92, rec 1.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | text or signage | text/signage | 24.9% | 0.1161 | 0.0384 | YES | Thai text on storefront signs (IoU 0.59, rec 0.98) |
| 1 | building facade | architecture | 35.9% | 0.0730 | 0.0298 | YES | Thai text on storefront signs (IoU 0.39, rec 0.94) |
| 2 | vehicle | vehicles/license plates | 2.0% | 0.0411 | 0.0060 | YES | Left-hand traffic with vehicles (IoU 0.92, rec 1.00) |
| 3 | person's clothing | commercial/cultural | 0.5% | 0.0318 | 0.0124 | YES | _no match_ (best IoU 0.00) |
| 4 | tree or vegetation | environment | 1.7% | 0.0337 | 0.0099 | YES | _no match_ (best IoU 0.00) |
| 5 | road surface | road/infrastructure | 3.0% | 0.0345 | 0.0231 | YES | _no match_ (best IoU 0.00) |
| 6 | street furniture | road/infrastructure | 1.3% | 0.0360 | 0.0096 | YES | _no match_ (best IoU 0.00) |

### Budapest, Hungary — `312179336_340ace22c6_104_36093984@N00.jpg`

v(N) GPT-4o **0.2929** (3 cues) vs vocab **0.2711** (4 cues); c_img = 0.1223 (inpaint control) → threshold 0.0408 (GPT-4o) / 0.0306 (vocab). Order-2 residual share 0.250 / 1.219.  _phi magnitudes approximate where m >= 4._

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Cyrillic text on gravestone | text/signage | 1.5% | 0.2925 | 0.2070 | YES | statue or monument (IoU 0.93, rec 0.94) |
| 1 | Orthodox cross on mausoleum | landmarks/buildings | 3.3% | 0.1564 | -0.0139 | - | building facade (IoU 1.00, rec 1.00) |
| 2 | Autumn foliage | environment | 27.3% | 0.1593 | 0.0997 | YES | tree or vegetation (IoU 0.72, rec 0.73) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | text or signage | text/signage | 0.6% | 0.0775 | 0.0414 | YES | Cyrillic text on gravestone (IoU 0.36, rec 0.36) |
| 1 | building facade | architecture | 3.3% | 0.1430 | 0.0205 | - | Orthodox cross on mausoleum (IoU 1.00, rec 1.00) |
| 2 | tree or vegetation | environment | 20.5% | 0.1771 | 0.0510 | YES | Autumn foliage (IoU 0.72, rec 0.73) |
| 3 | statue or monument | landmarks/buildings | 1.5% | 0.2984 | 0.1582 | YES | Cyrillic text on gravestone (IoU 0.93, rec 0.94) |

### Chicago, United States — `169956693_5edea53724_78_70323761@N00.jpg`

v(N) GPT-4o **0.4719** (1 cues) vs vocab **0.1751** (2 cues); c_img = 0.1271 (inpaint control) → threshold 0.1271 (GPT-4o) / 0.0635 (vocab). Order-2 residual share 0.000 / 0.000.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Wrigley Field text on cup | text/signage | 20.2% | 0.4719 | 0.4719 | YES | _no match_ (best IoU 0.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | person's clothing | commercial/cultural | 7.9% | 0.0920 | 0.0979 | YES | _no match_ (best IoU 0.00) |
| 1 | street furniture | road/infrastructure | 4.0% | 0.0714 | 0.0772 | YES | _no match_ (best IoU 0.00) |

### Cuba — `370717727_f9564e3587_150_13527886@N00.jpg`

v(N) GPT-4o **0.2937** (2 cues) vs vocab **0.1418** (3 cues); c_img = 0.0509 (inpaint control) → threshold 0.0255 (GPT-4o) / 0.0170 (vocab). Order-2 residual share 0.000 / 0.183.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Stone fortress walls | landmarks/buildings | 17.7% | 0.2493 | 0.0383 | YES | _no match_ (best IoU 0.00) |
| 1 | Lush green landscape | environment | 16.1% | 0.4665 | 0.2555 | YES | tree or vegetation (IoU 0.95, rec 0.95) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | person's clothing | commercial/cultural | 1.0% | 0.1181 | 0.0515 | YES | _no match_ (best IoU 0.00) |
| 1 | tree or vegetation | environment | 15.2% | 0.1678 | 0.0897 | YES | Lush green landscape (IoU 0.95, rec 0.95) |
| 2 | mountain | environment | 3.9% | 0.0738 | 0.0006 | - | _no match_ (best IoU 0.00) |

### New York, United States — `158307292_d73d7226ef_73_70323761@N00.jpg`

v(N) GPT-4o **1.0212** (4 cues) vs vocab **0.8181** (6 cues); c_img = 0.0876 (inpaint control) → threshold 0.0219 (GPT-4o) / 0.0146 (vocab). Order-2 residual share 0.415 / 0.600.  _phi magnitudes approximate where m >= 4._

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Equestrian statue | landmarks/buildings | 11.2% | 0.2514 | 0.2646 | YES | statue or monument (IoU 0.35, rec 1.00) |
| 1 | American flag | commercial/cultural | 0.6% | 0.0494 | 0.1922 | YES | flag (IoU 1.00, rec 1.00) |
| 2 | Historic building architecture | architecture | 10.1% | 0.4287 | 0.4621 | YES | building facade (IoU 0.98, rec 1.00) |
| 3 | Trees with autumn foliage | environment | 25.6% | 0.2423 | 0.1024 | YES | tree or vegetation (IoU 0.98, rec 1.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | text or signage | text/signage | 1.8% | 0.0651 | 0.1420 | YES | American flag (IoU 0.32, rec 1.00) |
| 1 | building facade | architecture | 10.2% | 0.6632 | 0.4220 | YES | Historic building architecture (IoU 0.98, rec 1.00) |
| 2 | person's clothing | commercial/cultural | 2.7% | 0.1200 | -0.0661 | - | Equestrian statue (IoU 0.19, rec 0.20) |
| 3 | tree or vegetation | environment | 25.9% | 0.1699 | -0.0049 | - | Trees with autumn foliage (IoU 0.98, rec 1.00) |
| 4 | flag | commercial/cultural | 0.6% | 0.0494 | 0.1402 | YES | American flag (IoU 1.00, rec 1.00) |
| 5 | statue or monument | landmarks/buildings | 31.9% | 0.2164 | 0.1848 | YES | Equestrian statue (IoU 0.35, rec 1.00) |

### Okazaki, Japan — `261517384_292417efcc_117_60558526@N00.jpg`

v(N) GPT-4o **0.1779** (4 cues) vs vocab **0.0980** (5 cues); c_img = 0.0486 (inpaint control) → threshold 0.0122 (GPT-4o) / 0.0097 (vocab). Order-2 residual share 0.565 / 2.524.  _phi magnitudes approximate where m >= 4._

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Japanese text on festival banners | text/signage | 12.4% | 0.0770 | 0.0549 | YES | flag (IoU 0.75, rec 0.96) |
| 1 | Samurai armor on a person | commercial/cultural | 5.4% | 0.0500 | 0.0337 | YES | person's clothing (IoU 0.42, rec 0.81) |
| 2 | Cherry blossoms | environment | 25.2% | 0.0895 | 0.0585 | YES | tree or vegetation (IoU 0.32, rec 0.32) |
| 3 | Japanese police uniforms | commercial/cultural | 5.4% | 0.0383 | 0.0309 | YES | person's clothing (IoU 0.25, rec 0.55) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | text or signage | text/signage | 0.7% | 0.0344 | 0.0085 | - | _no match_ (best IoU 0.04) |
| 1 | vehicle | vehicles/license plates | 5.2% | 0.0661 | 0.0127 | YES | _no match_ (best IoU 0.00) |
| 2 | person's clothing | commercial/cultural | 9.6% | 0.0337 | 0.0277 | YES | Samurai armor on a person (IoU 0.42, rec 0.81) |
| 3 | tree or vegetation | environment | 8.3% | 0.0608 | 0.0168 | YES | Cherry blossoms (IoU 0.32, rec 0.32) |
| 4 | flag | commercial/cultural | 15.3% | 0.0808 | 0.0322 | YES | Japanese text on festival banners (IoU 0.75, rec 0.96) |

### Paris, France — `166869956_84928fd11b_60_55085284@N00.jpg`

v(N) GPT-4o **0.0541** (3 cues) vs vocab **0.0467** (1 cues); c_img = 0.0312 (inpaint control) → threshold 0.0104 (GPT-4o) / 0.0312 (vocab). Order-2 residual share 0.368 / 0.029.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Golden clock with ornate design | landmarks/buildings | 0.4% | 0.0388 | 0.0268 | YES | _no match_ (best IoU 0.00) |
| 1 | Rooftop sculptures and balustrades | architecture | 0.9% | 0.0377 | 0.0170 | YES | statue or monument (IoU 0.37, rec 0.68) |
| 2 | French-style windows with wrought iron | architecture | 5.8% | 0.0230 | 0.0103 | - | _no match_ (best IoU 0.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | statue or monument | landmarks/buildings | 1.3% | 0.0481 | 0.0467 | YES | Rooftop sculptures and balustrades (IoU 0.37, rec 0.68) |

### Seville, Spain — `486597679_e1a557718f_198_51162504@N00.jpg`

v(N) GPT-4o **0.4608** (3 cues) vs vocab **0.4608** (1 cues); c_img = 0.0898 (inpaint control) → threshold 0.0299 (GPT-4o) / 0.0898 (vocab). Order-2 residual share 1.000 / 0.000.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Golden religious statues | commercial/cultural | 24.8% | 0.4608 | 0.1536 | YES | statue or monument (IoU 1.00, rec 1.00) |
| 1 | Crown and scepter on statue | commercial/cultural | 24.8% | 0.4608 | 0.1536 | YES | statue or monument (IoU 1.00, rec 1.00) |
| 2 | Religious iconography | commercial/cultural | 24.8% | 0.4608 | 0.1536 | YES | statue or monument (IoU 1.00, rec 1.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | statue or monument | landmarks/buildings | 24.8% | 0.4608 | 0.4608 | YES | Golden religious statues (IoU 1.00, rec 1.00) |

### Slovenia — `754780171_13ffd51dec_1229_78247883@N00.jpg`

v(N) GPT-4o **1.3896** (2 cues) vs vocab **1.3335** (3 cues); c_img = 0.0368 (inpaint control) → threshold 0.0184 (GPT-4o) / 0.0123 (vocab). Order-2 residual share 0.000 / 0.271.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Church on an island | landmarks/buildings | 2.3% | 0.3817 | 0.7187 | YES | building facade (IoU 0.99, rec 1.00) |
| 1 | Mountainous forested background | environment | 24.6% | 0.3339 | 0.6709 | YES | tree or vegetation (IoU 0.98, rec 0.98) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | building facade | architecture | 2.3% | 0.3777 | 0.7850 | YES | Church on an island (IoU 0.99, rec 1.00) |
| 1 | tree or vegetation | environment | 24.3% | 0.3464 | 0.3709 | YES | Mountainous forested background (IoU 0.98, rec 0.98) |
| 2 | mountain | environment | 12.3% | 0.1942 | 0.1776 | YES | Mountainous forested background (IoU 0.49, rec 0.50) |

### Tinum, Mexico — `453636890_c425e657ab_239_55126705@N00.jpg`

v(N) GPT-4o **0.6886** (3 cues) vs vocab **0.0288** (1 cues); c_img = 0.0572 (inpaint control) → threshold 0.0191 (GPT-4o) / 0.0572 (vocab). Order-2 residual share 0.077 / 0.000.

**GPT-4o cues**

| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | Mayan architectural style | architecture | 47.4% | 0.6271 | 0.3283 | YES | _no match_ (best IoU 0.00) |
| 1 | Stone masks and carvings | landmarks/buildings | 0.8% | 0.0291 | -0.0060 | - | _no match_ (best IoU 0.00) |
| 2 | Ruined stone structures | landmarks/buildings | 47.5% | 0.6513 | 0.3662 | YES | _no match_ (best IoU 0.00) |

**Vocabulary cues**

| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |
|---:|---|---|---:|---:|---:|:--:|---|
| 0 | tree or vegetation | environment | 1.5% | 0.0288 | 0.0288 | - | _no match_ (best IoU 0.00) |

## Aggregates

### (i) Which inventory removes more leakage?

- sum over 10 images of v(N): GPT-4o **4.9254** vs vocabulary **3.5030** → **GPT-4o** removes more (vocab/gpt = 0.71x)
- per-image winners: GPT-4o 8 / vocabulary 1 / exact tie 1
- mean v(N): 0.4925 (GPT-4o) vs 0.3503 (vocab); cues scored: 28 vs 33

### (ii) Vocabulary cues with NO GPT-4o counterpart (IoU < 0.1) — did GPT-4o miss leaky regions?

- unmatched vocabulary cues: **11** of 33; sensitive: **8** (73%)
- matched vocabulary cues: 22; sensitive: 19 (86%)
- phi median: unmatched **0.0127** vs matched **0.0489** (mean 0.0302 vs 0.1441; unmatched range 0.0006..0.0979)

| image | vocab cue | category | area | phi | sensitive |
|---|---|---|---:|---:|:--:|
| Chicago, United States | person's clothing | commercial/cultural | 7.9% | 0.0979 | YES |
| Chicago, United States | street furniture | road/infrastructure | 4.0% | 0.0772 | YES |
| Cuba | person's clothing | commercial/cultural | 1.0% | 0.0515 | YES |
| Tinum, Mexico | tree or vegetation | environment | 1.5% | 0.0288 | - |
| Bangkok, Thailand | road surface | road/infrastructure | 3.0% | 0.0231 | YES |
| Okazaki, Japan | vehicle | vehicles/license plates | 5.2% | 0.0127 | YES |
| Bangkok, Thailand | person's clothing | commercial/cultural | 0.5% | 0.0124 | YES |
| Bangkok, Thailand | tree or vegetation | environment | 1.7% | 0.0099 | YES |
| Bangkok, Thailand | street furniture | road/infrastructure | 1.3% | 0.0096 | YES |
| Okazaki, Japan | text or signage | text/signage | 0.7% | 0.0085 | - |
| Cuba | mountain | environment | 3.9% | 0.0006 | - |

### (iii) GPT-4o cues with no vocabulary counterpart — what the fixed vocabulary would lose

- unmatched GPT-4o cues: **7** of 28; sensitive: **5** (71%)
- matched GPT-4o cues: 21; sensitive: 20 (95%)
- phi median: unmatched **0.0383** vs matched **0.1024**

| image | GPT-4o cue | category | area | phi | sensitive |
|---|---|---|---:|---:|:--:|
| Chicago, United States | Wrigley Field text on cup | text/signage | 20.2% | 0.4719 | YES |
| Tinum, Mexico | Ruined stone structures | landmarks/buildings | 47.5% | 0.3662 | YES |
| Tinum, Mexico | Mayan architectural style | architecture | 47.4% | 0.3283 | YES |
| Cuba | Stone fortress walls | landmarks/buildings | 17.7% | 0.0383 | YES |
| Paris, France | Golden clock with ornate design | landmarks/buildings | 0.4% | 0.0268 | YES |
| Paris, France | French-style windows with wrought iron | architecture | 5.8% | 0.0103 | - |
| Tinum, Mexico | Stone masks and carvings | landmarks/buildings | 0.8% | -0.0060 | - |

By category (unmatched GPT-4o cues):

| category | n | sensitive | cues |
|---|---:|---:|---|
| landmarks/buildings | 4 | 3 | Stone fortress walls; Golden clock with ornate design; Stone masks and carvings; Ruined stone structures |
| architecture | 2 | 1 | French-style windows with wrought iron; Mayan architectural style |
| text/signage | 1 | 1 | Wrigley Field text on cup |

### (iv) Matched pairs (IoU >= 0.5, mutual best match): do the two inventories agree on phi?

- pairs: **13**; Spearman(phi_gpt, phi_vocab) = **0.802** (Pearson 0.837)
- sensitive-flag agreement: **12/13** (both sensitive 11, neither 1, GPT-4o only 1, vocab only 0)

| image | GPT-4o cue | vocab cue | IoU | phi_gpt | phi_vocab | sens gpt/vocab |
|---|---|---|---:|---:|---:|:--:|
| Bangkok, Thailand | Thai text on storefront signs | text or signage | 0.59 | 0.0293 | 0.0384 | Y/Y |
| Bangkok, Thailand | Left-hand traffic with vehicles | vehicle | 0.92 | 0.0160 | 0.0060 | Y/Y |
| Budapest, Hungary | Cyrillic text on gravestone | statue or monument | 0.93 | 0.2070 | 0.1582 | Y/Y |
| Budapest, Hungary | Orthodox cross on mausoleum | building facade | 1.00 | -0.0139 | 0.0205 | -/- |
| Budapest, Hungary | Autumn foliage | tree or vegetation | 0.72 | 0.0997 | 0.0510 | Y/Y |
| Cuba | Lush green landscape | tree or vegetation | 0.95 | 0.2555 | 0.0897 | Y/Y |
| New York, United States | American flag | flag | 1.00 | 0.1922 | 0.1402 | Y/Y |
| New York, United States | Historic building architecture | building facade | 0.98 | 0.4621 | 0.4220 | Y/Y |
| New York, United States | Trees with autumn foliage | tree or vegetation | 0.98 | 0.1024 | -0.0049 | Y/- |
| Okazaki, Japan | Japanese text on festival banners | flag | 0.75 | 0.0549 | 0.0322 | Y/Y |
| Seville, Spain | Golden religious statues | statue or monument | 1.00 | 0.1536 | 0.4608 | Y/Y |
| Slovenia | Church on an island | building facade | 0.99 | 0.7187 | 0.7850 | Y/Y |
| Slovenia | Mountainous forested background | tree or vegetation | 0.98 | 0.6709 | 0.3709 | Y/Y |

### (v) Hybrid inventory: union of sensitive regions across both sides

- sensitive cues: GPT-4o **25**, vocabulary **27**
- union of sensitive *regions* (merging pairs with IoU >= 0.5): **38** — 11 found by both, **11** only by GPT-4o, **16** only by the vocabulary

| image | sens gpt | sens vocab | union | both | gpt only | vocab only |
|---|---:|---:|---:|---:|---:|---:|
| Bangkok, Thailand | 3 | 7 | 7 | 2 | 0 | 5 |
| Budapest, Hungary | 2 | 3 | 3 | 2 | 0 | 1 |
| Chicago, United States | 1 | 2 | 3 | 0 | 1 | 2 |
| Cuba | 2 | 2 | 3 | 1 | 1 | 1 |
| New York, United States | 4 | 4 | 6 | 2 | 2 | 2 |
| Okazaki, Japan | 4 | 4 | 7 | 1 | 3 | 3 |
| Paris, France | 2 | 1 | 3 | 0 | 2 | 1 |
| Seville, Spain | 3 | 1 | 1 | 1 | 0 | 0 |
| Slovenia | 2 | 3 | 3 | 2 | 0 | 1 |
| Tinum, Mexico | 2 | 0 | 2 | 0 | 2 | 0 |

## Data consistency warnings

Cue order and naming agree across the mask files, the manifests, the sweep `per_cue` order and the scored result records — any mismatch would be listed here. What *is* listed below are geometrically **duplicate cues inside one inventory**: SAM 3 grounded two differently-named cues to the same pixels, which inflates m and splits phi across redundant entries.

- 123515458_4aa6de7f6b_41_18684820@N00.jpg [gpt]: duplicate masks IoU=1.000 — "Thai text on storefront signs" vs "Fujifilm signage"
- 453636890_c425e657ab_239_55126705@N00.jpg [gpt]: duplicate masks IoU=0.998 — "Mayan architectural style" vs "Ruined stone structures"
- 486597679_e1a557718f_198_51162504@N00.jpg [gpt]: duplicate masks IoU=1.000 — "Golden religious statues" vs "Crown and scepter on statue"
- 486597679_e1a557718f_198_51162504@N00.jpg [gpt]: duplicate masks IoU=1.000 — "Golden religious statues" vs "Religious iconography"
- 486597679_e1a557718f_198_51162504@N00.jpg [gpt]: duplicate masks IoU=1.000 — "Crown and scepter on statue" vs "Religious iconography"

