# Fixed-vocabulary (bottom-up) vs GPT-4o (top-down) cue coverage

Images: 10. GPT-4o cues from `cue_extract/results_sam3/`, vocabulary cues from `cue_extract/results_vocab/`, both masked with the `cue_masks_of` convention (maskable cue, union of non-degenerate instances).
`phi` = Shapley value from `belief_elicit/shapley_v2_results.json`; an *important* cue has phi above that image's median (if ties or a single cue leave nobody above the median, the image's max-phi cue(s) count as important).
`IoU` and `recall = |gpt ∩ vocab| / |gpt|` are against the single best-matching vocabulary cue. Hit thresholds: IoU >= 0.5, recall >= 0.7.

## Per image

### Bangkok, Thailand — `123515458_4aa6de7f6b_41_18684820@N00.jpg`  (1036x756)

GPT-4o cues: 3 | vocabulary cues: 7 | median phi: 0.026

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Thai text on storefront signs | text/signage | 0.026 | * | 15.3% | text or signage | 0.594 | 0.980 |
| Fujifilm signage | commercial/cultural | 0.026 | * | 15.3% | text or signage | 0.594 | 0.980 |
| Left-hand traffic with vehicles | vehicles/license plates | 0.021 |  | 1.8% | vehicle | 0.919 | 0.998 |

Vocabulary cues present: text or signage, building facade, vehicle, person's clothing, tree or vegetation, road surface, street furniture

Vocabulary cues matching NO GPT-4o cue (IoU < 0.1) — candidates GPT-4o missed:

| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |
|---|---|---|---|---|
| person's clothing | commercial/cultural | 0.5% | 0.001 | 1% |
| tree or vegetation | environment | 1.7% | 0.000 | 0% |
| road surface | road/infrastructure | 3.0% | 0.000 | 0% |
| street furniture | road/infrastructure | 1.3% | 0.001 | 2% |

### Budapest, Hungary — `312179336_340ace22c6_104_36093984@N00.jpg`  (1036x756)

GPT-4o cues: 3 | vocabulary cues: 4 | median phi: 0.164

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Cyrillic text on gravestone | text/signage | 0.262 | * | 1.5% | statue or monument | 0.933 | 0.937 |
| Orthodox cross on mausoleum | landmarks/buildings | -0.025 |  | 3.3% | building facade | 0.996 | 0.998 |
| Autumn foliage | environment | 0.164 |  | 27.3% | tree or vegetation | 0.719 | 0.731 |

Vocabulary cues present: text or signage, building facade, tree or vegetation, statue or monument

All vocabulary cues matched some GPT-4o cue.

### Chicago, United States — `169956693_5edea53724_78_70323761@N00.jpg`  (1036x672)

GPT-4o cues: 1 | vocabulary cues: 2 | median phi: 0.405

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Wrigley Field text on cup | text/signage | 0.405 | * | 20.2% | - | 0.000 | 0.000 |

Vocabulary cues present: person's clothing, street furniture

Vocabulary cues matching NO GPT-4o cue (IoU < 0.1) — candidates GPT-4o missed:

| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |
|---|---|---|---|---|
| person's clothing | commercial/cultural | 7.9% | 0.000 | 0% |
| street furniture | road/infrastructure | 4.0% | 0.000 | 0% |

### Cuba — `370717727_f9564e3587_150_13527886@N00.jpg`  (1036x756)

GPT-4o cues: 2 | vocabulary cues: 3 | median phi: 0.259

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Stone fortress walls | landmarks/buildings | 0.178 |  | 17.7% | - | 0.000 | 0.000 |
| Lush green landscape | environment | 0.341 | * | 16.1% | tree or vegetation | 0.948 | 0.948 |

Vocabulary cues present: person's clothing, tree or vegetation, mountain

Vocabulary cues matching NO GPT-4o cue (IoU < 0.1) — candidates GPT-4o missed:

| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |
|---|---|---|---|---|
| person's clothing | commercial/cultural | 1.0% | 0.000 | 0% |
| mountain | environment | 3.9% | 0.000 | 0% |

### New York, United States — `158307292_d73d7226ef_73_70323761@N00.jpg`  (672x1036)

GPT-4o cues: 4 | vocabulary cues: 6 | median phi: 0.106

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Equestrian statue | landmarks/buildings | 0.133 | * | 11.2% | statue or monument | 0.351 | 1.000 |
| American flag | commercial/cultural | 0.080 |  | 0.6% | flag | 1.000 | 1.000 |
| Historic building architecture | architecture | 0.556 | * | 10.1% | building facade | 0.984 | 1.000 |
| Trees with autumn foliage | environment | 0.054 |  | 25.6% | tree or vegetation | 0.983 | 0.998 |

Vocabulary cues present: text or signage, building facade, person's clothing, tree or vegetation, flag, statue or monument

All vocabulary cues matched some GPT-4o cue.

### Okazaki, Japan — `261517384_292417efcc_117_60558526@N00.jpg`  (1036x756)

GPT-4o cues: 4 | vocabulary cues: 5 | median phi: 0.044

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Japanese text on festival banners | text/signage | 0.055 | * | 12.4% | flag | 0.753 | 0.957 |
| Samurai armor on a person | commercial/cultural | 0.037 |  | 5.4% | person's clothing | 0.417 | 0.813 |
| Cherry blossoms | environment | 0.050 | * | 25.2% | tree or vegetation | 0.320 | 0.322 |
| Japanese police uniforms | commercial/cultural | 0.009 |  | 5.4% | person's clothing | 0.246 | 0.549 |

Vocabulary cues present: text or signage, vehicle, person's clothing, tree or vegetation, flag

Vocabulary cues matching NO GPT-4o cue (IoU < 0.1) — candidates GPT-4o missed:

| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |
|---|---|---|---|---|
| text or signage | text/signage | 0.7% | 0.040 | 70% |
| vehicle | vehicles/license plates | 5.2% | 0.000 | 0% |

### Paris, France — `166869956_84928fd11b_60_55085284@N00.jpg`  (1036x672)

GPT-4o cues: 3 | vocabulary cues: 1 | median phi: 0.029

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Golden clock with ornate design | landmarks/buildings | 0.029 |  | 0.4% | - | 0.000 | 0.000 |
| Rooftop sculptures and balustrades | architecture | 0.016 |  | 0.9% | statue or monument | 0.374 | 0.685 |
| French-style windows with wrought iron | architecture | 0.042 | * | 5.8% | - | 0.000 | 0.000 |

Vocabulary cues present: statue or monument

All vocabulary cues matched some GPT-4o cue.

### Seville, Spain — `486597679_e1a557718f_198_51162504@N00.jpg`  (1036x756)

GPT-4o cues: 3 | vocabulary cues: 1 | median phi: 0.132

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Golden religious statues | commercial/cultural | 0.132 | * | 24.8% | statue or monument | 1.000 | 1.000 |
| Crown and scepter on statue | commercial/cultural | 0.132 | * | 24.8% | statue or monument | 1.000 | 1.000 |
| Religious iconography | commercial/cultural | 0.132 | * | 24.8% | statue or monument | 1.000 | 1.000 |

Vocabulary cues present: statue or monument

All vocabulary cues matched some GPT-4o cue.

### Slovenia — `754780171_13ffd51dec_1229_78247883@N00.jpg`  (700x1036)

GPT-4o cues: 2 | vocabulary cues: 3 | median phi: 0.673

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Church on an island | landmarks/buildings | 0.749 | * | 2.3% | building facade | 0.992 | 0.996 |
| Mountainous forested background | environment | 0.597 |  | 24.6% | tree or vegetation | 0.981 | 0.983 |

Vocabulary cues present: building facade, tree or vegetation, mountain

All vocabulary cues matched some GPT-4o cue.

### Tinum, Mexico — `453636890_c425e657ab_239_55126705@N00.jpg`  (1036x756)

GPT-4o cues: 3 | vocabulary cues: 1 | median phi: 0.219

| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |
|---|---|---|---|---|---|---|---|
| Mayan architectural style | architecture | 0.219 |  | 47.4% | - | 0.000 | 0.000 |
| Stone masks and carvings | landmarks/buildings | 0.021 |  | 0.8% | - | 0.000 | 0.000 |
| Ruined stone structures | landmarks/buildings | 0.246 | * | 47.5% | - | 0.000 | 0.000 |

Vocabulary cues present: tree or vegetation

Vocabulary cues matching NO GPT-4o cue (IoU < 0.1) — candidates GPT-4o missed:

| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |
|---|---|---|---|---|
| tree or vegetation | environment | 1.5% | 0.000 | 0% |

## Summary

- **all GPT-4o cues** (n=28): IoU >= 0.5 for 0.571 (16/28); recall >= 0.7 for 0.643 (18/28); mean IoU 0.575, mean recall 0.674.
- **important only (phi > per-image median)** (n=15): IoU >= 0.5 for 0.667 (10/15); recall >= 0.7 for 0.733 (11/15); mean IoU 0.631, mean recall 0.741.

- Vocabulary cues with no GPT-4o counterpart: 11/33 (0.33 of all vocabulary cues) across 10 images.
- Vocabulary cues per image: 3.3 (GPT-4o: 2.8).
