# clue_leak — 逐线索 mPL 位置泄露消融

用 mPL 量化"遮掉某条(或某组)线索后,攻击者信念相对原图移动多少"。先验 = 遮掉子集 S 的图,后验 = 原图。

## 度量

对固定候选集(gallery,subset100 的唯一 GT 标签,25km 聚簇)上,某图先验/后验两个信念分布:

```
mPL(i,j) = |ln(post_i/prior_i) − ln(post_j/prior_j)| / d_ij × 1000     # nats / 1000km
```
每图对全部候选对求 **mean**(主指标,稳)。`plot_*` 里内置此计算(几何来自 `data/forward_geocode_cache.json`)。

## 组合与重合

`combo.nonempty_subsets(m)` 枚举全部非空子集(m>6 时降为 单条+leave-one-out+全集)。组合遮蔽 = 各线索掩码的**并集**(`masking.mask_solid_from_masks`,重叠像素只遮一次)。

## 文件

| 文件 | 作用 |
|---|---|
| `combo.py` | 子集枚举 |
| `masking.py` | 掩码涂灰:`mask_solid_from_masks`(不规则 SAM 掩码)/`mask_solid_regions`(方框) |
| `run_combo2.py` | 消融运行器:读 `cue_extract/results` 的掩码,逐子集打分 → `combo2_results/` |
| `run_50.py` | 批量驱动:选 N 张有可遮线索的图跑消融 + 出图 |
| `plot_one_mpl.py` | 单样本图:原图+掩码 \| 每条线索&组合的 mPL |
| `plot_clue_mpl.py` | 跨图:所有线索 mPL 排序条形图 |
| `plot_combo2.py` | 多样本多面板 + 冗余/协同打印 |
| `prep_geo100.py` | 反向/正向地理编码,建 gallery 几何缓存 |

## 输出 JSON（`combo2_results/<id>.json`）

```json
{"image_id":"...", "true_label":"City, Country", "n_maskable": 4,
 "cue_meta":[{"cue":"...","category":"...","risk":"..."}],
 "posterior": {"label": prob, ...}, "mask_type": "sam_irregular",
 "combos":[{"subset":[0,1], "cues":[...], "prior":{...},
            "kl_bits": 0.05, "prior_prob_true": 0.02}]}
```

## 运行

```bash
python -m clue_leak.prep_geo100                       # 一次性：几何缓存
python -m clue_leak.run_combo2 --ids <id1,id2,...>    # 消融（后验缺则现算，自洽）
python -m clue_leak.plot_one_mpl <id前缀>             # 出单样本图
```

## 读数提醒
- 单条 mPL = **边际泄露**(其余线索在场),冗余会压平;
- 组合价值函数**非单调**(遮超集可能 < 遮子集)→ 不做 Shapley/可加归因;
- mPL 绝对值小是正常的(全 pair 平均 + 距离归一);比的是线索**相对**大小。
