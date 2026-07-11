# GeoBayes 复现报告（REPRODUCTION_REPORT）

> 对象：Shi et al., *GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating*, AAAI-26, pp. 8997–9005。
> 复现者目标（**非全表复现**）：对一张图片，得到攻击者的**初始先验**与**看完全部线索后的最终后验**两个概率分布，用于 location-privacy 威胁建模。
> 日期：2026-07-04 · 模型：GPT-4o（OpenAI 兼容端点）· 检索：Tavily（文本）· 评测经多代理对抗审计。

> **修订说明（重要）**：本报告早期版本用"国家名精确匹配"作准确率口径，得出了"更新有净负贡献 / 攻击者能力双峰"的结论。经复核，那个口径错误（论文用 GPS 距离阈值），相关结论已**撤回**。本版全部改用论文的距离阈值口径，并新增了论文真正的 zero-shot baseline 对照。结论随之实质改变，见 §1、§5。

---

## 1. 摘要与核心发现（经距离阈值口径 + 对抗审计）

我们复现了 GeoBayes 的概率核心（Eq.5 先验 / Eq.6 似然 / Eq.7 更新，Fig.3 全链数字精确对拍）与两阶段代理系统（v1 国家单层无搜索；v2 层级转移 + WebSearch）。在 Im2GPS3k 随机 50 图子集、GPT-4o 上，用论文的 GPS 距离阈值口径得到三个发现：

1. **朴素单次提问的 GPT-4o 攻击者极强**：让模型一次性直接给"街道, 城市, 国家"（zero-shot，无任何框架），50 图中 **48% 落在 25km 内、12% 落在 1km 内**——甚至高于论文里 Qwen2.5-VL 的 zero-shot（31%@25km）。对隐私威胁而言，这是一个**无需任何工具、仅靠一次提问就已很高的攻击基线**。

2. **我们复现的 GeoBayes 反而显著弱于它自己的 zero-shot 基线**（25km：10% vs 48%；1km：2% vs 12%），方向与论文**相反**（论文中 GeoBayes > zero-shot）。

3. **经四视角对抗审计，(2) 的主因是"下钻瓶颈"这一实现/参数缺陷，而非"脚手架对强模型有害"的根本结论**（后者审计判定 **不成立，已撤回**）。具体：
   - **转移门几乎打不开**：转移要求某层 `maxP ≥ τ_transition=0.7`，但 46 张国家层图中仅 **7 张**达到 0.7，其余 39 张终止在国家层 → geocode 成国家质心 → 细尺度几乎必错。
   - **先验天花板**：`k=5`（我方假设，论文未规定候选数）叠加 Eq.5 截断（τp=0.6, T=1.5）使任一候选先验上限仅约 **0.27**（实测最大 0.255），要靠证据累积到 0.7 极难，故转移门结构性地难以触发。
   - **缺 Google Lens**：论文的下钻引擎是 ImageSearch（Google Lens），我们只接了 Tavily 文本检索（`image_search` 已实现但管线从未调用）。这是一处真实且已记录的偏差。
   - 于是 GPT-4o 本已知道"卡普里岛 Marina Piccola"，却被框架强制先压成"Italy"（国家质心距真值 267km），且多数图无法下钻回去——细粒度知识被丢弃。

**对隐私研究的诚实结论**：在强 MLLM（GPT-4o）上，**最强攻击基线来自朴素单次提问**；我们当前参数下的 GeoBayes 脚手架并未增强、反而节流了这个攻击者。这是**复现质量的局限**，不是"GeoBayes 方法在强模型上失效"的证据——论文的增益建立在更弱的模型 + 图像检索之上，其是否迁移到强模型，本复现**尚未公正检验**（需补 Google Lens + 修转移门，见 §8）。架构本身忠于论文（国家优先 + 门控下钻，Eq.11 允许国家层输出）。

---

## 2. 精确复现命令

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # 或 DASHSCOPE_API_KEY（Qwen 锚点）
export TAVILY_API_KEY=tvly-...        # 仅 v2

python -m pytest tests/ -q            # 125 passed（含 Fig.3 全链黄金测试）

python run.py <image.jpg> -o out.json          # 单图
python scripts/build_subset.py --n 50          # 抽样 + Flickr 取图 + Nominatim 反编码
python scripts/run_batch.py                    # GeoBayes 全子集（config.yaml 控 v1/v2）
python scripts/run_zeroshot.py                 # zero-shot baseline
python scripts/eval_distance.py                # 三方距离阈值统一评测（论文口径）
```

## 3. 环境与版本

| 项 | 值 |
|---|---|
| Python | 3.14.2 · openai 2.44.0 · pillow 12.3.0 · PyYAML 6.0.3 · pytest 9.1.1 |
| MLLM | GPT-4o（temperature=0，本地 smart_resize 后上传）|
| 检索 | Tavily 文本；SerpApi Google Lens **未接入** |
| GPU | 无（纯 API）|

## 4. 数据来源与完整性

- **GT 坐标**：Im2GPS3k 元数据 CSV（GeoRanker 镜像，2997 行）。
- **图片**：按 IMG_ID 重建 Flickr 静态 URL 按需下载。**存活偏差**：抽样中 19/69 张因 404/过小被跳过 → 子集偏向仍在线图片（`data/subset50_meta.json`）。
- **GT 国家/前向地名**：Nominatim 反向/前向地理编码（1 req/s + 缓存）。
- **子集**：seed=42，50 图，46 张国家层起步、4 张因决定性地标直接从城市层起步。

## 5. 实测结果

### 5.1 距离阈值准确率（论文 Table 1 口径，公平公共分母 n=50，无法编码=未命中）

| 方法 | <1km | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|
| **zero-shot（GPT-4o 直接）** | **12.0** | **48.0** | **60.0** | **76.0** | **84.0** |
| GeoBayes v1（搜索关） | 0.0 | 6.0 | 14.0 | 44.0 | 80.0 |
| GeoBayes v2（搜索开） | 2.0 | 10.0 | 16.0 | 44.0 | 76.0 |
| 论文 Qwen2.5-VL zero-shot | 5.1 | 31.0 | 51.1 | 70.4 | 83.8 |
| 论文 Qwen2.5-VL GeoBayes | 6.3 | 34.7 | 53.6 | 73.7 | 85.9 |

注：zero-shot 有 4 张为细粒度但 Nominatim 无法编码的地名（如 "Amédée Island Lighthouse"），本表按未命中计（对 zero-shot 保守）；用更好的 geocoder 其分数只会更高。绝对值与论文不可直接比（模型不同、n=50、存活偏差）。

### 5.2 v2 相对 v1（搜索的作用，同图同缓存）

搜索**在细尺度小幅改善** v1（25km 6.0→10.0、1km 0.0→2.0），方向与论文一致（Enhance/Hierarchy 只在细尺度起效）；但在粗尺度略降（2500km 80→76，因泛化线索检索加噪）。改善幅度远不足以逼近 zero-shot 上限——因为只有 11/50 张成功下钻。

### 5.3 下钻瓶颈诊断（审计确认）

- **输出特异性**：zero-shot 46/50 张给到城市或更细（34 街道 / 12 城市 / 4 仅国家）；GeoBayes 仅 **11/50** 达细粒度，**39/50 终止在国家层**。
- **转移门**：46 张国家层图仅 7 张 `maxP≥0.7` 触发转移；另有 4 张 maxP∈[0.60,0.70) 差一点。先验上限约 0.27（k=5 + Eq.5）。
- **典型对比**：zero-shot "Marina Piccola, Capri"（2km）vs GeoBayes "Italy"（267km）；"Beijing"（6km）vs "China"（895km）；"Cusco, Peru"（1km）vs "Peru"（811km）。

### 5.4 先验质量（Table 3 口径，n=46）

country Top-1/3/5 recall = 60.9 / 76.1 / 80.4，支撑覆盖率 80.4%（论文 Qwen 70.4/77.6/81.5，量级参照）。

## 6. 实现假设与偏差（三级标签，详见 `docs/paper_to_code_map.md`）

- `[paper]`：τp=0.6、T=1.5、β=ln2、τ_transition=0.7、τ_enhance=0.05、每步归一化 + 层内不剪枝（Fig.3 反推锁定）。
- `[assumption]`（关键）：**全部 prompt**；**k=5**（论文未规定，与 Eq.5 交互造成先验天花板，是下钻瓶颈的主因之一）；judge 对比式打分且不可见后验；停止阈值 τ=τ_transition=0.7；Replace 证伪门；`max_replace=0`（无搜索下 Replace 净负，实证偏离）。
- **与论文的偏差**：模型 GPT-4o 而非 Qwen；**检索仅 Tavily 文本，缺 Google Lens 图像检索**（论文下钻主力）；未做全量评测与 Table 1/2/5 数字复现。

## 7. 正确性验证

- **125 单元测试全过**，重心：Eq.5/6/7 性质 + **Fig.3 三层全链黄金测试**（对拍论文数字，3 位小数）+ mock 控制流触发矩阵 + 距离评测口径（含"无法编码=未命中"公平分母）。
- 三方距离对照经**四视角对抗审计**（公平性 / 方法学 / 根因 / 论文忠实度）+ 逐条对抗验证，修复了 evaluator 分母偏袒 zero-shot 的 bug（改用公共分母 n=50）。

## 8. 局限与下一步（使对照公正的明确路径）

1. **修转移门以释放下钻**（最关键，最便宜）：现 `τ_transition=0.7` 在 k=5 先验天花板下几乎不可达。降低 τ_transition（如 0.4–0.5）或调整候选结构，重跑 v2，检验下钻频率与细尺度准确率是否回升——这将**直接判定** §1(3) 的"瓶颈说"是否正确。
2. **接入 Google Lens（SerpApi）图像检索**：论文的下钻引擎，预计是逼近/反超 zero-shot 的关键。
3. **Qwen2.5-VL 锚点**：用论文原模型严格对拍 Table 1/3（DashScope，几美元），排除"GPT-4o 特有"的混淆。
4. **扩样本至 200–300 张** + 去存活偏差；用更鲁棒的 geocoder 消除 4 张 ungeocodable 的口径噪声。
5. 未解分歧：在强模型上 GeoBayes 能否超过 zero-shot，本复现**未公正检验**；(1)(2) 完成前不下定论。

## 9. 复现产物清单

```
geobayes/           核心包（core/ mllm/ search/ eval/ analysis/）
tests/              125 单元测试
docs/paper_to_code_map.md   方程→代码映射 + 歧义决策
config.yaml         全部超参与开关（含来源标签）
results_zeroshot/   zero-shot baseline 逐图输出
results_baseline/   GeoBayes 搜索 OFF（v1）
results/            GeoBayes 搜索 ON（v2）
data/distance_eval.json   三方距离阈值结果
scripts/            build_subset / run_batch / run_zeroshot / eval_distance
```
每张图完整 `St` 轨迹（先验 → 逐线索 (c,α)/W/后验 → 事件 → 层级）均以 JSON 落盘，可逐步复盘。
