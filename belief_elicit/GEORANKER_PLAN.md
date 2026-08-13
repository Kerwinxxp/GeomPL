# GeoRanker 全量替换计划(待审阅,批准前不执行)

目标:**GeoRanker 全面替换 GeoCLIP,成为 mPL 的唯一信念计**——5 图全子集消融、200 张
sweep、全部图表都换到 GeoRanker 口径重出;GeoCLIP 结果仅归档为 baseline 对照(论文可比用)。
本文档先交代**我对代码库的完整理解**(已克隆通读,结合论文),再给出实现步骤、成本与风险。
中间的验证步骤不是"决定要不要换",而是换测量仪器前的必要体检:仪器没坏就直接铺满。

---

## 一、代码库分析(已通读,不是凭 README)

仓库:`Applied-Machine-Learning-Lab/GeoRanker`(已浅克隆到 scratchpad,共 5 个核心 py,~815 行)。

### 1.1 模型架构(`utils/geo_ranker.py`,102 行)

```
RewardModel = Qwen2VLForConditionalGeneration 子类
            + value_head: nn.Linear(hidden_size, 1, bias=False)
forward: VLM 前向 → 取最后一层 hidden_states 的【最后一个 token】→ value_head → 标量 reward
```
- **reward 语义:越高 = 距离越近**(README 明示;训练目标是排序损失,论文 §3)。
- 发布的 checkpoint 是 **LoRA adapter(13.8MB,已在仓库里)**:r=16,目标 q/k/v_proj,
  `modules_to_save=["value_head"]`(value_head 权重随 adapter 一起发布)。
- 底座 `Qwen/Qwen2-VL-7B-Instruct` 需从 HF 下载(~16.5GB)。

### 1.2 训练时的输入格式(`finetune_geo_ranker.py` L95–104,决定 checkpoint 见过什么)

```python
content = [
  {"image": query图},
  {"text": f"How far is this place from latitude: {lat}, longitude: {lon}, {ref_texts}?"},
  {"image": 候选参考图},                       # ← 训练时有
  {"text": f"Negative examples: {负例文本…}"},   # 负例 = "latitude:…, longitude:…, 文字描述" ×5
]
```
`ref_texts` 来自 GeoRanking 数据集(城市/国家等文字描述)。

### 1.3 关键发现:官方推理本身就有"无参考图"格式(`evaluate.py` L100–109)

官方评测对 **LVLM 生成候选 C_g** 的打分 prompt 是:
```python
content = [
  {"image": query图},
  {"text": f"How far is this place from latitude: {lat}, longitude: {lon}?"},   # 无 ref_texts
  {"text": f"Negative examples: {…}"},                                          # 无候选参考图!
]
```
**论文 Table 1 的 SOTA 数字就是在这种混合输入下取得的**(IM2GPS3K 用 12 个带图检索候选 + 3 个无图生成候选)。
结合论文 Table 2 消融:`w/o c_img`(重训无图变体)street 18.79→15.58,country 几乎不变(76.31→75.40),
且仍显著强于 GeoCLIP。**结论:无参考图输入不是灾难性 OOD,官方管线自己在用。**
(`quick_start.py` 甚至把 ref_texts 和负例都省了——作者把这些组件当可选。)

### 1.4 Windows/硬件隐患(已定位,均可绕过)

| 问题 | 影响 | 对策 |
|---|---|---|
| `utils/geo_ranker.py` 顶层 `import flash_attn`、`import deepspeed` | Windows 装不上,直接 import 即炸 | **不 import 他们的文件**;自写 loader(RewardModel 逻辑仅 ~40 行,照抄) |
| 仓库环境 torch==2.6.0 | **不支持 Blackwell(RTX 5080, sm_120)**,需 torch≥2.7+cu128 | 新建 venv 装新 torch(SAM3 venv 已证明新 torch 在 5080 可用) |
| transformers==4.52.0.dev0(未发布 dev 版) | 装不到 | README 自己说"换正式新版";装 4.5x 稳定版 |
| bf16 7B ≈ 15GB vs 显存 17.1GB | 紧张 | 小 batch;不够则 4-bit 量化回退 |
| adapter_config 的 auto_mapping 指向 `utils.geo_ranker` | AutoPeft 会尝试 import 该模块 | 用显式 `PeftModel.from_pretrained(model, path)`,不走 auto |

---

## 二、我们的接入设计

### 2.1 新文件(全部在 `belief_elicit/`,可整体删除,不碰线上代码)

**`georanker_belief.py`** —— 与 `geoclip_belief.py` 接口完全一致:
1. 自写 loader:内联 RewardModel 定义(Qwen2-VL + value_head,取末 token)→
   `from_pretrained(底座, attn="sdpa", bf16)` → `PeftModel.from_pretrained(model, checkpoints/)`;
2. `score_labels(image, labels, coord_cache) → {label: prob}`:
   对每个候选构 prompt → 批量前向取 reward → `softmax(rewards / τ)`;
3. τ 默认 1.0,作为显式配置记录(图内比较中 τ 只是单位,不影响排序)。

**输入格式做成两个变体,都测**(用数据说话,不猜哪个好):
- **变体 A(= 官方 C_g 格式)**:`query图 + "How far … latitude: X, longitude: Y?"`,无文字无负例;
- **变体 B(最贴近训练格式、减去参考图)**:`query图 + "How far … latitude: X, longitude: Y, {label文字}?" + 负例`
  - label 文字 = gallery_v2 的 `"City, Country"`(它 `text` 字段的第一次实际使用);
  - 负例 = 该图 GeoCLIP 打分最低的 5 个候选,按训练格式拼 `"latitude:…, longitude:…, 文字"`。

**一致性约束**:original / masked 图用**同一** processor 设置(如需限 `max_pixels` 控显存,两边同值)。

### 2.2 复用(不新写)

- 掩码/遮蔽:`clue_leak.masking` + `cue_extract.rle`(与 GeoCLIP 消融同一套);
- 几何/mPL:`run_geoclip_sweep200.py` 里的 `build_geometry`/`mpl`(同口径,结果直接可比);
- gallery:`data/gallery_v2.json`(138 候选,GPS+label 都有,**无需 MP16-Pro、无需参考图**)。

---

## 三、执行步骤(每步一个闸门)

| 步 | 内容 | 产出/判定 | 预估 |
|---|---|---|---|
| 0 | 磁盘已查:1.2TB 空闲 ✅;新建 `belief_elicit/.venv_gr`(python 3.11) | venv 就绪 | ~5 分钟 |
| 1 | 装依赖:torch(cu128 新版)+ transformers 4.5x + peft + qwen-vl-utils + accelerate;**不装** flash-attn/deepspeed | import 全通 | 10–20 分钟(下载 ~3GB) |
| 2 | 下载 Qwen2-VL-7B-Instruct 底座(~16.5GB → HF 缓存) | 权重就绪 | 视网速 10–40 分钟 |
| 3 | 写 `georanker_belief.py`;**冒烟测**:Okazaki 原图 + 2 个候选(冈崎/伦敦),reward 应冈崎>伦敦;同输入跑 2 次验证确定性;记录显存峰值 | 跑通 + 方向对 + 确定性 | 半小时(含调试) |
| 4 | **仪器体检(Okazaki 单图,变体 A、B 各跑)**:① 原图对 138 候选打分 → 真值 rank(对照 GeoCLIP);② 全 15 子集遮蔽 → p_true 是否降、mPL 是否随覆盖升、有无尾部爆炸。**据此选定 A/B 之一为标准格式** | 体检表 + 定格式 | 每变体 ~16 次打分 ×138 前向,估 15–40 分钟 |
| 5 | **替换第一波:5 图全子集消融**(GeoRanker 口径)→ 重出 5 张逐图 mPL 图(对应 `plot_geoclip_ablation` 版式) | 5 张新图 | ~80 次打分 ×138 前向,估 0.5–1.5 小时 |
| 6 | **替换第二波:200 张全量 sweep**(逐单线索 + 全遮)→ 重出精度体检 + 逐类别泄露图 | 全量结果 + 汇总图 | ~855 次打分 ×138 ≈ **11.8 万次 7B 前向,估 5–13 小时(过夜跑)**;开跑前跟你确认一次时间窗口 |

## 四、体检标准(第 4 步,事先定死;这是仪器检查,不是"要不要换"的投票)

1. **准确性**:原图上真值(或其 25km 簇)进入 top-3?与 GeoCLIP 的 rank 对比;
2. **遮蔽响应**:全遮后 p_true 下降(方向对)且 |Δ| 明显大于 0;
3. **mPL 形态**:随覆盖大体单调、无 1e-12 级尾部伪信号(不需 clamp 就稳定)。

三条通过 → **不再询问,直接进第 5、6 步铺满**(仅第 6 步开跑前和你约时间窗口)。
若有条目不过 → 停下来把体检表给你看,一起定(比如换 prompt 变体/调温度),**不自作主张放弃全量替换**。

## 五、成本汇总与风险

- **磁盘** ~20GB(充足);**API 零花费**,全本地;
- **时间**:环境+下载 ≤1 小时,验证 ≤1 小时;200 张全量另议(数小时,过闸门 5 才做);
- **最大技术风险**:新 transformers 版本上 Qwen2-VL + 自写 RewardModel + LoRA 加载的兼容细节(hidden_states 索引、processor 行为)。对策:冒烟测里用"冈崎 vs 伦敦"这种常识对照直接暴露问题;
- **最大科学风险**:无参考图时 reward 对**遮蔽**的敏感度未知(官方只证明了它能排序,没证明它对像素扰动敏感)——这正是第 4 步要测的,也是整个验证的核心问题。

## 六、明确不做的事 + GeoCLIP 的去向

- 不下载 MP16-Pro、不建检索索引、不重训/微调;
- 不动线索提取(`cue_extract/`)与 mPL 口径(几何/聚类/公式全部沿用,保证与旧结果同尺);
- **GeoCLIP 的角色**:替换后其结果与脚本归档保留(`geoclip_*`),作为论文的 baseline
  对照("换更强攻击者,逐线索泄露结论是否稳"本身就是一节好内容),不再更新;
- 第 6 步(200 张过夜跑)开跑前跟你确认时间窗口,其余步骤连续执行不再逐步请示。

---
**请审阅**:①第 4 步判定标准是否认可;②变体 A/B 的设计是否符合你对"文字描述"的预期;③闸门位置是否合适。批准后我从步骤 0 开始。
