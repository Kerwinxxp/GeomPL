# GeoBayes Paper → Code 映射（Phase 1 交付物）

> 依据：REPRODUCTION_PLAN.md v2。原则：歧义优先贴近原文；目标 = 单图 → 先验 + 全线索后验；先跑通后优化。
> 标签体系：`[paper]` = 论文明确规定；`[inferred]` = 标准实践推断；`[assumption]` = 我方设计假设。

---

## 0. 决策快照（2026-07-04 与用户确认）

| 决策 | 结论 |
|---|---|
| v2 层级转移 | 做。搜索 API：Tavily（免费档 1000 credits/月够 50 图试点）+ SerpApi Google Lens（免费档 250 次/月，不够时 $25 档） |
| MLLM 后端 | API（无本地 GPU）。锚点模型 `qwen2.5-vl-7b-instruct`，DashScope（阿里云百炼）或 SiliconFlow 的 OpenAI 兼容端点 |
| 数据 | Im2GPS3k 抽样 50 图起步 |
| 实施顺序 | v1（country 单层，无搜索）先跑通 → 对拍锚点 → v2 层级 |

---

## 1. 方程 → 代码映射

| 式号 | 内容 | 模块.函数 | 关键实现点 | 标签 |
|---|---|---|---|---|
| Eq.3/4 | MAP 任务定义 | `core/controller.py: run()` 返回 argmax | 最终输出 = 当前层 MAP + 完整分布 | paper |
| Eq.5 | 先验 | `core/prior.py: compute_prior(raw_scores)` | `P0 = softmax(min(si, 0.6)/1.5)`；**raw si 原样落盘**；断言任意两候选比值 ≤ e^0.4≈1.492 | paper |
| Eq.6 | 似然替代 | `core/likelihood.py: support_score(c, alpha)` | `W = exp(alpha * ln2 * (c-3))`；c 钳制到 {1..5} 整数，α 钳制 [0,1]；W∈[1/4,4] | paper |
| Eq.7 | 序贯更新 | `core/update.py: bayes_step(post, W)` | `post ← normalize(post ⊙ W)`，**每步归一化**（Fig.3 全链定死）；**不剪枝** | paper |
| Eq.8 | 状态 St | `core/state.py: State{H,V,M,C}` | 全 JSON 可序列化；每步快照进轨迹 | paper |
| Eq.9 | verification task | `core/state.py: Task{desc,reason,bbox,status}` | bbox 取 `[x1,y1,x2,y2]`（以 Eq.9/Fig.3 实例为准，正文 [x,y,w,h] 系笔误） | inferred |
| Eq.10 | ΔPt | `core/controller.py: delta_p()` | L∞：`max_l |Pt(l)−Pt−1(l)|`；v1 只记 `enhance_flag` 事件不执行 | assumption |
| Eq.11 | 停止 | `core/controller.py: should_stop()` | 见 §2 状态机（论文公式与正文矛盾，采用合并读法，τ=0.7） | assumption |

数值断言（每次更新后）：`sum≈1 (atol 1e-9)`、无 NaN、键集与上一步一致（除 replace/transition 事件）。

---

## 2. 控制流状态机

### 2.1 v1 主循环（country 单层）

```
S0 ← Hypothesize(image)                      # H0(k=5候选+si), V0(≤6任务), M0=∅, C0={level, history}
P ← compute_prior(S0.raw_scores)             # 分布一，落盘
for task in V0（生成顺序）:                    # §1.5#7：全部执行，不早停
    ev  ← Verify(crop(image, task.bbox))
    cw  ← Judge(ev, H.labels)                # {li: (ci, αi)}，看不到概率
    W   ← {li: support_score(ci, αi)}
    P   ← bayes_step(P, W)                   # 轨迹落盘：ev, (c,α), W, P, log2WoE
    记录: ΔP<0.05 → enhance_flag；maxP≥0.7 → threshold_crossed（v1 不动作）
    key evidence 判定：max_l |log2 W(ev|l)| ≥ 1.0 → 存入 M   # [assumption] 见 §5#18
if maxP < 0.7 and replace_count < 2:          # Replace（贴近原文 + 终止上限）
    H,V ← Replace(context)；P ← compute_prior(新 raw si)；继续循环
输出: final_posterior = P（分布二）, MAP, events, trajectory
```

- **v1 不因 maxP≥0.7 早停**：转移已禁用，且研究目标要求"看完所有线索"。事件照记，分析时可复演早停版本。`[assumption]`（目标驱动，偏离 w/o Hierarchy 消融的未知实现）
- Replace 后支撑集改变 → `events` 记 `support_changed`，分析先验/后验可比性时过滤。

### 2.2 v2 追加（层级转移）

| 触发 | 动作 |
|---|---|
| country/city 层 `maxP ≥ 0.7` | 转移：取 M 中 key evidence → Lens/Tavily 查询 "`{Oi}` in which cities of `{top_l}`?" → 解析出下层候选+分数 → Eq.5 生成下层先验（**不乘上层后验**，条件化于已提交假设，贴近原文）→ 新 Vt（优先未用视觉对象 + 深挖 key evidence） |
| 下层所有假设本轮 `c ≤ 2` | Backtrack：回 country 层，用新证据对存储的 country 假设重打分 |
| `ΔPt < 0.05` | Enhance：ImageSearch(crop) 优先，失败则 TextSearch(文字描述)，query 模板随层级适配（"in which country?" / "in which US city?"） |
| street 层 `maxP ≥ 0.7` 或 Vt 耗尽 | 停止，MAP 输出 |
| 非 street 层 Vt 耗尽且无法转移 | maxP≥0.7 → 停止输出；否则 Replace（≤2 次）→ 停止 |

---

## 3. MLLM 调用契约（prompt 文本全部 `[assumption]`）

通用：OpenAI 兼容 `chat.completions`，`temperature=0`，要求仅输出 JSON；解析失败重试 ≤2（重试时附报错提示）；仍失败按调用类型走兜底并记 `events`。

### 3.1 Hypothesize（1 次/图，输入=全图）

输出 schema：
```json
{"scene_summary": "str",
 "candidates": [{"location": "France", "confidence": 0.55, "reason": "str"}],   // 恰好 k=5，country 层
 "verification_tasks": [{"desc": "Examine the sign", "reason": "text may reveal language/region", "bbox": [x1,y1,x2,y2]}]}  // 3–6 个
```
v0 草稿 prompt 要点：
- "List exactly 5 candidate countries… confidence in [0,1], judged independently (do NOT normalize). Anchors: 0.9 = near-certain, 0.6 = clearly plausible, 0.3 = weak guess, 0.1 = long shot."（锚点跨越 0.6 截断线，降低全截平风险）
- 地标例外（贴近原文）：若有决定性城市级地标，允许直接给 city 候选并返回 `"level": "city"`；`initial_level` 落盘。
- 校验：canonicalize 后去重补齐到 5；si 钳制 [0,1]；bbox 钳制图界、短边 ≥16px、退化框回退全图并记事件。

### 3.2 Verify（每任务 1 次，输入=crop）

- 输入：按 bbox + 10% padding 裁剪 `[inferred]`；默认**仅 crop**（论文字面："extracts the corresponding image region"）；`verify_with_context: false` 为默认，轨迹审查后再定。
- 输出：`{"observation": "str", "geo_clues": ["str"]}`——只描述与推断线索，**不让它直接猜地点分布**（打分是 Judge 的职责，职责分离贴近论文两阶段结构）。

### 3.3 Judge（每证据 1 次，输入=纯文本）

输入：evidence 文本 + 假设标签列表（**无概率**——防先验二次注入，见 PLAN §1.4 禁令）。
输出 schema：
```json
{"ratings": {"United Kingdom": {"c": 5, "alpha": 0.45}, "United States": {"c": 1, "alpha": 0.44}, "Sweden": {"c": 1, "alpha": 0.42}}}
```
v0 草稿 prompt 要点：
- c 五档语义照论文：1=强矛盾, 2=较矛盾, 3=中性/无信息, 5=强支持；"if the evidence is unreadable or uninformative for a hypothesis, output c=3"（Fig.4 "MLLM neutral" 路径）。
- α 锚点：0.9=证据明确具体、0.5=有一定把握、0.2=模糊。
- "Rate each hypothesis independently; deliberately look for evidence AGAINST the currently plausible-looking ones."（抗确认偏误措辞）
- 校验：键集必须与 H.labels 恰好一致，缺失/多余重试；持续失败对缺失假设兜底 (c=3, α=0) → W=1，记事件。
- (c, α) 原始值全量落盘 → 监测 α 塌缩（PLAN §1.5#11）。

### 3.4 v2：Transition-Hypotheses / Enhance（搜索 → 结构化）

- Lens/Tavily 返回的网页摘要 → 一次 MLLM 调用抽取 `{candidates: [{location, confidence}]}` → Eq.5 成下层先验。搜索原始响应落盘缓存（SerpApi 缓存命中免费）。
- 无 API 降级路径（config: `hypothesis_source: mllm`）：MLLM 基于 M 的 key evidence 生成下层候选，**报告中标注为偏离论文**。

---

## 4. API 后端与图像/坐标细节

- 端点：DashScope 兼容模式 `https://dashscope.aliyuncs.com/compatible-mode/v1`（或 SiliconFlow）。模型 `qwen2.5-vl-7b-instruct`。`[inferred]`
- **坐标一致性（关键坑，PLAN §1.5#6）**：本地先复刻 `qwen_vl_utils.smart_resize`（对齐 28 的倍数，`max_pixels = 1280*28*28 ≈ 1.0MP`）把图 resize 后再上传 → 模型看到的像素空间 = 我们持有的像素空间，返回的绝对像素 bbox 可直接用于裁剪，无需反推服务端 resize。全图 vision tokens 同时被压到 ≤1280，控成本。
- 缓存：key = `sha256(image_bytes) + sha256(prompt + model + params)`，落盘 JSON。注意缓存冻结的是首次输出（temperature=0 下无影响）。
- 限速/重试：指数退避 ≤3；Nominatim 地理编码 1 req/s + 本地缓存。

---

## 5. 歧义决策表

PLAN §1.5 的 17 条为准据清单，此处只列代码级落点与新增第 18 条：

| # | 决策 → config 键 / 代码落点 |
|---|---|
| 1 | Eq.11 合并读法 → `should_stop()`；`tau_stop: 0.7`（= τ_transition，assumption） |
| 3 | `k: 5` |
| 4/5 | si∈[0,1] 独立打分 + prompt 锚点；raw si 落盘 `prior.raw_scores` |
| 6 | bbox [x1,y1,x2,y2] + smart_resize 对齐（§4）；`bbox_padding: 0.10` |
| 7 | 任务生成顺序全部执行；`max_tasks_per_level: 6` |
| 8/9 | judge 单次多假设；prompt 不含概率（§3.3） |
| 10 | 证据独立打分（贴近原文）；重叠 bbox（IoU>0.5）只记 warning 不干预 |
| 11 | α 锚点 + 钳制 + 塌缩监测 |
| 12 | 地标跳层保留；`initial_level` 落盘 |
| 13 | Replace 保留，`max_replace: 2`；`events: support_changed` |
| 14 | ΔP = L∞ |
| 15 | `temperature: 0` |
| 16 | `eval/geocode.py`: pycountry + 别名表（Scotland/England/Wales/N.Ireland → GB 记 recall；UK=United Kingdom 等）；GT 经纬度反编码用 reverse_geocoder（离线）`[inferred]` |
| 17 | v2 `hypothesis_source: websearch | mllm` |
| **18（新增）** | **key evidence 判定规则**：论文只说"高信息增益"没给标准。取 `max_l |log2 W| ≥ 1.0`（即某假设的证据权重至少 1 bit）→ 存入 M。`key_evidence_bits: 1.0` `[assumption]` |
| **19（新增，来自首次真实运行的教训）** | **Replace 证伪门**：论文触发条件（maxP<0.7 且计划耗尽）预设 Enhance/搜索在场；v1 无搜索时会在"领先者有支持证据"的状态误触发（实测：UK 0.529 领先、c=5 支持证据在手，Replace 强制排除 UK 后 MAP 变成南非）。且论文对 Replace 的语义是"所有已验证候选**无效**"——有 c≥4 证据时该前提不成立。门：当前假设集获得过任何支持性证据 (c≥4 且 α>0) → 不触发 Replace，改记 `replace_gated` 事件并输出当前分布；Replace 后新集重新计账。`replace_requires_refutation: true` `[assumption]`，与论文 Backtrack 的 "全部 ct≤2" 证伪信号同源 |

---

## 6. 数据管线（50 图子集）

1. 下载 Im2GPS3k 图片与 GT 坐标（Hays & Efros 发布版），校验完整性。
2. 固定 seed 抽 50 张；`data/subset50.jsonl` 记 `{image_id, lat, lon, gt_country(反编码)}`。
3. 锚点对拍：50 图 Hypothesize 的 Top-1/3/5 country recall vs Table 3 (70.4/77.6/81.5)，±5% 为通过（样本小，报告须附二项置信区间）。
4. 隐私产出（`analysis/belief.py`）：H(P0)、H(P_final)、KL(P_final‖P0)、逐线索 log₂WoE、校准分桶、支撑覆盖率。

---

## 7. v1 完成定义（DoD）

- [ ] §2.3 全部单元测试过（含 Fig.3 三层全链黄金测试，3 位小数）
- [ ] mock 控制流触发矩阵过（Replace 上限 / 不早停 / flag 记录）
- [ ] `python run.py <img>` 产出 PLAN §0 定义的完整 JSON
- [ ] 单图轨迹人工审查合理（候选、bbox、(c,α) 分布、后验演化可解释）
- [ ] 50 图先验 Top-K recall 在 Table 3 ±5% 带内；(c,α) 无严重塌缩
- [ ] 全部 `[assumption]` 汇总进 REPRODUCTION_REPORT.md 草稿
