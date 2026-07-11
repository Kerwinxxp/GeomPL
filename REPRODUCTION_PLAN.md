# GeoBayes 复现计划 v2（Reimplementation Plan）

> 适配对象：Shi et al., *GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating* (AAAI-26, pp. 8997–9005)。
> GeoBayes 是 **training-free 的 MLLM agent 系统**：冻结权重、无 loss、无 backward、无 checkpoint。验证方式为数值断言 + 控制流测试，而非训练循环验证。

## v2 相对 v1 的修订摘要

1. **【目标重定义】** 核心交付物改为：`单图输入 → 初始先验 P0 → 逐线索更新轨迹 → 最终后验` 的完整概率分布输出。基准表复现降级为可选。
2. **【错误修正】** 删除"Table 5 `w/o Enhance` 行是无搜索对照目标"的前提——经核实，**Enhance 与层级转移中的 WebSearch 是两个独立机制**，`w/o Enhance` 消融仍在转移时用 WebSearch 生成下层假设集；且该行 750km 数字为 **72.8**（非 73.7，73.7 是完整 GeoBayes）。
3. **【黄金测试升级】** 由单步对拍升级为 **Fig.3 三层五步全链对拍**（见 §2.3），并由此定死两个论文未明说的实现细节：每步更新后归一化、层内不剪枝。
4. **【新增决策原则】** 歧义处一律优先贴近原文（见下）。
5. **【歧义清单大幅扩充】** §1.5 从 7 条扩到 17 条，每条给出论文证据、默认选择与标签。
6. **【分阶段范围】** v1 = country 单层完整循环（无搜索依赖，两个分布同支撑）；v2 = 层级转移（贴近原文需 WebSearch API，无 API 则降级并标注偏差）。

---

## 决策总原则（新增，优先级从高到低）

1. **贴近原文优先**：论文有明确规定的照抄；论文歧义的，选择与原文其余部分（公式、图、示例数字）最自洽的读法，并在 §1.5 记录依据。
2. **目标导向**：一切设计服务于"拿到一张图的初始先验与看完所有线索后的最终后验"。与此无关的模块（基准表复现、外部搜索）可延后，但**不因简化而改变概率计算本身**。
3. **先跑通，后优化**：v1 端到端打通优先；prompt 调优、层级扩展、评测规模均放在跑通之后。

**Role**：保守实现，所有假设显式标注三级标签。绝不把论文没说的细节当成论文说的。绝不为凑数字做任意调参。

---

## 0. 任务定义（Task Definition）

- **Paper**: `GeoBayes_Probabilistic_Image_Geo-Localization_Inference.pdf`（AAAI-26, pp. 8997–9005）
- **Project directory**: `C:\Users\phdwf\OneDrive\Desktop\GeoBayes`
- **核心交付物【v2 修订】**：命令行单图入口
  ```
  python run.py <image.jpg> --config config.yaml
  ```
  输出一个 JSON，至少包含：
  ```jsonc
  {
    "prior": {                       // 分布一：初始先验
      "level": "country",
      "hypotheses": {"UK": 0.387, "US": 0.317, "SE": 0.296},
      "raw_scores": {"UK": 0.62, "US": 0.31, "SE": 0.22}   // 未截断/未软化的原始 si，必存
    },
    "trajectory": [                  // 每条证据一步
      {"task": "verify bus", "evidence": "...", "judgments": {"UK": [5, 0.45], "US": [1, 0.44], "SE": [1, 0.42]},
       "W": {"UK": 1.87, "US": 0.54, "SE": 0.56}, "posterior": {"UK": 0.682, "US": 0.161, "SE": 0.156},
       "log2_woe": {"UK": 0.90, "US": -0.89, "SE": -0.84}}
    ],
    "final_posterior": {"level": "country", "hypotheses": {"US": 0.813, "UK": 0.167, "SE": 0.019}},
    "events": []                     // replace/backtrack/transition/enhance 触发记录
  }
  ```
- **范围【v2 修订】**：
  - **v1（先跑通）**：country 单层完整 Hypothesize–Verify–Judge–Update 循环 + Replace + Eq.11 停止。无任何外部搜索。这是论文机制在首次层级转移之前的忠实子集，且保证先验与最终后验**同支撑、直接可比**。
  - **v2（层级扩展，待确认后做）**：层级转移到 city/street。贴近原文做法需 WebSearch（论文用 Google Lens API + Tavily API）生成下层假设集；若无 API 预算，降级为 MLLM 基于 key evidence 生成下层假设，**显式标注为与论文的偏差**。Enhance 模块同属 v2（依赖搜索）。
- **验证锚点【v2 修订，替换原 Table 2 / w/o Enhance 目标】**：
  1. **Fig.3 全链黄金测试**（§2.3）——纯数值层，确定性断言，不依赖模型。
  2. **Table 3 先验质量锚点**：Im2GPS3k 上 country-level Top-1/3/5 recall = **70.4 / 77.6 / 81.5**（Qwen2.5-VL）。只需 Hypothesize 一次调用即可对拍，不需要搜索、不需要 update 循环——是先验实现正确性的天然检验。
  3. （可选，方向性）Table 2：CoT 70.4 → Ours 73.7 @750km，验证"概率推理 > 线性 CoT"的趋势。
- **成功标准**：v1 端到端跑通并产出上述 JSON；黄金测试全过；小样本（≥50 图）先验 Top-K recall 与 Table 3 趋势一致（绝对值 ±5% 内可接受，prompt 未公开）。

---

## Phase 1 — 设计先于编码（Design before coding）

**产出唯一文件 `docs/paper_to_code_map.md`，完成前不写任何实现代码。**

### 1.1 需要从论文抽取的内容（含已核实的实现细节）

四个核心方程（出处页/式号）：
- **Eq.5 先验**：`P0(li) = exp(min(si, τp)/T) / Σj exp(min(sj, τp)/T)`，`T=1.5`，`τp=0.6`。
  - 【已核实】截断+温度使**任意两候选先验之比 ≤ e^0.4 ≈ 1.49**——先验被刻意压平，属论文设计（"smooth overconfident predictions"）。Fig.3 先验 {0.387, 0.317, 0.296}（比值 1.31）与此一致。
  - 【实现要求】同时落盘原始 si（`raw_scores`），P0 只是其校准版。
- **Eq.6 似然替代**：`W(et|li) = exp[αt · β · (ct−3)]`，`β=ln2`，`ct∈{1..5}`，`αt∈[0,1]`。
  - 【已核实】`(ct, αt)` 是 **per-(evidence, hypothesis)** 的——论文记号只有下标 t，但 worked example 中每个假设的 W 不同，证明逐假设打分。
  - 【已核实】`W ∈ [1/4, 4]`；配合每步归一化，**无需 log-space 防下溢**。
  - 【注意】论文仅声称 W 与真实似然**保序**（"preserves ranking consistency"），未声称校准。最终分布是伪后验/信念分数。
- **Eq.7 贝叶斯更新**：`P(l|E1:t) ∝ P0(l) · Π W(et|l)`。
  - 【已核实，由 Fig.3 全链定死】**每步更新后立即归一化**（Fig.3 所有中间分布均归一化）；**层内不剪枝**（SE 以 0.019 一路带到层末）。
- **Eq.11 停止条件**：⚠️ **公式与正文互相矛盾**（详见 §1.5 条目 1），且停止阈值 τ 未在超参列表中定义。采用的解读见 §1.3。

其他：
- 状态四元组 **`St = {Ht, Vt, Mt, Ct}`**（Eq.8）及各分量语义。
- verification task 结构（Eq.9：`desc / reason / bbox / status`）。
- 数据流：global 分析 → 先验 →（verify → judge → update）循环 →（v2：层级转移）→ MAP 输出（Eq.4）。
- 超参：`τp=0.6`、`T=1.5`、`β=ln2`、`τ_transition=0.7`、`τ_enhance=0.05`。

### 1.2 三级标签（对每一处实现选择必须标注其一）

- `[directly specified by paper]` —— 例：全部超参数值；每步归一化与不剪枝（由 Fig.3 数字链反推，视为论文事实）。
- `[inferred from standard practice]` —— 例：JSON 约束输出；bbox 裁剪加 padding；地名→GPS 用 Nominatim。
- `[my proposed assumption]` —— 例：全部 prompt 文本；k=5；judge 一次性对全部假设打分；Replace 重试上限。

### 1.3 状态转移表【v2 修订：并入 Eq.11 矛盾的解读】

每行 = `(当前 level, 触发条件) → (动作)`。**τ 取 0.7（假设 τ = τ_transition，依据 Fig.3 单一 "Level/Stop threshold" 标注）`[my proposed assumption]`**：

| 层级 | 触发条件 | 动作 | 依据 |
|---|---|---|---|
| < street | `max Pt ≥ 0.7` 且可转移（v2 且搜索可用） | 层级转移：基于 Mt 的 key evidence 生成下层假设集 + 新 Vt | Hierarchical Transition 段 |
| < street | `Exh(Vt)` 且 `max Pt ≥ 0.7` 且不可转移 | **停止，输出当前层 MAP** | Eq.11 印刷版第一分支（与 Replace 段自洽） |
| < street | `Exh(Vt)` 且 `max Pt < 0.7` | **Replace**：告知 MLLM 全部候选未证实，基于推理上下文重生成假设集+Vt；**重试上限 max_replace=2** `[my proposed assumption]`（论文无界，照抄不终止），超限后停止输出当前 MAP | Replace 段 + 终止性要求 |
| street | `max Pt ≥ 0.7` **或** `Exh(Vt)` | 停止，MAP 输出 | Eq.11 正文表述 + Fig.2 流程图佐证 |
| > country（v2） | 本轮所有假设 `ct ≤ 2` | **Backtrack**：回 country 层用新证据重评估 | Replace and Backtrack 段（v1 单层不适用） |
| 任意 | `ΔPt < 0.05` | 触发 **Enhance**（v1 仅记 flag 不执行；v2 接搜索） | Eq.10 |

> Eq.11 矛盾的处理依据：印刷公式（阈值条件挂在 street 以下、用 ∧）与正文（阈值条件挂在 street 层、用"或"）无法用单一提取错误解释，系论文自身表述冲突。上表取两者与 Replace 段、Fig.2/Fig.3 最自洽的合并读法，完整论证记录进 `paper_to_code_map.md`。**v1 单层下的实际行为退化为：跑完 Vt 全部任务（含至多 2 轮 Replace）后停止**——这正好等于"看完所有线索后输出后验"，与研究目标一致。

### 1.4 Prompt I/O 契约【v2 修订：细化 schema 与关键禁令】

对每一次 MLLM 调用写明输入/输出 schema 并打标签（prompt 文本全部为 `[my proposed assumption]`，论文未公开任何 prompt）：

1. **Hypothesize**：输入=全图。输出 JSON：
   `{candidates: [{location, confidence si ∈ [0,1]}] × k, verification_tasks: [{desc, reason, bbox}]}`
   - k=5 `[my proposed assumption]`（Fig.4 两例均 5 个；Table 3 报到 Top-5；Fig.3 为 3 个，说明 k 不定，取 5 为上界习惯）。
   - si 明确要求 [0,1] 且**各候选独立打分、不要求归一**（Eq.8 示例 {0.261, 0.188, 0.185} 和不为 1）。
   - prompt 中给 si 语言锚点（如 0.9=几乎确定，0.5=中等，0.2=微弱猜测），降低"全部 ≥0.6 被截平成均匀"的风险。
2. **Verify**：输入=按 bbox 裁剪的图像区域（+padding）。输出=证据自然语言描述。
   - 默认**只喂裁剪区域** `[directly specified by paper]`（"extracts the corresponding image region, and prompts the MLLM"）；`config.verify_with_context` 可选同时附全图（论文示例输出含场景上下文，纯 crop 可能给不出，跑通后视轨迹质量决定）。
3. **Judge**：输入=证据文本 + **假设标签列表**。输出=每假设 `(ct, αt)`，一次调用对全部假设打分 `[my proposed assumption]`。
   - ⚠️ **禁令：judge 的 prompt 不得包含当前后验概率**（只给地名标签）。若泄入 P(li|E1:t−1)，ct 会与当前信念相关，先验被二次乘入似然，恰好重新引入论文要消除的确认偏误。`[my proposed assumption]`，依据=论文的反确认偏误设计意图。
   - prompt 给 αt 语言锚点；越界值钳制；**中性路径**：证据无信息/看不清 → 指示输出 ct=3（W=1），对应 Fig.4 的 "MLLM neutral" 行为。
   - judge 是否可见已积累证据历史：默认**不可见**（每条证据独立打分，与 worked example 一致）`[my proposed assumption]`；Eq.7 的条件项 E1:t−1 记为未实现的论文歧义。

### 1.5 缺失/歧义清单【v2 修订：17 条，每条含默认决策】

| # | 歧义 | 论文证据 | 默认选择 | 标签 |
|---|---|---|---|---|
| 1 | Eq.11 公式 vs 正文矛盾；τ 未定义 | 公式把 `≥τ ∧ Exh` 挂在 <street；正文把"≥τ 或完成"挂在 street | §1.3 合并读法；τ=0.7 | assumption |
| 2 | 正文把 W={1.87,0.54,0.56} 归给 "verify bus"，Fig.3 归给 "Trolley Wires" | 图文自相矛盾 | 仅记录；数字链不受影响 | — |
| 3 | 候选数 k | Fig.3=3，Fig.4=5，Table 3 到 Top-5 | k=5 | assumption |
| 4 | si 范围与引出方式 | τp=0.6 暗示 [0,1]；prompt 未公开 | [0,1]，独立打分，语言锚点 | assumption |
| 5 | 多个 si≥0.6 全截平成均匀 | Eq.5 截断的直接后果 | 接受（属论文设计）；raw si 落盘供分析 | paper |
| 6 | bbox 格式矛盾 | 正文写 [x,y,w,h]，Eq.9/Fig.3 实例均为 [x1,y1,x2,y2] | 取 [x1,y1,x2,y2]（以实例为准）；Qwen2.5-VL 输出绝对像素坐标（smart-resize 后），裁剪须匹配；钳制到图界+最小尺寸+退化框回退全图 | inferred |
| 7 | Vt 任务数与选取顺序 | Fig.3 每层约 2–3 个；"selects a task" 未说顺序 | 生成顺序全部执行；数量由 Hypothesize 自主给出，上限 6/层 | assumption |
| 8 | judge 一次多假设 vs 逐个询问 | 未说明 | 一次多假设（省调用，利于对比打分） | assumption |
| 9 | judge 是否见后验概率 | 未说明（"guided by St" 易误导） | **禁止**（见 §1.4） | assumption |
| 10 | 证据相关性（Eq.7 条件项 E1:t−1） | worked example 各证据独立打分；London bus/US plate/trolley wires 同物三面仍独立连乘 | v1 照论文独立打分（贴近原文）；相关性膨胀作为已知局限写入报告，重叠 bbox 记 warning | paper |
| 11 | α 无 rubric，且 W 中仅 α·(c−3) 乘积可辨识 | 论文未给 | prompt 锚点+钳制；落盘 (ct,αt) 经验分布，监测 α 塌缩 | assumption |
| 12 | 地标跳层（初始层级可能非 country） | "Sydney Opera House → 直接 city 层" | 保留论文行为；`initial_level` 落盘。分析两分布时按初始层分桶 | paper |
| 13 | Replace 换掉假设集 → 先验/后验支撑不同 | Replace 段 | 保留（贴近原文），`events` 记录；分析时可过滤该类图 | paper |
| 14 | ΔPt 的范数（Eq.10 只写 \|Pt−Pt−1\|，未说对哪个 l） | 未说明 | max-abs（L∞）over 假设 | assumption |
| 15 | 解码参数 | 未说明 | temperature=0（贪心），保证可复现；注明缓存冻结的是首次输出 | inferred |
| 16 | 地名规范化 | 论文自己把 Scotland 当国家、UK/United Kingdom 混用 | pycountry+别名表；UK 构成国映射到 GB 记 recall；GT 经纬度反编码到国家用于 Table 3 对拍 | inferred |
| 17 | city 级假设在无搜索时如何生成（v2 降级方案）；YFCC 失效图 | 论文转移必用 WebSearch | v2 决策点：有 API 照论文，无 API 用 MLLM 生成并标注偏差；YFCC 推迟 | assumption |

### 1.6 最小可行实现声明

锚定 §0：**v1 = 纯 MLLM + 单层贝叶斯循环，无外部搜索**，即可交付两个同支撑分布 + 全轨迹，并可对拍 Fig.3 全链与 Table 3 先验锚点。

**Phase 1 结束后停下，等确认，再进入 Phase 2。**

---

## Phase 2 — 可测试骨架（Testable skeleton）

### 2.1 工程结构

```
geobayes/
├── run.py                   # 单图入口：image → JSON（§0 schema）
├── config.yaml              # 全部超参与开关（τp/T/β/τ_transition/τ_enhance/k/max_replace/
│                            #   enable_hierarchy/enable_enhance/verify_with_context/force_country_start...）
├── requirements.txt         # 固定版本
├── mllm/client.py           # 后端封装（vLLM 本地或 OpenAI 兼容 API）：JSON schema 约束 + 解析失败重试 + 磁盘缓存
├── mllm/mock_client.py      # 返回固定 JSON 的 mock，脱离模型测控制流
├── core/
│   ├── state.py             # St={Ht,Vt,Mt,Ct}，全部可 JSON 序列化
│   ├── prior.py             # Eq.5（含 raw si 保留）
│   ├── likelihood.py        # Eq.6（含钳制与中性路径）
│   ├── update.py            # Eq.7（每步归一化）
│   └── controller.py        # 主循环 + §1.3 转移表（Replace 上限 / Eq.11 停止 / v2 转移与 Backtrack）
├── eval/metrics.py          # haversine + 1/25/200/750/2500km 阈值准确率 + Top-K country recall
├── eval/geocode.py          # 地名→GPS 与 GPS→国家（Nominatim，带缓存+限速）；地名规范化
├── analysis/belief.py       # 隐私侧产出：熵、KL(P_final‖P0)、逐线索 log2 weight-of-evidence
└── tests/                   # 见 2.3
```

（注：MLLM 后端可配置。复现论文锚点须 Qwen2.5-VL-7B；本机无 GPU 时可用 DashScope/OpenRouter 的同型号 API，属部署细节不影响算法。）

### 2.2 通用工程要求

- 固定依赖版本；`set_seed()` 覆盖 python/numpy（torch 仅本地推理时）。
- 所有 MLLM 调用以 `(image_hash, prompt_hash)` 为 key 落盘缓存（省算力 + 可复现）。
- 概率分布每次更新后断言 `sum≈1`、无 NaN、键集不变（除 Replace/转移事件外）。
- 结构化日志：每图 dump 完整 `St` 演化轨迹 + `events` + (ct, αt) 原始值。

### 2.3 单元测试【v2 修订：黄金测试升级为 Fig.3 全链】

- **Eq.5 性质**：截断 τp 生效；温度 T 生效；归一化；**任意两候选比值 ≤ e^0.4**。
- **Eq.6 性质**：`ct=3 → W=1`；`α=0 → W=1`；同 α 下 c=5 与 c=1 的 W 互为倒数；关于 c 单调；输出域 [1/4, 4]。
- **Eq.7 黄金测试（Fig.3 全链对拍，断言到小数点后 3 位）**：

  | 层 | 链 |
  |---|---|
  | Country | {0.387, 0.317, 0.296} ×W{1.87, 0.54, 0.56}→ {0.682, 0.161, 0.156} ×W{0.55, 3.48, 0.55}→ {0.367, 0.549, 0.084} ×W{0.57, 1.86, 0.29}→ {0.167, 0.813, 0.019} |
  | City | {0.574, 0.425} ×W{1.74, 0.56}→ {0.809, 0.191} |
  | Street | {0.517, 0.483} ×W{1.62, 0.66}→ {0.724, 0.276} |

  全链已手工核算通过。该测试同时锁定"每步归一化"与"不剪枝"两个实现细节。
- **控制流断言**（配 mock）：Replace 恰在 `Exh ∧ maxP<0.7` 触发且不超过上限；street 层 `maxP≥0.7` 提前停止；`ΔP<0.05` 只记 flag（v1）。

### 2.4 Toy example

Fig.3 全链即为多步 toy（country 层 3 步 + 两次层级衔接的数字都有），无需另造；v2 做层级转移时补一条含转移的 mock 轨迹。

---

## Phase 3 — 渐进验证【v2 修订】

1. **纯数值层**：§2.3 全部单元测试（含全链黄金测试）。完全脱离 MLLM，确定性可验证。
2. **Mock-MLLM 层**：`mock_client` 喂固定 JSON 跑通 `controller`——专门验证控制流触发矩阵（Replace 上限、Eq.11 两分支、flag 记录）。模型不确定性与控制逻辑正确性解耦。
3. **真实 MLLM，单图**：跑 1 张图，人工审 `St` 全轨迹：候选合理性、si 分布、bbox 有效性、(ct,αt) 是否塌缩、后验演化是否可解释。
4. **真实 MLLM，30–50 图**：prompt 迭代主战场。同时对拍 **Table 3 先验锚点**（country Top-1/3/5 recall vs 70.4/77.6/81.5）——先验对不上先修 Hypothesize prompt 与地名规范化，再看 update 环节。
5. **（可选）扩展**：Qwen2.5-VL zero-shot baseline 校准（Table 1 行 83.8/70.4/51.1/31.0/5.1）→ Table 2 方向性对比 → v2 层级。

### 隐私研究产出物（新增，v1 即可产出）

- 逐线索 **log₂W 差 = weight of evidence**：每条线索把信念推动了多少 bit 的量化分解。
- `H(P0) − H(P_final)`、`KL(P_final ‖ P0)`（同支撑，v1 天然可算）。
- **校准检查**：按 final top-1 概率分桶 vs 实际命中率（可靠性图）。论文只保证保序，此检查决定你在隐私论文里能否把它当校准后验使用。
- 支撑覆盖率：真值国家落在先验支撑内的比例（对照 Table 3 的 81.5%），以及不在支撑内时系统的行为记录。

---

## Phase 4 — 复现报告（`REPRODUCTION_REPORT.md`）

必含：精确复现命令；环境与版本；数据来源与完整性；**全部三级标签假设汇总**（重点：全部 prompt、§1.5 的 17 项决策）；与论文的实现偏差（v1 无搜索/无层级；judge 打分方式；Replace 上限）；实测 vs 论文（Table 3 锚点、可选 Table 2）；已知局限（相关证据独立连乘的膨胀效应、伪后验非校准、支撑集外无质量）。

---

## 约束（Constraints）

1. 歧义决策一律先在 §1.5 记录，再选与原文最自洽的默认值。
2. 绝不虚构细节冒充论文内容——prompt 全部标 `[my proposed assumption]`。
3. 不为匹配论文数字调参；复现价值在诚实。
4. 概率计算路径（Eq.5/6/7、归一化时机、不剪枝）不允许任何"简化"偏离。
5. 本系统无 loss/backward/checkpoint；正确性验证重心在 Eq.5/6/7 数值断言与控制流（Phase 3 步骤 1–2）。

---

## 待确认的三个问题（审阅时请回答）

1. **v2 层级转移**：是否需要做到 city/street 层？若需要，是否有 Google Lens / Tavily API 预算？（无 API 则用 MLLM 降级生成下层假设，标注偏差。）
2. **MLLM 后端**：本地有无可用 GPU（Qwen2.5-VL-7B 需 ~24GB bf16）？还是走 API（DashScope 等提供同型号）？
3. **数据**：v1 验证用 Im2GPS3k 抽样 50 图起步是否可以？（GT 为经纬度，需反编码成国家名对拍 Table 3。）
