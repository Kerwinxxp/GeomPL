# 地理线索提取管线(Cue Extraction Pipeline)— 方案 v2

> 基于用户 2026-07-08 提出的五模块方案修订。
> **当前阶段目标变更**:mPL 测量暂停。本阶段唯一交付物 = **每张图上"位置相关线索区域"的高质量标注(SAM 掩码级)+ 可视化**,由人工目检验收"线索区域到底合理否"。验收通过后,才回到 mPL 测量(届时 clue_leak 消融直接换用本管线的输出)。
>
> 独立包 `cue_extract/`,不动 `clue_leak/` 与 `geobayes/` 现有代码。

---

## 0. 背景:为什么要重做线索提取

5 图消融试点(clue_leak/results_combo)暴露的三个病灶,均源自"GPT-4o 一次调用同时做发现+画框":

| 病灶 | 实例 | 根因 |
|---|---|---|
| 退化框 | Venice"建筑"bbox=100% 全图 | 全局属性硬塞进矩形框 |
| 检测器失明 | Biwer 遮掉全部检出线索,信念纹丝不动(KL=0) | VLM 检测召回低,漏掉真正驱动信念的小字/背景 |
| 框粗糙重叠 | Okazaki 4 框互相覆盖,遮 A 连带遮 B | VLM 空间定位能力弱 |

解法 = 分工:VLM 说"有什么"(语义强),专业模型定"在哪"(定位强),OCR 兜文字底,SAM 出像素级掩码,verifier 把关。

---

## 1. 管线结构

```
Input Image
   │
   ├── ① VLM Cue Proposal(GPT-4o,1 次调用)
   │      └── 候选线索列表(8 类 checklist,不画框)
   │
   ├── ② Grounding Branch(Grounding DINO,本地 GPU)
   │      └── 每条 cue 短语 → bbox(可多实例)
   │
   ├── ③ OCR Branch(PaddleOCR,本地,独立于①②)
   │      └── 文字框 + 识别文本 → LLM 评 geo_relevance
   │
   ├── ④ SAM 2.1(本地 GPU)
   │      └── bbox prompt → 像素级 mask
   │
   └── ⑤ VLM Verifier(GPT-4o,1 次调用)
          └── 风险评级 + 质检(拒退化框、标 maskable)
   │
   └── ⑥ 可视化 QA(本阶段的验收出口)
          └── 原图 + mask 叠加图(按类别着色、编号、图例)
```

## 2. 各模块规格

### ① Cue Proposal — GPT-4o
- Prompt 用用户拟稿(8 类 cue categories,"Do not infer the final location / visible evidence only")。
- 修订 1:**按类别逐项输出、允许空**(checklist 结构化输出,召回 > 自由列表)。
- 修订 2:每条 cue 输出 `{cue, category, grounding_phrase, is_text}`——`grounding_phrase` 为喂给②的简短英语名词短语;`is_text=true` 的线索同时交给③交叉核对。
- 模型:先用 GPT-4o(已有 key、已接好);proposal 模块接口留 model 参数,换 Gemini/GPT-5.x 只改配置。

### ② Grounding — Grounding DINO(本地,transformers 实现)
- 用 HuggingFace `IDEA-Research/grounding-dino-base`(纯 PyTorch,Windows 无需编译 CUDA 算子,RTX 5080 无压力)。
- 每条 `grounding_phrase` 独立查询 → 得分过阈值的全部 bbox。
- **多实例规则:同一 cue 的多个实例合并为一条线索的实例列表**(将来遮蔽消融必须 union 遮,否则 leave-one-out 从另一实例漏)。

### ③ OCR — PaddleOCR 3.0(本地,不能省)
- 全图跑 det+rec → `{text, bbox, conf}`。
- 文本列表打包给 LLM(可并入⑤的调用)评 `geo_relevance: high/medium/low + reason`;low 丢弃。
- 与②的框按 IoU>0.5 去重(OCR 结果优先保留 text 字段)。

### ④ SAM 2.1 — bbox → mask(本地)
- `facebook/sam2.1-hiera-large`,box prompt,每框一 mask。
- 输出 RLE 存 JSON;可视化用轮廓+半透明填充。
- 注:将来消融时用 mask 还是 bbox 遮蔽再定(不规则剪影本身可能是形状线索);**本阶段 mask 的作用是让人看清线索区域是否合理**,是核心交付。

### ⑤ Verifier — GPT-4o
- 用户的风险规则表(街牌/门牌/车牌/店名/邮编=high;地标/公交/连锁=high-medium;建筑风格/道路/植被/气候=medium;天空/普通树/人=low)。
- 每条 cue 输出 `{risk_level, geo_specificity(generic/regional/city/street), searchability, explanation}`。
- **新增两个硬性职责**(比 risk 更影响有效性):
  1. **面积上限**:bbox > 40% 全图 → 标记 `degenerate: true`(治 Venice 病);
  2. **`maskable: true/false`**:第 4/5 类全局弥散属性(建筑风格、地形、气候、植被)本质不可框选 → `maskable:false`,保留在 JSON 里(信息完整)但不进将来的遮蔽消融。**这是原方案缺的关键概念,Venice 病的病根。**

### ⑥ 可视化 QA(本阶段验收出口)
- 每图一张标注图:原图 + SAM mask 半透明叠加,颜色=类别,编号+图例(cue 名/类别/risk/是否 maskable/OCR 文本)。
- 全部试点图拼 contact sheet,人工目检:区域准不准、漏没漏、有无退化框。
- 复用/扩展 `clue_leak/viz_one.py` 的对照式布局。

## 3. 输出 JSON(schema 按用户稿,含新增字段)

```json
{
  "image_id": "xxx",
  "geo_privacy_cues": [
    {
      "cue": "street sign",
      "category": "text/signage",
      "instances": [
        {"bbox": [122, 88, 302, 145], "mask_rle": "...", "score": 0.71, "source": "grounding"}
      ],
      "text": "Denton Square",          // OCR 分支时填
      "risk_level": "high",
      "geo_specificity": "street-level",
      "searchability": "high",
      "maskable": true,
      "degenerate": false,
      "reason": "Street names can directly narrow down the location."
    }
  ]
}
```

## 4. 实施顺序(每步有可看的产物)

| 步骤 | 内容 | 产物/验收 |
|---|---|---|
| S1 | 环境:torch(CUDA)+ transformers-GroundingDINO + sam2 + paddleocr;GPU 冒烟测试 | 三模型本地推理各通过 1 张测试图 |
| S2 | ① Proposal(TDD,mock LLM) | 5 张试点图的 cue 列表,人工过目 |
| S3 | ② Grounding + ③ OCR + 合并去重 | bbox 叠加图 v0,对比旧 GPT-4o 框 |
| S4 | ④ SAM mask + ⑥ 完整可视化 | **mask 级标注图 + contact sheet ← 本阶段核心交付** |
| S5 | ⑤ Verifier(risk/maskable/degenerate) | 最终 JSON + 带风险标签的标注图 |
| S6 | 用户目检验收 → 迭代 prompt/阈值 → 通过后回到 mPL | 验收会 |

试点集:先用消融那 5 张(有旧标注可直接对比新旧质量),验收后扩 20-30 张再全量。

## 5. 成本

- API:2 次 GPT-4o 调用/图(proposal + verifier),5 图试点 ≈ 10 次,可忽略。
- 本地:Grounding DINO ~0.5s/查询、SAM ~0.2s/框、OCR ~1s/图(RTX 5080);模型下载约 3-4 GB 一次性。
