# GeoBayes — MLLM 地理定位攻击者建模 & 逐线索位置隐私泄露

用多模态大模型(GPT-4o)作为"位置隐私攻击者",研究图像中**哪些视觉线索泄露了地理位置、各自泄露多少**。

仓库包含两条独立的线:

| 线 | 目录 | 状态 | 说明 |
|---|---|---|---|
| **① 逐线索 mPL 泄露研究**（当前主线）| `cue_extract/` + `clue_leak/` | 活跃 | 提取图中定位线索 → SAM 掩码 → 逐条/组合遮蔽 → 用 mPL 量化每条线索的位置泄露 |
| **② GeoBayes 论文复现**（存档）| `geobayes/` + `scripts/` | 冻结 | 复现 GeoBayes(AAAI-26)训练-free 贝叶斯地理定位;详见 [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) |

> 度量定义:**mPL**(metric-normalized posterior leakage,Chen et al. 2026)= 逐候选对 `|Δln后验-odds − Δln先验-odds| / 地理距离`。这里先验 = 遮掉线索后的图、后验 = 原图,mPL 越大 = 该线索携带的位置信息越多。

---

## 快速试用（无需下载数据集 / API / GPU）

仓库自带 **5 张样例图**(`data/sample_images/`)及其**预算好的结果**(线索标注 + 消融 + 后验),clone 后装好 `matplotlib` 即可直接复现单图 mPL 图:

```bash
pip install pillow matplotlib pyyaml       # 出图只需这几个
python -m clue_leak.plot_one_mpl cuba      # 也可 newyork / okazaki / newdelhi / venice
# → clue_leak/figures/per_image_mpl/cuba_370717727_mpl.png
python -m clue_leak.plot_clue_mpl          # 跨样例的线索 mPL 排序图
```
样例覆盖 5 种典型:Cuba(单点要害)、New York(强文字/冗余)、Okazaki(文化线索)、New Delhi(完美冗余)、Venice(建筑+铭文)。想重跑消融(需 `OPENAI_API_KEY`):`python -m clue_leak.run_combo2 --ids <见 data/subset_sample.jsonl>`——gallery 已固化在 `data/gallery_labels.json`,结果与全量可比。

## 环境

两套环境(主逻辑纯 Python;线索提取需 GPU 深度模型):

```bash
# 主环境（打分/分析/绘图）
pip install -r requirements.txt      # pillow, pyyaml, openai, pytest
export OPENAI_API_KEY=sk-...          # 攻击者模型（GPT-4o），代码只从环境变量读

# cue_extract GPU 栈（Grounding DINO + SAM 2.1 + RapidOCR + LaMa，需 CUDA）
python -m venv cue_extract/.venv
cue_extract/.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cue_extract/.venv/Scripts/pip install transformers accelerate rapidocr onnxruntime \
    simple-lama-inpainting scipy pillow matplotlib numpy openai pyyaml
```
模型权重(Grounding DINO ~700MB、SAM2.1、PP-OCRv5、LaMa ~200MB)首次运行自动下载到 HuggingFace 缓存。实测显卡:RTX 5080 16GB。

## 数据

入库的数据:`data/subset_sample.jsonl`(5 张样例)、`data/sample_images/`(样例图)、`data/gallery_labels.json`(固化的 77 候选集)、`data/*_cache.json`(地理编码缓存)。

全量复现需源数据集 **IM2GPS3k**(不入库,见 `.gitignore`)。`data/im2gps3k.csv` 放好后:

```bash
python scripts/build_subset.py --n 100    # 从 Flickr 拉图 + 反向地理编码 → data/subset100.jsonl
```
地理编码缓存(`data/*_cache.json`)已入库,可直接复用。

---

## 主线 ① 运行顺序

```bash
# 1) 建候选集几何缓存（gallery 标签 → 坐标，供 mPL 距离矩阵）
python -m clue_leak.prep_geo100

# 2) 线索提取（venv）：VLM 提议 → Grounding DINO 定位 → OCR → SAM 掩码 → VLM 风险审计
cue_extract/.venv/Scripts/python -m cue_extract.run_extract --ids <id1,id2,...>
#   产物：cue_extract/results/<id>.json（含每条线索的 mask_rle）
#   质检总览：python -m cue_extract.contact_sheet → cue_extract/figures/contact_sheet.png

# 3) 逐线索 mPL 消融（主环境）：先验=遮子集S的图，后验=原图
python -m clue_leak.run_combo2 --ids <id1,id2,...>
#   或批量凑 N 张：python -m clue_leak.run_50 --target 50
#   产物：clue_leak/combo2_results/<id>.json

# 4) 出图
python -m clue_leak.plot_one_mpl <id前缀>   # 单图：原图+掩码 | 每条线索&组合的 mPL
python -m clue_leak.plot_clue_mpl           # 跨图：所有线索 mPL 排序
```
输出图在 `clue_leak/figures/per_image_mpl/`(每张一个样本)。

## 目录结构

```
cue_extract/        线索提取管线（GPU）
  proposal / grounding / ocr / sam_mask / verify / merge / rle / prompts / viz
  run_extract.py    编排器  ·  contact_sheet.py 质检总览  ·  inpaint.py + viz_mask_compare.py 稳健性检查
clue_leak/          逐线索 mPL 消融
  combo.py masking.py           子集枚举 + 掩码涂灰
  run_combo2.py run_50.py       消融运行器 / 批量驱动
  plot_one_mpl.py plot_clue_mpl.py plot_combo2.py   绘图
  prep_geo100.py                几何缓存准备
  cache_fullimage_posterior/    原图后验缓存（可选，run_combo2 缺则现算）
  combo2_results/               消融结果  ·  figures/  产出图
geobayes/           【存档】GeoBayes 复现：core(贝叶斯循环)/search/eval/mllm/analysis
scripts/            【存档】复现用批处理 + 数据构建（build_subset.py 仍在用）
data/               subset*.jsonl + 地理编码缓存（源图不入库）
tests/              pytest（主线 + 复现，190 passing）
```

## 测试

```bash
python -m pytest tests/ -q          # 全部（含复现存档）
python -m pytest tests/test_clueleak_combo.py tests/test_cue_extract.py -q   # 仅主线
```

## 已知局限（写论文注意）

- mPL 是**边际泄露**(leave-one-out),受线索冗余影响;绝对值偏小(全 pair 平均 + 距离归一);
- 遮蔽组合的价值函数**非单调**,不满足 Shapley 可加性 —— 只报单条 mPL,组合仅作定性;
- 结论是**特定攻击者(GPT-4o)+ 特定候选集**下的,非普适。

## 参考

- GeoBayes: Shi et al., AAAI-26
- mPL: Chen et al., 2026, *Metric-Normalized Posterior Leakage*
- Grounding DINO · SAM 2.1 · PaddleOCR/RapidOCR · LaMa · IM2GPS3k
