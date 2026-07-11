# cue_extract — 地理线索提取管线

从一张图中提取"可能泄露地理位置的视觉线索",并给出**像素级掩码**。为下游的逐线索 mPL 消融(`clue_leak/`)提供干净的线索区域。

## 五阶段（`run_extract.extract_one`）

```
输入图
 ├─① proposal.py   GPT-4o 按 8 类 checklist 列出候选线索（只说"有什么"，不画框）
 ├─② grounding.py  Grounding DINO 把每条 cue 短语 → bbox（可多实例）
 ├─③ ocr.py        RapidOCR(PP-OCRv5 server) 读文字 → {text,bbox}，按 IoU 并入线索
 ├─④ sam_mask.py   SAM 2.1：bbox → 像素级掩码（RLE）
 └─⑤ verify.py     GPT-4o 审计：risk / geo_specificity / maskable / degenerate
```

`merge.py` 负责合并逻辑:
- `prune_uncorroborated_text_boxes` — 文字线索的框必须有 OCR 佐证(否则是 grounding 幻觉框,剔除)
- `flag_degenerate` — 实例 bbox > 40% 全图 → 标记不遮蔽(治"线索占满全画幅"退化)
- `assign_maskable` — maskable 由证据决定:有非退化实际框才可遮

`rle.py` 极简 RLE 编解码(无 pycocotools 依赖)。`viz.py` 出标注叠加图,`contact_sheet.py` 出全批质检总览,`inpaint.py`+`viz_mask_compare.py` 是灰块 vs LaMa 修复的稳健性对照。

## 输出 JSON（`cue_extract/results/<id>.json`）

```json
{
  "image_id": "...", "image_size": [W, H],
  "geo_privacy_cues": [{
    "cue": "street sign", "category": "text/signage", "is_text": true,
    "instances": [{"bbox": [x1,y1,x2,y2], "score": 0.7, "source": "grounding|ocr",
                   "mask_rle": {"size": [H,W], "counts": [...]}, "degenerate": false,
                   "text": "Denton Square"}],
    "risk_level": "high", "geo_specificity": "street-level",
    "searchability": "high", "maskable": true, "degenerate": false,
    "reason": "..."
  }]
}
```

## 运行（需 GPU venv，见根 README）

```bash
cue_extract/.venv/Scripts/python -m cue_extract.run_extract --ids <id1,id2,...>
cue_extract/.venv/Scripts/python -m cue_extract.contact_sheet   # 质检总览
```

## 已知局限
- 竖排 CJK 文字(如日文横幅)行式 OCR 读不出——语义线索仍被 proposal 捕获,遮蔽不受影响;
- 抽象/非视觉短语偶尔 grounding 乱框(已用 OCR 佐证过滤大部分);
- "整幅即地标"的图 → 无可遮线索(g=0),不进消融。
