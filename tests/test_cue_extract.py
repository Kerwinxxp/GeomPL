"""cue_extract 纯逻辑部分:proposal 解析、IoU 合并去重、RLE、退化框标记。

重模型(Grounding DINO/SAM/OCR)不在此测——它们在 cue_extract/.venv 里用冒烟脚本验证;
本文件只测不依赖 torch 的可离线逻辑。
"""
import numpy as np
import pytest

from cue_extract.proposal import parse_proposal
from cue_extract.merge import (iou, merge_ocr_into_cues, flag_degenerate, group_instances,
                               prune_uncorroborated_text_boxes, assign_maskable)
from cue_extract.rle import mask_to_rle, rle_to_mask


# ---------- proposal 解析 ----------

def test_parse_proposal_flattens_categories_and_fills_fields():
    raw = {"cues": {
        "text/signage": [{"cue": "street sign", "grounding_phrase": "street sign", "is_text": True}],
        "environment": [{"cue": "palm trees", "grounding_phrase": "palm tree"}],
        "landmarks/buildings": [],
    }}
    out = parse_proposal(raw)
    assert len(out) == 2
    assert out[0]["category"] == "text/signage" and out[0]["is_text"] is True
    assert out[1]["category"] == "environment" and out[1]["is_text"] is False  # 默认补 False


def test_parse_proposal_tolerates_flat_list():
    raw = {"cues": [{"cue": "bridge", "category": "landmarks/buildings",
                     "grounding_phrase": "bridge"}]}
    out = parse_proposal(raw)
    assert len(out) == 1 and out[0]["category"] == "landmarks/buildings"


def test_parse_proposal_skips_malformed_entries():
    raw = {"cues": {"text/signage": [{"no_cue_key": 1}, {"cue": "sign", "grounding_phrase": "sign"}]}}
    assert len(parse_proposal(raw)) == 1


# ---------- IoU / 合并 ----------

def test_iou_disjoint_and_identical():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_half_overlap():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_merge_ocr_dedupes_against_grounding_by_iou():
    cues = [{"cue": "storefront sign", "category": "text/signage", "instances":
             [{"bbox": [10, 10, 100, 40], "score": 0.8, "source": "grounding"}]}]
    ocr = [{"text": "PIZZERIA ROMA", "bbox": [12, 11, 98, 39], "conf": 0.95},   # 与上框重合 → 并入
           {"text": "Via Garibaldi 5", "bbox": [200, 200, 300, 220], "conf": 0.9}]  # 新文字线索
    out = merge_ocr_into_cues(cues, ocr, iou_thr=0.5)
    sign = out[0]
    assert sign["instances"][0]["text"] == "PIZZERIA ROMA"      # 文本并入已有实例
    news = [c for c in out if c.get("source") == "ocr"]
    assert len(news) == 1 and news[0]["text"] == "Via Garibaldi 5"
    assert news[0]["category"] == "text/signage"


def test_flag_degenerate_by_area_ratio():
    cues = [{"cue": "architecture", "instances": [{"bbox": [0, 0, 100, 90]}]},
            {"cue": "sign", "instances": [{"bbox": [0, 0, 30, 20]}]}]
    out = flag_degenerate(cues, image_size=(100, 100), max_ratio=0.4)
    assert out[0]["degenerate"] is True      # 90% > 40%
    assert out[1]["degenerate"] is False


def test_flag_degenerate_is_per_instance_good_instance_survives():
    # 一条线索一大一小两个实例:大的标退化,小的保留;线索级不连坐
    cues = [{"cue": "banner", "instances": [{"bbox": [0, 0, 100, 90]},
                                            {"bbox": [10, 10, 30, 40]}]}]
    out = flag_degenerate(cues, image_size=(100, 100), max_ratio=0.4)
    assert out[0]["instances"][0]["degenerate"] is True
    assert out[0]["instances"][1]["degenerate"] is False
    assert out[0]["degenerate"] is False     # 只要有好实例,线索不算退化


def test_prune_uncorroborated_text_boxes_drops_hallucinated_grounding():
    # 文字线索:一个 grounding 框有 OCR 文本(佐证)保留,一个无文本(幻觉)剔除
    cues = [{"cue": "website URL", "category": "text/signage", "is_text": True, "instances": [
        {"bbox": [10, 10, 60, 30], "source": "grounding", "text": "www.porsche.lu"},
        {"bbox": [200, 200, 400, 350], "source": "grounding"}]}]     # 无文本 → 幻觉框
    out = prune_uncorroborated_text_boxes(cues)
    assert len(out[0]["instances"]) == 1
    assert out[0]["instances"][0]["text"] == "www.porsche.lu"


def test_prune_keeps_ocr_source_and_nontext_cues_untouched():
    cues = [{"cue": "text: X", "category": "text/signage", "is_text": True, "source": "ocr",
             "instances": [{"bbox": [0, 0, 10, 10], "source": "ocr", "text": "X"}]},
            {"cue": "bridge", "category": "landmarks/buildings", "is_text": False,
             "instances": [{"bbox": [5, 5, 90, 40], "source": "grounding"}]}]   # 非文字不动
    out = prune_uncorroborated_text_boxes(cues)
    assert len(out[0]["instances"]) == 1 and len(out[1]["instances"]) == 1


def test_assign_maskable_from_evidence():
    cues = [
        {"cue": "skyline", "instances": [{"bbox": [0, 0, 100, 40], "degenerate": False}]},
        {"cue": "architecture style", "instances": [{"bbox": [0, 0, 100, 90], "degenerate": True}]},
        {"cue": "cherry blossoms", "instances": []},              # 未 ground → 不可遮
    ]
    out = assign_maskable(cues)
    assert out[0]["maskable"] is True      # 有非退化框 → 可遮(修 NYC 天际线误判)
    assert out[1]["maskable"] is False     # 只有退化框
    assert out[2]["maskable"] is False     # 无框


def test_group_instances_same_phrase_multiple_boxes():
    dets = [{"bbox": [0, 0, 10, 10], "score": 0.9}, {"bbox": [50, 50, 60, 60], "score": 0.7}]
    cue = {"cue": "banner", "category": "text/signage", "grounding_phrase": "banner"}
    g = group_instances(cue, dets)
    assert len(g["instances"]) == 2
    assert all(i["source"] == "grounding" for i in g["instances"])


# ---------- RLE ----------

def test_rle_roundtrip():
    m = np.zeros((6, 8), dtype=bool)
    m[2:5, 3:7] = True
    r = mask_to_rle(m)
    back = rle_to_mask(r)
    assert back.shape == (6, 8) and np.array_equal(back, m)


def test_rle_empty_and_full():
    for m in (np.zeros((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)):
        assert np.array_equal(rle_to_mask(mask_to_rle(m)), m)
