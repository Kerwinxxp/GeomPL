"""cue_extract 纯逻辑:退化框标记、证据驱动可遮判定、RLE。

重模型(SAM3/torch)不在此测——它们在 cue_extract/.venv 里用真实图验证;
本文件只测不依赖 torch 的可离线逻辑。
"""
import numpy as np

from cue_extract.merge import flag_degenerate, assign_maskable
from cue_extract.rle import mask_to_rle, rle_to_mask


# ---------- 退化框标记 ----------

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


# ---------- 证据驱动可遮判定 ----------

def test_assign_maskable_from_evidence():
    cues = [
        {"cue": "skyline", "instances": [{"bbox": [0, 0, 100, 40], "degenerate": False}]},
        {"cue": "architecture style", "instances": [{"bbox": [0, 0, 100, 90], "degenerate": True}]},
        {"cue": "cherry blossoms", "instances": []},              # 未 ground → 不可遮
    ]
    out = assign_maskable(cues)
    assert out[0]["maskable"] is True      # 有非退化框 → 可遮
    assert out[1]["maskable"] is False     # 只有退化框
    assert out[2]["maskable"] is False     # 无框


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
