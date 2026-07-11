"""线索子集消融:纯色块屏蔽 + 非空子集枚举。"""
import numpy as np
import pytest
from PIL import Image

from clue_leak.masking import mask_solid_regions, mask_solid_from_masks
from clue_leak.combo import nonempty_subsets


def img(w=100, h=80):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "RGB")


# ---------- 纯色块屏蔽 ----------

def test_mask_solid_fills_box_with_gray():
    out = mask_solid_regions(img(), [[10, 10, 50, 40]])
    a = np.asarray(out)
    assert (a[10:40, 10:50] == 128).all()          # 默认中性灰
    assert not (np.asarray(img())[10:40, 10:50] == 128).all()


def test_mask_solid_outside_unchanged_input_untouched():
    src = img()
    before = np.asarray(src).copy()
    out = mask_solid_regions(src, [[10, 10, 50, 40]])
    a, b = np.asarray(src), np.asarray(out)
    assert np.array_equal(a, before)                # 不就地修改
    assert np.array_equal(a[50:80, :, :], b[50:80, :, :])
    assert np.array_equal(a[:, 60:100, :], b[:, 60:100, :])


def test_mask_solid_custom_color_and_multiple_boxes():
    out = mask_solid_regions(img(), [[0, 0, 10, 10], [90, 70, 100, 80]], color=(0, 0, 0))
    a = np.asarray(out)
    assert (a[0:10, 0:10] == 0).all() and (a[70:80, 90:100] == 0).all()


def test_mask_solid_clips_out_of_bounds():
    out = mask_solid_regions(img(), [[-10, -10, 300, 300]])
    assert (np.asarray(out) == 128).all()
    assert out.size == img().size


def test_mask_solid_empty_boxes_is_identity():
    src = img()
    assert np.array_equal(np.asarray(mask_solid_regions(src, [])), np.asarray(src))


# ---------- 不规则掩码涂灰 ----------

def test_mask_from_masks_fills_only_true_pixels():
    m = np.zeros((80, 100), dtype=bool)
    m[20:40, 30:50] = True          # 一块不规则(此处矩形便于断言)区域
    out = np.asarray(mask_solid_from_masks(img(), [m]))
    assert (out[20:40, 30:50] == 128).all()
    assert not (np.asarray(img())[20:40, 30:50] == 128).all()
    # 掩码外像素不变
    assert np.array_equal(out[0:20, :], np.asarray(img())[0:20, :])


def test_mask_from_masks_union_and_shape_guard():
    m1 = np.zeros((80, 100), dtype=bool); m1[0:10, 0:10] = True
    m2 = np.zeros((80, 100), dtype=bool); m2[70:80, 90:100] = True
    wrong = np.ones((5, 5), dtype=bool)          # 尺寸不符 → 忽略,不报错
    out = np.asarray(mask_solid_from_masks(img(), [m1, m2, wrong]))
    assert (out[0:10, 0:10] == 128).all() and (out[70:80, 90:100] == 128).all()


def test_mask_from_masks_empty_is_identity():
    src = img()
    assert np.array_equal(np.asarray(mask_solid_from_masks(src, [])), np.asarray(src))


# ---------- 子集枚举 ----------

def test_nonempty_subsets_m3():
    subs = nonempty_subsets(3)
    assert subs == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def test_nonempty_subsets_sizes():
    assert len(nonempty_subsets(2)) == 3
    assert len(nonempty_subsets(4)) == 15
    # 按大小升序,同大小按字典序 → 输出稳定可缓存
    sizes = [len(s) for s in nonempty_subsets(4)]
    assert sizes == sorted(sizes)
