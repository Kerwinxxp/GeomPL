"""clue_leak 噪声遮挡：对指定 bbox 区域加高斯噪声,破坏其中的可定位线索。"""
import numpy as np
import pytest
from PIL import Image

from clue_leak.masking import add_laplace_noise_to_regions, add_noise_to_regions


def gray(w=100, h=80, v=128):
    return Image.new("RGB", (w, h), (v, v, v))


def test_region_inside_box_changes():
    img = gray()
    out = add_noise_to_regions(img, [[20, 20, 60, 60]], sigma=80, seed=1)
    a = np.asarray(img, dtype=int)
    b = np.asarray(out, dtype=int)
    # 框内像素被改动
    assert not np.array_equal(a[30:50, 30:50], b[30:50, 30:50])


def test_region_outside_box_unchanged():
    img = gray()
    out = add_noise_to_regions(img, [[20, 20, 60, 60]], sigma=80, seed=1)
    a = np.asarray(img, dtype=int)
    b = np.asarray(out, dtype=int)
    # 框外像素完全不变
    assert np.array_equal(a[0:15, 0:15], b[0:15, 0:15])
    assert np.array_equal(a[70:80, 80:100], b[70:80, 80:100])


def test_sigma_zero_leaves_image_unchanged():
    img = gray()
    out = add_noise_to_regions(img, [[10, 10, 50, 50]], sigma=0.0, seed=1)
    assert np.array_equal(np.asarray(img), np.asarray(out))


def test_deterministic_given_seed():
    img = gray()
    a = add_noise_to_regions(img, [[10, 10, 50, 50]], sigma=60, seed=42)
    b = add_noise_to_regions(img, [[10, 10, 50, 50]], sigma=60, seed=42)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_output_size_mode_preserved_and_input_untouched():
    img = gray()
    before = np.asarray(img).copy()
    out = add_noise_to_regions(img, [[10, 10, 50, 50]], sigma=50, seed=1)
    assert out.size == img.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(img), before)   # 原图不被就地修改


def test_multiple_boxes_all_masked():
    img = gray()
    out = add_noise_to_regions(img, [[5, 5, 20, 20], [70, 60, 95, 78]], sigma=80, seed=1)
    b = np.asarray(out, dtype=int); a = np.asarray(img, dtype=int)
    assert not np.array_equal(a[8:18, 8:18], b[8:18, 8:18])
    assert not np.array_equal(a[62:76, 72:92], b[62:76, 72:92])
    assert np.array_equal(a[30:50, 30:50], b[30:50, 30:50])   # 中间无框区不变


def test_out_of_bounds_box_is_clipped():
    img = gray()
    # 越界框不应报错,按图界裁剪
    out = add_noise_to_regions(img, [[-10, -10, 300, 300]], sigma=40, seed=1)
    assert out.size == img.size


def test_noise_values_stay_in_valid_range():
    img = gray(v=250)   # 接近上界,验证钳制到 [0,255]
    out = add_noise_to_regions(img, [[10, 10, 50, 50]], sigma=120, seed=3)
    arr = np.asarray(out)
    assert arr.min() >= 0 and arr.max() <= 255


# ---------- 拉普拉斯机制(像素级 ε-DP) ----------

def _mean_abs_perturb(orig, out, box):
    x1, y1, x2, y2 = box
    a = np.asarray(orig, dtype=float)[y1:y2, x1:x2]
    b = np.asarray(out, dtype=float)[y1:y2, x1:x2]
    return float(np.abs(a - b).mean())


def test_laplace_region_changes_outside_unchanged():
    img = gray()
    out = add_laplace_noise_to_regions(img, [[20, 20, 60, 60]], epsilon=1.0, seed=1)
    a, b = np.asarray(img, dtype=int), np.asarray(out, dtype=int)
    assert not np.array_equal(a[30:50, 30:50], b[30:50, 30:50])
    assert np.array_equal(a[0:15, 0:15], b[0:15, 0:15])


def test_laplace_smaller_epsilon_more_noise():
    # ε 越小 → 噪声尺度 b=Δ/ε 越大 → 平均扰动越大
    img = gray()
    box = [10, 10, 70, 70]
    hi_priv = add_laplace_noise_to_regions(img, [box], epsilon=0.5, seed=7)   # 强隐私
    lo_priv = add_laplace_noise_to_regions(img, [box], epsilon=10.0, seed=7)  # 弱隐私
    assert _mean_abs_perturb(img, hi_priv, box) > _mean_abs_perturb(img, lo_priv, box)


def test_laplace_deterministic_given_seed():
    img = gray()
    a = add_laplace_noise_to_regions(img, [[10, 10, 50, 50]], epsilon=2.0, seed=42)
    b = add_laplace_noise_to_regions(img, [[10, 10, 50, 50]], epsilon=2.0, seed=42)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_laplace_clipped_to_valid_range():
    img = gray(v=250)
    out = add_laplace_noise_to_regions(img, [[10, 10, 50, 50]], epsilon=0.3, seed=3)
    arr = np.asarray(out)
    assert arr.min() >= 0 and arr.max() <= 255


def test_laplace_input_untouched_and_size_preserved():
    img = gray()
    before = np.asarray(img).copy()
    out = add_laplace_noise_to_regions(img, [[10, 10, 50, 50]], epsilon=1.0, seed=1)
    assert out.size == img.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(img), before)
