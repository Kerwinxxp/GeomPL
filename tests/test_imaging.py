"""图像工具：bbox 裁剪（padding/钳制/退化回退）与 Qwen smart_resize 复刻。"""
import pytest
from PIL import Image

from geobayes.mllm.imaging import crop_with_padding, smart_resize_dims


def test_crop_basic_with_padding():
    img = Image.new("RGB", (200, 150))
    crop, used_bbox, fallback = crop_with_padding(img, [50, 50, 100, 100], pad_ratio=0.1)
    assert not fallback
    x1, y1, x2, y2 = used_bbox
    # padding 向外扩
    assert x1 < 50 and y1 < 50 and x2 > 100 and y2 > 100
    assert crop.size == (x2 - x1, y2 - y1)


def test_crop_clamped_to_image_bounds():
    img = Image.new("RGB", (200, 150))
    _, used_bbox, fallback = crop_with_padding(img, [-20, -20, 500, 500], pad_ratio=0.1)
    assert not fallback
    assert used_bbox == [0, 0, 200, 150]


def test_degenerate_bbox_falls_back_to_full_image():
    img = Image.new("RGB", (200, 150))
    for bad in ([5, 5, 5, 5], [30, 30, 40, 38], [80, 60, 20, 10], None):
        crop, used_bbox, fallback = crop_with_padding(img, bad, pad_ratio=0.1, min_side=16)
        assert fallback
        assert crop.size == (200, 150)
        assert used_bbox == [0, 0, 200, 150]


def test_smart_resize_multiples_of_28_and_pixel_budget():
    max_pixels = 1280 * 28 * 28
    w, h = smart_resize_dims(4032, 3024, max_pixels=max_pixels)
    assert w % 28 == 0 and h % 28 == 0
    assert w * h <= max_pixels
    # 长宽比近似保留
    assert w / h == pytest.approx(4032 / 3024, rel=0.05)


def test_nan_inf_bbox_falls_back_instead_of_crashing():
    img = Image.new("RGB", (200, 150))
    for bad in ([float("nan"), 10, 90, 70], [10, 10, float("inf"), 70],
                [10, float("-inf"), 90, float("nan")]):
        crop, used_bbox, fallback = crop_with_padding(img, bad, pad_ratio=0.1)
        assert fallback
        assert crop.size == (200, 150)


def test_smart_resize_exact_reference_value():
    # 参照 qwen_vl_utils 算法手算：4032x3024 @ max_pixels=1280*28*28 → 1148x840
    assert smart_resize_dims(4032, 3024, max_pixels=1280 * 28 * 28) == (1148, 840)


def test_smart_resize_extreme_aspect_ratio_raises():
    # 与 qwen_vl_utils 一致：长宽比 > 200 直接拒绝
    with pytest.raises(ValueError):
        smart_resize_dims(10000, 10)


def test_smart_resize_small_image_scaled_up_to_min():
    min_pixels = 56 * 56
    w, h = smart_resize_dims(40, 30, min_pixels=min_pixels)
    assert w % 28 == 0 and h % 28 == 0
    assert w * h >= min_pixels
