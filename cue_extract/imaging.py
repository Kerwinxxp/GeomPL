"""图像尺寸工具。

smart_resize_dims:复刻 qwen_vl_utils.smart_resize —— 本地先 resize 再上传,
使模型坐标空间 == 我方持有的像素空间(掩码与 bbox 因此逐像素可比)。
"""
import math


def smart_resize_dims(width: int, height: int, factor: int = 28,
                      min_pixels: int = 56 * 56,
                      max_pixels: int = 1280 * 28 * 28):
    """按 Qwen2.5-VL smart_resize 规则求 (new_width, new_height)。"""
    if min(width, height) <= 0:
        raise ValueError(f"invalid image dims {width}x{height}")
    if max(width, height) / min(width, height) > 200:
        # 与 qwen_vl_utils 一致：极端长宽比直接拒绝
        raise ValueError(f"aspect ratio > 200 unsupported: {width}x{height}")
    w_bar = max(factor, round(width / factor) * factor)
    h_bar = max(factor, round(height / factor) * factor)
    if w_bar * h_bar > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
    elif w_bar * h_bar < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w_bar = math.ceil(width * beta / factor) * factor
        h_bar = math.ceil(height * beta / factor) * factor
    return w_bar, h_bar
