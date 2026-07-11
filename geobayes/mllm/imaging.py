"""图像工具。

- crop_with_padding：bbox 取 [x1,y1,x2,y2]（Eq.9/Fig.3 实例格式），10% padding，
  钳制图界；退化框（越界后短边 < min_side 或坐标非法）回退全图并置 fallback 标志。
- smart_resize_dims：复刻 qwen_vl_utils.smart_resize——本地先 resize 再上传，
  使模型坐标空间 == 我方持有像素空间（map §4 坐标一致性方案）。
"""
import math


def crop_with_padding(img, bbox, pad_ratio: float = 0.10, min_side: int = 16):
    """返回 (crop_image, used_bbox[x1,y1,x2,y2], fallback_flag)。"""
    W, H = img.size
    full = [0, 0, W, H]
    if not bbox or len(bbox) != 4:
        return img.copy(), full, True
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return img.copy(), full, True
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return img.copy(), full, True
    if x2 <= x1 or y2 <= y1:
        return img.copy(), full, True

    pw, ph = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    x1, y1 = max(0, int(math.floor(x1 - pw))), max(0, int(math.floor(y1 - ph)))
    x2, y2 = min(W, int(math.ceil(x2 + pw))), min(H, int(math.ceil(y2 + ph)))
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return img.copy(), full, True
    return img.crop((x1, y1, x2, y2)), [x1, y1, x2, y2], False


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
