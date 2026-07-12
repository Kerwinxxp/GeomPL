"""Set-of-Mark 步骤二:把编号区域叠到图上(半透明色块 + 轮廓 + 数字徽标),供 VLM 选区。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(229, 57, 53), (30, 136, 229), (67, 160, 71), (251, 140, 0), (142, 36, 170),
           (0, 172, 193), (240, 98, 146), (124, 179, 66), (94, 87, 194), (141, 110, 99),
           (255, 179, 0), (0, 137, 123), (216, 27, 96), (57, 73, 171), (109, 76, 65),
           (191, 54, 12)]


def _font(size):
    import os
    cands = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
             "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
    try:                                        # matplotlib 一定带 DejaVuSans
        import matplotlib
        cands.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                  "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    except Exception:
        pass
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # 新版 PIL 支持 size
    except Exception:
        return ImageFont.load_default()


def render_som(image, regions, fill=0.0, edge_width=3):
    """image: PIL RGB(已 prepare);regions: segment_regions 输出(1-based 编号 = index+1)。
    fill=0(默认,轻量):只画轮廓+数字,不涂色 → 不污染图,不带偏 VLM 的定位推理;
    fill>0:半透明填色(旧重标记,已知会污染)。返回叠加编号的新图。"""
    from scipy.ndimage import binary_erosion
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for i, r in enumerate(regions):                     # 轮廓(+可选半透明填色)
        col = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        m = r["mask"]
        if fill > 0:                                    # 轻量模式 fill=0:不涂色,不污染图
            base[m] = (1 - fill) * base[m] + fill * col
        edge = m & ~binary_erosion(m, iterations=edge_width)
        base[edge] = col
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    fs = max(14, img.width // 32)
    font = _font(fs)
    for i, r in enumerate(regions):                     # 数字徽标(白字 + 色底圆)
        col = PALETTE[i % len(PALETTE)]
        x, y = r["point"]
        label = str(i + 1)
        # 不用 draw.textbbox(在 Windows 上 FreeType getbbox 会触发 native 访问违规)
        # 直接按字号估算徽标尺寸:等宽近似,单/双位数都够包住
        tw, th = len(label) * fs * 0.62, fs
        pad = fs // 3
        rad = max(tw, th) / 2 + pad
        draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                     fill=col, outline=(255, 255, 255), width=2)
        draw.text((x - tw / 2, y - th / 2), label, fill=(255, 255, 255), font=font)
    return img
