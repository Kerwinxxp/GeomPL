"""【实验性 · 可整体删除】GeoCLIP 作为"信念计":对自定义坐标 gallery 输出连续 softmax 概率。

动机:GPT-4o 嘴报打分量化在 0.1 网格 + 0.05 地板,对遮蔽失明。
GeoCLIP(检索式)天生输出闭集上的连续 softmax → 遮线索会平滑改变图像 embedding → 概率连续变。
且 GeoCLIP 训练集 MP-16(Flickr)与 im2gps3k 同源,gallery 天生就是 GPS 坐标(匹配 haversine mPL)。

补丁:transformers 5.13 的 CLIP.get_image_features 返回对象(且池化维度变),geoclip 1.2 期望旧张量。
     这里 monkeypatch ImageEncoder.forward,手动走 vision_model→visual_projection 复现旧的 768-d 投影。
运行环境:cue_extract/.venv(有 torch + transformers)。
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_MODEL = None


def _patched_image_forward(self, x):
    """复现旧 get_image_features:vision_model 池化 → visual_projection(768) → mlp。"""
    vo = self.CLIP.vision_model(pixel_values=x)
    emb = self.CLIP.visual_projection(vo.pooler_output)    # (n, 768)
    return self.mlp(emb)


def get_model(device=None):
    global _MODEL
    if _MODEL is None:
        from geoclip import GeoCLIP
        from geoclip.model.image_encoder import ImageEncoder
        ImageEncoder.forward = _patched_image_forward       # 打补丁(兼容 transformers 5.x)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = GeoCLIP().eval().to(dev)
        _MODEL._dev = dev
    return _MODEL


def score_gallery(pil_image, coords, model=None, smoothing=0.0):
    """pil_image + coords[list[(lat,lon)]] → {i: prob}。连续 softmax,可选轻度平滑保非零。

    coords 顺序即返回 dict 的键顺序(0..N-1)。与主管线一致:调用方自行把 i 映射回标签。
    """
    m = model or get_model()
    dev = m._dev
    x = m.image_encoder.preprocess_image(pil_image.convert("RGB")).to(dev)
    gps = torch.tensor([[c[0], c[1]] for c in coords], dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = m(x, gps)                 # (1, N) = logit_scale * cos(img, loc)
        probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
    n = len(coords)
    out = {i: float(probs[i]) for i in range(n)}
    if smoothing > 0:                      # 与 client 一致的轻度平滑(可选,mPL 尾部保护更常用)
        out = {i: (1 - smoothing) * v + smoothing / n for i, v in out.items()}
        z = sum(out.values()); out = {i: v / z for i, v in out.items()}
    return out


def score_labels(pil_image, labels, coord_cache, model=None, smoothing=0.0):
    """便捷版:labels[list[str]] + 名->坐标缓存 → {label: prob}(只保留有坐标的标签)。"""
    have = [l for l in labels if coord_cache.get(l)]
    coords = [coord_cache[l] for l in have]
    p = score_gallery(pil_image, coords, model=model, smoothing=smoothing)
    return {have[i]: p[i] for i in range(len(have))}
