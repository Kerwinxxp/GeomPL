"""③ OCR 分支:RapidOCR(PP-OCR 模型 + onnxruntime)→ {text, bbox, conf}(PLAN §2③)。

问题②修复:默认从 PP-OCRv6 small 升级到 **PP-OCRv5 server(ch)**——覆盖拉丁字母/数字/
横排 CJK,精度大幅提升(实测 NYC "ROMENA"→"PROMENADE"、Biwer 网址满分)。
已知局限:竖排 CJK(如日文祭典横幅)行式 OCR 读不出——语义线索由 proposal 捕获、
遮蔽用 grounding 框,只是缺文字转写。懒加载单例。
"""
_OCR = None


def _load():
    global _OCR
    if _OCR is None:
        from rapidocr import ModelType, OCRVersion, RapidOCR
        _OCR = RapidOCR(params={
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER,
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.SERVER,
        })
    return _OCR


def run_ocr(image, min_conf: float = 0.5) -> list:
    """PIL 图 → [{text, bbox:[x1,y1,x2,y2], conf}](conf 过阈值)。"""
    import numpy as np
    ocr = _load()
    arr = np.asarray(image.convert("RGB"))
    result = ocr(arr)
    out = []
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or txts is None:
        return out
    for p, t, s in zip(boxes, txts, scores or [1.0] * len(txts)):
        if float(s) < min_conf or not str(t).strip():
            continue
        xs = [float(pt[0]) for pt in p]
        ys = [float(pt[1]) for pt in p]
        out.append({"text": str(t), "conf": round(float(s), 3),
                    "bbox": [round(min(xs), 1), round(min(ys), 1),
                             round(max(xs), 1), round(max(ys), 1)]})
    return out
