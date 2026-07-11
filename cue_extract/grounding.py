"""② Grounding:开放词汇检测,cue 短语 → bbox(PLAN §2②)。

HuggingFace 版 Grounding DINO(纯 PyTorch,Windows 免编译)。懒加载单例。
"""
_MODEL = None
_PROCESSOR = None
MODEL_ID = "IDEA-Research/grounding-dino-base"


def _load(device="cuda"):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    return _MODEL, _PROCESSOR


def ground_phrase(image, phrase: str, device="cuda",
                  box_threshold: float = 0.3, text_threshold: float = 0.25) -> list:
    """一条短语 → [{bbox:[x1,y1,x2,y2], score}](可多实例,已按分数降序)。"""
    import torch
    model, processor = _load(device)
    text = phrase.strip().lower()
    if not text.endswith("."):
        text += "."                    # GroundingDINO 要求句号结尾
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
    dets = [{"bbox": [round(float(v), 1) for v in box], "score": round(float(s), 3)}
            for box, s in zip(res["boxes"], res["scores"])]
    return sorted(dets, key=lambda d: -d["score"])


def ground_cues(image, cues: list, device="cuda", **thr) -> list:
    """proposal 线索列表 → 各自带 instances(group_instances 规则)。"""
    from .merge import group_instances
    return [group_instances(c, ground_phrase(image, c["grounding_phrase"], device, **thr))
            for c in cues]
