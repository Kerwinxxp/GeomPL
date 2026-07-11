"""④ SAM 2.1:bbox prompt → 像素级 mask(PLAN §2④)。懒加载单例,输出 RLE。"""
_MODEL = None
_PROCESSOR = None
MODEL_ID = "facebook/sam2.1-hiera-large"


def _load(device="cuda"):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import Sam2Model, Sam2Processor
        _PROCESSOR = Sam2Processor.from_pretrained(MODEL_ID)
        _MODEL = Sam2Model.from_pretrained(MODEL_ID).to(device).eval()
    return _MODEL, _PROCESSOR


def boxes_to_masks(image, boxes: list, device="cuda") -> list:
    """[[x1,y1,x2,y2],...] → [RLE dict](与输入同序)。空列表直接返回空。"""
    if not boxes:
        return []
    import numpy as np
    import torch
    from .rle import mask_to_rle
    model, processor = _load(device)
    inputs = processor(images=image, input_boxes=[[list(map(float, b)) for b in boxes]],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"])[0]   # (n_boxes, 1, H, W)
    out = []
    for m in masks:
        arr = np.asarray(m[0].numpy() > 0.5, dtype=bool)
        out.append(mask_to_rle(arr))
    return out
