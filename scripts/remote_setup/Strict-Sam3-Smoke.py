"""Run pinned SAM3 image-and-text segmentation and validate its mask."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import Sam3Model, Sam3Processor


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: Strict-Sam3-Smoke.py MODEL_ID REVISION")
    model_id, revision = sys.argv[1:]
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    assert snapshot.name == revision, f"resolved {snapshot.name}, expected {revision}"
    root = Path.cwd()
    image_path = root / "data" / "sample_images" / "261517384_292417efcc_117_60558526@N00.jpg"
    image = Image.open(image_path).convert("RGB")
    processor = Sam3Processor.from_pretrained(snapshot, local_files_only=True)
    model = Sam3Model.from_pretrained(snapshot, local_files_only=True).to("cuda").eval()
    inputs = processor(images=image, text="Japanese banner", return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs)
    result = processor.post_process_instance_segmentation(
        output, threshold=0.3, target_sizes=[(image.height, image.width)]
    )[0]
    masks = result.get("masks")
    assert masks is not None and len(masks) > 0, "SAM3 returned no mask"
    mask = masks[0].detach().cpu().numpy()
    assert np.isfinite(mask).all(), "SAM3 returned non-finite mask values"
    mask = mask.astype(bool)
    assert mask.shape == (image.height, image.width), mask.shape
    area = int(np.count_nonzero(mask))
    assert 0 < area < mask.size, f"invalid mask area {area}/{mask.size}"
    print(f"PASS SAM3 image+text segmentation revision={revision} area={area}")


if __name__ == "__main__":
    main()
