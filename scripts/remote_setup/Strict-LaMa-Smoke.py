"""Instantiate LaMa on CUDA and execute an inpainting forward pass."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is unavailable"
    image = Image.new("RGB", (64, 64), "steelblue")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((24, 24, 39, 39), fill=255)
    model = SimpleLama(device=torch.device("cuda"))
    output = model(image, mask)
    assert output.size == image.size, f"unexpected LaMa output size: {output.size}"
    output = output.convert("RGB")
    array = np.asarray(output)
    assert output.size == image.size
    assert array.shape == (64, 64, 3)
    assert np.isfinite(array).all()
    print("PASS LaMa CUDA forward on", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
