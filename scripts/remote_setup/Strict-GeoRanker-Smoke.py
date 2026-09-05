"""Strict scientific GeoRanker smoke test; any failed condition exits nonzero."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image

from huggingface_hub import snapshot_download
from belief_elicit import georanker_belief


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: Strict-GeoRanker-Smoke.py MODEL_ID REVISION")
    model_id, revision = sys.argv[1:]
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    assert snapshot.name == revision, f"resolved {snapshot.name}, expected {revision}"
    georanker_belief.BASE = str(snapshot)
    image_path = ROOT / "data" / "sample_images" / "261517384_292417efcc_117_60558526@N00.jpg"
    assert image_path.is_file(), f"smoke image missing: {image_path}"
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert "4090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    image = Image.open(image_path).convert("RGB")
    gps = [(34.9551, 137.1740), (51.4973, -0.1372)]
    labels = ["Okazaki, Japan", "City of Westminster, UK"]
    r1 = georanker_belief.score_rewards(image, gps, cand_texts=labels, batch_size=2)
    r2 = georanker_belief.score_rewards(image, gps, cand_texts=labels, batch_size=2)
    assert len(r1) == len(r2) == 2
    assert all(math.isfinite(x) for x in [*r1, *r2]), (r1, r2)
    assert r1[0] > r1[1], f"direction failed: Okazaki={r1[0]}, London={r1[1]}"
    delta = max(abs(a - b) for a, b in zip(r1, r2))
    assert delta < 1e-3, f"determinism failed: max delta={delta}"
    print(f"PASS GeoRanker direction margin={r1[0]-r1[1]:.6f}; determinism delta={delta:.3e}")


if __name__ == "__main__":
    main()
