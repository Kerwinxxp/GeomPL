"""端到端冒烟（Phase 3 步骤 2）：磁盘图片 → run_image → 完整 JSON 落盘。"""
import json

from PIL import Image

from geobayes.mllm.mock_client import MockClient
from run import run_image


def default_script():
    return {
        "hypothesize": {
            "level": "country",
            "scene_summary": "urban street",
            "candidates": [
                {"location": "United Kingdom", "confidence": 0.6},
                {"location": "United States", "confidence": 0.4},
                {"location": "Sweden", "confidence": 0.3},
            ],
            "verification_tasks": [
                {"desc": "verify bus", "reason": "livery hints", "bbox": [10, 10, 120, 90]},
                {"desc": "check plates", "reason": "plate style", "bbox": [40, 60, 160, 110]},
            ],
        },
        "verify": [{"observation": "a red double-decker bus", "geo_clues": ["bus"]},
                   {"observation": "yellow rear plate", "geo_clues": ["plate"]}],
        "judge": [
            {"ratings": {"United Kingdom": {"c": 5, "alpha": 0.45},
                         "United States": {"c": 2, "alpha": 0.89},
                         "Sweden": {"c": 2, "alpha": 0.84}}},
            {"ratings": {"United Kingdom": {"c": 4, "alpha": 0.9},
                         "United States": {"c": 2, "alpha": 0.5},
                         "Sweden": {"c": 3, "alpha": 0.2}}},
        ],
    }


def test_run_image_end_to_end(tmp_path):
    img_path = tmp_path / "photo.png"
    Image.new("RGB", (320, 240), "gray").save(img_path)
    out_path = tmp_path / "result.json"

    result = run_image(str(img_path), client=MockClient(default_script()),
                       config={"max_replace": 0}, output_path=str(out_path))

    # 两个目标分布齐备且归一
    assert abs(sum(result["prior"]["hypotheses"].values()) - 1) < 1e-9
    assert abs(sum(result["final_posterior"]["hypotheses"].values()) - 1) < 1e-9
    assert result["prior"]["raw_scores"]["United Kingdom"] == 0.6
    assert len(result["trajectory"]) == 2
    assert result["map_estimate"] == "United Kingdom"
    assert result["image"] == str(img_path)

    # 落盘文件与返回值一致
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(result))
