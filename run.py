"""GeoBayes v1 单图入口：image → {prior, trajectory, final_posterior, ...} JSON。

用法：
    python run.py photo.jpg                      # 真实 MLLM（需 DASHSCOPE_API_KEY）
    python run.py photo.jpg -o out.json
    python run.py photo.jpg --config config.yaml
"""
import argparse
import json
import os

from PIL import Image

from geobayes.core.controller import Controller


def load_config(path: str | None) -> dict:
    """path=None → 空配置；显式给出的路径不存在 → 报错（防止静默用默认值跑付费 API）。"""
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    import yaml
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded


def build_client(config: dict):
    from geobayes.mllm.client import MLLMClient
    return MLLMClient(
        model=config.get("model", "qwen2.5-vl-7b-instruct"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        cache_dir=config.get("cache_dir", ".mllm_cache"),
        temperature=config.get("decode_temperature", 0.0),
        max_pixels=config.get("max_pixels", 1280 * 28 * 28),
    )


def build_search_client(config: dict):
    """v2 才构建搜索客户端（Tavily + SerpApi，key 从环境变量读取）。"""
    if not (config.get("enable_hierarchy") or config.get("enable_enhance")):
        return None
    from geobayes.search.client import WebSearchClient
    return WebSearchClient(
        cache_dir=config.get("search_cache_dir", ".search_cache"),
        max_results=config.get("search_max_results", 3),
    )


def run_image(image_path: str, client=None, config: dict | None = None,
              output_path: str | None = None, search_client="auto") -> dict:
    config = config or {}
    client = client or build_client(config)
    if search_client == "auto":
        search_client = build_search_client(config)
    image = Image.open(image_path)
    result = Controller(client, config, search_client=search_client).run(image)
    result["image"] = image_path
    result["model"] = getattr(client, "model", "mock")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="GeoBayes v1: image -> belief distributions")
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default=None, help="write result JSON here")
    ap.add_argument("--config", default=None,
                    help="config yaml (default: ./config.yaml if present)")
    args = ap.parse_args()

    cfg_path = args.config
    if cfg_path is None and os.path.exists("config.yaml"):
        cfg_path = "config.yaml"
    result = run_image(args.image, config=load_config(cfg_path),
                       output_path=args.output)
    print(json.dumps(
        {k: result[k] for k in ("prior", "final_posterior", "map_estimate", "events")},
        ensure_ascii=False, indent=2))
    if args.output:
        print(f"\nfull trajectory written to {args.output}")


if __name__ == "__main__":
    main()
