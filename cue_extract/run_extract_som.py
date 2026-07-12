"""Set-of-Mark 版线索提取:
  ① SAM 自动分割 → 编号候选区域(auto_segment)
  ② 编号叠图(som.render_som)
  ③ GPT-4o 看编号图 → 选出定位用到的区域 + 语义名/理由/置信度(不画坐标)
  ④ region_id → SAM 掩码 → mask_rle;flag_degenerate + assign_maskable
输出 cue_extract/results_som/<id>.json(与旧 schema 同,下游可用)+ figures_som/ 叠加图。
venv 运行:cue_extract/.venv/Scripts/python.exe -m cue_extract.run_extract_som [--ids ...]
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image

from cue_extract.auto_segment import segment_regions
from cue_extract.merge import assign_maskable, flag_degenerate
from cue_extract.rle import mask_to_rle
from cue_extract.som import render_som
from cue_extract import prompts
from cue_extract.viz import render
from run import build_client, load_config

OUTDIR = os.path.join(os.path.dirname(__file__), "results_som")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures_som")
SOMDIR = os.path.join(os.path.dirname(__file__), "figures_som", "_marked")   # 存编号图供核对

PILOT_IDS = [
    "181848051_3f34be1b5d_68_70323761@N00.jpg", "261517384_292417efcc_117_60558526@N00.jpg",
    "311344213_4b003f4ab1_114_63163416@N00.jpg", "370717727_f9564e3587_150_13527886@N00.jpg",
    "847733166_0573338bfb_1321_89904893@N00.jpg",
]
TEXT_CATS = {"text/signage"}


def load_subsets():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); out.setdefault(it["image_id"], it)
    return out


def extract_one(client, image):
    regions = segment_regions(image)                       # ① 自动分割
    marked = render_som(image, regions)                    # ② 编号叠图
    prompt = prompts.SOM_LOCATE.replace(
        "{categories}", json.dumps(prompts.CATEGORIES, ensure_ascii=False))
    raw = client.vision_json(prompt, marked)               # ③ VLM 选区(看编号图)
    gp = []
    n = len(regions)
    for c in raw.get("cues", []):
        if not isinstance(c, dict):
            continue
        rid = c.get("region_id")
        try:
            idx = int(rid) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n):                             # 过滤幻觉编号
            continue
        cat = str(c.get("category", "other"))
        rle = mask_to_rle(regions[idx]["mask"])            # ④ 编号 → SAM 掩码
        gp.append({
            "cue": str(c.get("name", f"region {rid}")), "category": cat,
            "is_text": cat in TEXT_CATS, "reasoning": str(c.get("reasoning", "")),
            "confidence": float(c.get("confidence", 0.0) or 0.0), "region_id": idx + 1,
            "instances": [{"bbox": regions[idx]["bbox"], "score": float(c.get("confidence", 0.0) or 0.0),
                           "source": "sam_som", "mask_rle": rle}],
        })
    gp = assign_maskable(flag_degenerate(gp, image.size))
    return {"location_guess": raw.get("location_guess", ""), "geo_privacy_cues": gp,
            "n_regions": n}, marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(PILOT_IDS))
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(SOMDIR, exist_ok=True)
    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    subset = load_subsets()

    t0 = time.time()
    for i, iid in enumerate(ids, 1):
        out = os.path.join(OUTDIR, iid + ".json")
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = client.prepare(Image.open(p))
        if os.path.exists(out):
            rec = json.load(open(out, encoding="utf-8"))
            print(f"[{i}] cached {iid[:22]}", flush=True)
        else:
            rec, marked = extract_one(client, img)
            rec["image_id"] = iid; rec["image_size"] = list(img.size)
            json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False)
            marked.save(os.path.join(SOMDIR, iid.split("_")[0] + "_marked.jpg"), quality=85)
            ng = sum(1 for c in rec["geo_privacy_cues"] if c.get("maskable"))
            print(f"[{i}/{len(ids)}] {iid[:22]} regions={rec['n_regions']} "
                  f"guess='{rec['location_guess'][:22]}' cues={len(rec['geo_privacy_cues'])} "
                  f"maskable={ng} ({time.time()-t0:.0f}s)", flush=True)
        render(img, rec["geo_privacy_cues"],
               os.path.join(FIGDIR, "cuesSOM_" + iid.split("_")[0] + ".png"),
               title=f"{iid.split('_')[0]} — guess: {rec['location_guess'][:28]} — {len(rec['geo_privacy_cues'])} cues (Set-of-Mark)")
    print("done", flush=True)


if __name__ == "__main__":
    main()
