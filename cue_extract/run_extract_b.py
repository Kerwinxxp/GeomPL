"""路线 B 编排器(精简版线索提取):
  ① GPT-4o grounded 定位推理 → 自报线索(name/category/bbox/reasoning/confidence)
  ② SAM 2.1: bbox → 像素级掩码
  ③ flag_degenerate + assign_maskable
输出 cue_extract/results_b/<id>.json(与旧 5 阶段同 schema,下游 run_combo2/plot 可直接用)。
不动旧的 cue_extract/results/。必须在 venv 下运行(SAM 需 torch)。
  cue_extract/.venv/Scripts/python.exe -m cue_extract.run_extract_b [--ids id1,id2]
"""
import argparse
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

from cue_extract.grounded import locate_and_ground
from cue_extract.merge import assign_maskable, flag_degenerate
from cue_extract.sam_mask import boxes_to_masks
from cue_extract.viz import render
from run import build_client, load_config

SUBSET = os.path.join(ROOT, "data", "subset_sample.jsonl")
SUBSET_FULL = os.path.join(ROOT, "data", "subset100.jsonl")
OUTDIR = os.path.join(os.path.dirname(__file__), "results_b")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures_b")

PILOT_IDS = [
    "181848051_3f34be1b5d_68_70323761@N00.jpg",   # New York
    "261517384_292417efcc_117_60558526@N00.jpg",  # Okazaki
    "311344213_4b003f4ab1_114_63163416@N00.jpg",  # New Delhi
    "370717727_f9564e3587_150_13527886@N00.jpg",  # Cuba
    "847733166_0573338bfb_1321_89904893@N00.jpg", # Venice
]


def load_subsets():
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); out.setdefault(it["image_id"], it)
    return out


def extract_one(client, image) -> dict:
    """一张(已 prepare 的)图 → {location_guess, geo_privacy_cues[...]}(同旧 schema)。"""
    res = locate_and_ground(client, image)          # ① 自报线索
    cues = res["cues"]
    boxes = [c["bbox"] for c in cues]
    rles = boxes_to_masks(image, boxes) if boxes else []   # ② SAM 掩码
    gp = []
    for c, rle in zip(cues, rles):
        gp.append({
            "cue": c["cue"], "category": c["category"], "is_text": c["is_text"],
            "reasoning": c["reasoning"], "confidence": c["confidence"],
            "instances": [{"bbox": c["bbox"], "score": c["confidence"],
                           "source": "vlm_grounded", "mask_rle": rle}],
        })
    gp = flag_degenerate(gp, image.size)            # ③ 退化标记 + 可遮判定
    gp = assign_maskable(gp)
    return {"location_guess": res["location_guess"], "geo_privacy_cues": gp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(PILOT_IDS))
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]
    os.makedirs(OUTDIR, exist_ok=True)

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    subset = load_subsets()

    t0 = time.time()
    for i, iid in enumerate(ids, 1):
        out = os.path.join(OUTDIR, iid + ".json")
        p = subset[iid]["path"]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = client.prepare(Image.open(p))
        if os.path.exists(out):
            rec = json.load(open(out, encoding="utf-8"))
            print(f"[{i}] cached {iid[:22]}", flush=True)
        else:
            rec = extract_one(client, img)
            rec["image_id"] = iid
            rec["image_size"] = list(img.size)
            json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False)
            n = len(rec["geo_privacy_cues"])
            ng = sum(1 for c in rec["geo_privacy_cues"] if c.get("maskable"))
            print(f"[{i}/{len(ids)}] {iid[:22]} guess='{rec['location_guess'][:24]}' "
                  f"cues={n} maskable={ng} ({time.time()-t0:.0f}s)", flush=True)
        render(img, rec["geo_privacy_cues"],
               os.path.join(FIGDIR, "cuesB_" + iid.split("_")[0] + ".png"),
               title=f"{iid.split('_')[0]} — guess: {rec['location_guess'][:30]} — {len(rec['geo_privacy_cues'])} cues")
    print("done", flush=True)


if __name__ == "__main__":
    main()
