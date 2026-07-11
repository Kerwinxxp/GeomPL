"""线索提取管线编排器(PLAN §1):
  ① GPT-4o proposal → ② Grounding DINO → ③ RapidOCR 合并 → ④ SAM2.1 mask
  → ⑤ GPT-4o verifier → JSON + ⑥ 可视化。
必须在 cue_extract/.venv 下运行(torch/transformers/rapidocr):
  cue_extract/.venv/Scripts/python.exe -m cue_extract.run_extract [--ids id1,id2]
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

from cue_extract.grounding import ground_cues
from cue_extract.merge import (assign_maskable, flag_degenerate, merge_ocr_into_cues,
                               prune_uncorroborated_text_boxes)
from cue_extract.ocr import run_ocr
from cue_extract.proposal import propose_cues
from cue_extract.sam_mask import boxes_to_masks
from cue_extract.verify import verify_cues
from cue_extract.viz import render
from run import build_client, load_config

SUBSET = os.path.join(ROOT, "data", "subset100.jsonl")
OUTDIR = os.path.join(os.path.dirname(__file__), "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

PILOT_IDS = [
    "1192463216_a5f06caf2c_1050_47311799@N00.jpg",   # Venice
    "181848051_3f34be1b5d_68_70323761@N00.jpg",      # New York
    "266287734_114d6cc260_94_40829484@N00.jpg",      # Biwer/Porsche
    "199802332_27a09191e6_64_85971448@N00.jpg",      # Denizli
    "261517384_292417efcc_117_60558526@N00.jpg",     # Okazaki
]


def extract_one(client, image, min_score: float = 0.3) -> list:
    """一张图 → 最终线索 JSON 列表(PLAN §3 schema)。"""
    cues = propose_cues(client, image)                       # ① API
    grounded = ground_cues(image, cues)                      # ② local
    ocr = run_ocr(image)                                     # ③ local
    merged = merge_ocr_into_cues(grounded, ocr)
    merged = prune_uncorroborated_text_boxes(merged)         # 问题①:剔除无 OCR 佐证的文字幻觉框
    merged = flag_degenerate(merged, image.size)
    boxes, owners = [], []
    for c in merged:                                          # ④ local(退化实例跳过 mask)
        for inst in c["instances"]:
            if inst.get("degenerate"):
                continue
            boxes.append(inst["bbox"]); owners.append(inst)
    for inst, rle in zip(owners, boxes_to_masks(image, boxes)):
        inst["mask_rle"] = rle
    verified = verify_cues(client, image, merged)             # ⑤ API
    return assign_maskable(verified)                          # 问题③:maskable 由证据决定


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(PILOT_IDS))
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]
    os.makedirs(OUTDIR, exist_ok=True)

    config = load_config(os.path.join(ROOT, "config.yaml"))
    client = build_client(config)
    subset = {json.loads(l)["image_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")}

    t0 = time.time()
    for i, iid in enumerate(ids, 1):
        out = os.path.join(OUTDIR, iid + ".json")
        img = client.prepare(Image.open(subset[iid]["path"]))
        if os.path.exists(out):
            cues = json.load(open(out, encoding="utf-8"))["geo_privacy_cues"]
            print(f"[{i}/{len(ids)}] cached {iid[:24]}", flush=True)
        else:
            cues = extract_one(client, img)
            json.dump({"image_id": iid, "image_size": list(img.size),
                       "geo_privacy_cues": cues},
                      open(out, "w", encoding="utf-8"), ensure_ascii=False)
            ng = sum(1 for c in cues if c["instances"])
            print(f"[{i}/{len(ids)}] {iid[:24]} cues={len(cues)} grounded={ng} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        render(img, cues, os.path.join(FIGDIR, "cues_" + iid.split("_")[0] + ".png"),
               title=f"{iid.split('_')[0]} — {len(cues)} cues")
    print("done", flush=True)


if __name__ == "__main__":
    main()
