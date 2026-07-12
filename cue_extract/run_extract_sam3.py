"""线索提取(route B + SAM3):
  ① GPT-4o grounded 定位推理(干净原图)→ 自报线索 {name, category, reasoning, confidence}
  ② 把每条线索的**语义名**当文字 prompt 喂 SAM3 → 干净掩码(替掉 VLM 画框/自动分割)
  ③ flag_degenerate + assign_maskable
输出 cue_extract/results_sam3/<id>.json(同下游 schema)+ figures_sam3/ 标注图。
venv 运行:cue_extract/.venv/Scripts/python.exe -m cue_extract.run_extract_sam3 [--ids ...]
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

from cue_extract.grounded import locate_and_ground
from cue_extract.merge import assign_maskable, flag_degenerate
from cue_extract.rle import mask_to_rle
from cue_extract.sam3_seg import segment_with_fallback
from cue_extract.viz import render
from run import build_client, load_config

OUTDIR = os.path.join(os.path.dirname(__file__), "results_sam3")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures_sam3")

PILOT_IDS = [
    "181848051_3f34be1b5d_68_70323761@N00.jpg", "261517384_292417efcc_117_60558526@N00.jpg",
    "311344213_4b003f4ab1_114_63163416@N00.jpg", "370717727_f9564e3587_150_13527886@N00.jpg",
    "847733166_0573338bfb_1321_89904893@N00.jpg",
]


def load_subsets():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); out.setdefault(it["image_id"], it)
    return out


def extract_one(client, image, seg_thresh=0.3):
    res = locate_and_ground(client, image)              # ① GPT-4o 自报线索(干净图)+ segment_query
    gp, unlocalized = [], 0
    for c in res["cues"]:
        insts, used = segment_with_fallback(            # ② SAM3 回退链(query→name→变体)
            image, c.get("segment_query", ""), c["cue"], threshold=seg_thresh)
        entry = {"cue": c["cue"], "category": c["category"], "is_text": c["is_text"],
                 "reasoning": c["reasoning"], "confidence": c["confidence"],
                 "segment_query": c.get("segment_query", ""), "used_query": used}
        if insts:
            entry["instances"] = [{"bbox": ins["bbox"], "score": round(ins["score"], 3),
                                   "source": "sam3", "mask_rle": mask_to_rle(ins["mask"])}
                                  for ins in insts]
        else:
            entry["instances"] = []; unlocalized += 1   # 保留但未定位(如实记录,不进消融)
        gp.append(entry)
    gp = assign_maskable(flag_degenerate(gp, image.size))   # ③(无实例 → maskable=False)
    return {"location_guess": res["location_guess"], "geo_privacy_cues": gp,
            "n_unlocalized": unlocalized}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(PILOT_IDS))
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]
    os.makedirs(OUTDIR, exist_ok=True)
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
            rec = extract_one(client, img, args.thresh)
            rec["image_id"] = iid; rec["image_size"] = list(img.size)
            json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False)
            ng = sum(1 for c in rec["geo_privacy_cues"] if c.get("maskable"))
            print(f"[{i}/{len(ids)}] {iid[:22]} guess='{rec['location_guess'][:22]}' "
                  f"cues={len(rec['geo_privacy_cues'])} maskable={ng} ({time.time()-t0:.0f}s)", flush=True)
        render(img, rec["geo_privacy_cues"],
               os.path.join(FIGDIR, "cuesSAM3_" + iid.split("_")[0] + ".png"),
               title=f"{iid.split('_')[0]} — guess: {rec['location_guess'][:26]} — {len(rec['geo_privacy_cues'])} cues (SAM3)")
    print("done", flush=True)


if __name__ == "__main__":
    main()
