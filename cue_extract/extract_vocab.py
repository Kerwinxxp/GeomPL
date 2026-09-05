"""Bottom-up 线索提取:固定通用词表 + SAM 3(无 GPT-4o,确定性,跨图可比)。

与 run_extract_sam3.py 的对比实验:那条线是 GPT-4o 先自报线索名、SAM3 再按名出掩码
(top-down,词表随图而变);这里把一份**固定的 ~12 条通用地理相关概念词表**无差别地
喂给每一张图,线索 = 词表命中什么就是什么。因此:
  · 确定性(同图同结果),不花 API,跨图词表完全一致 → 掩码可直接跨图比较;
  · 覆盖率受词表限制 —— 用 compare_vocab.py 量化它盖住了多少 GPT-4o 线索。

流程(每图、每个词表概念):
  ① 对该概念的每条 SAM3 query 跑 segment_phrase(threshold 与主线一致 = 0.5);
  ② 该概念的所有实例(跨 query)合成 **一条** cue(主线也是按 cue 合并实例);
  ③ 实例级 degenerate 沿用 merge.flag_degenerate(bbox > 40% 全图);
  ④ 非退化实例的 **并集掩码** 面积 < 0.3% 或 > 40% → 丢弃该 cue(记入 dropped_cues)。

输出 cue_extract/results_vocab/<id>.json,schema 与 results_sam3 完全一致
(image_id / image_size / geo_privacy_cues[{cue, category, maskable,
instances:[{bbox, score, source, mask_rle, degenerate}]}]),下游 cue_masks_of 可直接读。

运行:cue_extract/.venv/Scripts/python.exe -m cue_extract.extract_vocab \
        --ids 158307292 754780171 --out cue_extract/results_vocab
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

import numpy as np
from PIL import Image

from cue_extract.merge import assign_maskable, flag_degenerate
from cue_extract.rle import mask_to_rle
from cue_extract.sam3_seg import segment_phrase

VOCAB_VERSION = "v1"

# 固定词表:12 条通用、与地理线索相关的概念。
#   cue      = 写进 JSON 的线索名(= 词表短语,跨图恒定)
#   category = 映射到 results_sam3 已有的分类体系(prompts.CATEGORIES)
#   queries  = 实际喂给 SAM 3 的名词短语(经单图 probe 挑过:SAM3 对 "sign" 远好于
#              "signboard"/"text";"vehicle" 在 "car" 空手时仍能命中;"facade" 与
#              "building" 互补;"monument" 补 "statue" 漏的纪念碑)。同一 cue 的多条
#              query 结果合并成一条 cue。
VOCAB = [
    {"cue": "text or signage",    "category": "text/signage",
     "queries": ["sign"], "is_text": True},
    {"cue": "building facade",    "category": "architecture",
     "queries": ["building", "facade"]},
    {"cue": "vehicle",            "category": "vehicles/license plates",
     "queries": ["vehicle", "car"]},
    {"cue": "license plate",      "category": "vehicles/license plates",
     "queries": ["license plate"], "is_text": True},
    {"cue": "person's clothing",  "category": "commercial/cultural",
     "queries": ["person's clothing"]},
    {"cue": "tree or vegetation", "category": "environment",
     "queries": ["tree or vegetation"]},
    {"cue": "mountain",           "category": "environment",
     "queries": ["mountain"]},
    {"cue": "water body",         "category": "environment",
     "queries": ["water"]},
    {"cue": "road surface",       "category": "road/infrastructure",
     "queries": ["road"]},
    {"cue": "flag",               "category": "commercial/cultural",
     "queries": ["flag"]},
    {"cue": "statue or monument", "category": "landmarks/buildings",
     "queries": ["statue", "monument"]},
    {"cue": "street furniture",   "category": "road/infrastructure",
     "queries": ["lamp post", "bench", "traffic sign"]},
]

SEG_THRESH = 0.5       # 与 run_extract_sam3.py --thresh 默认一致(results_sam3 最低分正是 0.5)
MAX_INSTANCES = 16     # 通用概念实例更多(树/招牌),放宽到 16;主线单 cue 默认 8
MAX_AREA = 0.40        # 与 merge.flag_degenerate 的 max_ratio 一致
MIN_AREA = 0.003       # 并集掩码下限:< 0.3% 视为噪声碎片

OUTDIR = os.path.join(os.path.dirname(__file__), "results_vocab")
SAM3DIR = os.path.join(os.path.dirname(__file__), "results_sam3")


def load_subsets() -> dict:
    """data/subset*.jsonl → {image_id: item};后加载(高清)覆盖先加载,与主线一致。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            out[it["image_id"]] = it
    return out


def resolve_ids(prefixes, subset) -> list:
    """把 image_id 前缀(或整名)解析成完整 image_id;要求 results_sam3 里有对照记录。"""
    ids = []
    for p in prefixes:
        if p in subset:
            ids.append(p)
            continue
        hits = sorted(k for k in subset if k.startswith(p))
        if not hits:
            raise SystemExit(f"no image_id matches prefix {p!r}")
        ids.append(hits[0])
        if len(hits) > 1:
            print(f"[warn] prefix {p!r} matched {len(hits)}, took {hits[0]}", flush=True)
    return ids


def target_size(iid, image):
    """掩码坐标系:优先复用 results_sam3 的 image_size(保证与 GPT-4o 掩码逐像素可比),
    没有对照记录时退回 smart_resize(与主线 client.prepare 同一套)。"""
    p = os.path.join(SAM3DIR, iid + ".json")
    if os.path.exists(p):
        w, h = json.load(open(p, encoding="utf-8"))["image_size"]
        return int(w), int(h)
    from geobayes.mllm.imaging import smart_resize_dims
    return smart_resize_dims(image.width, image.height, max_pixels=1280 * 28 * 28)


def _dedupe(instances, iou_thresh=0.9):
    """同一 cue 的多条 query 常命中同一物体 → 按掩码 IoU 去重,保留高分实例。"""
    kept = []
    for ins in sorted(instances, key=lambda d: -d["score"]):
        m = ins["mask"]
        a = m.sum()
        dup = False
        for k in kept:
            inter = np.logical_and(m, k["mask"]).sum()
            union = a + k["mask"].sum() - inter
            if union and inter / union >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(ins)
    return kept[:MAX_INSTANCES]


def extract_one(image, vocab=VOCAB, thresh=SEG_THRESH, verbose=True):
    """一张图 → (geo_privacy_cues, dropped_cues)。"""
    W, H = image.size
    total = float(W * H)
    raw = []
    for spec in vocab:
        insts, used = [], []
        for q in spec["queries"]:
            hits = segment_phrase(image, q, threshold=thresh, max_instances=MAX_INSTANCES)
            if hits:
                used.append(q)
            for h in hits:
                h["query"] = q
                insts.append(h)
        insts = _dedupe(insts)
        raw.append({
            "cue": spec["cue"], "category": spec["category"],
            "is_text": bool(spec.get("is_text", False)),
            "reasoning": f"fixed-vocabulary concept ({VOCAB_VERSION}); "
                         f"SAM 3 open-vocabulary hit, no VLM in the loop",
            "confidence": round(max((i["score"] for i in insts), default=0.0), 3),
            "segment_query": "|".join(spec["queries"]),
            "used_query": "|".join(used),
            "instances": [{"bbox": [float(v) for v in i["bbox"]],
                           "score": round(i["score"], 3), "source": "sam3-vocab",
                           "query": i["query"], "mask_rle": mask_to_rle(i["mask"]),
                           "_mask": i["mask"]}
                          for i in insts],
        })

    raw = assign_maskable(flag_degenerate(raw, (W, H), max_ratio=MAX_AREA))

    cues, dropped = [], []
    for c in raw:
        good = [i for i in c["instances"] if not i["degenerate"]]
        u = np.zeros((H, W), bool)
        for i in good:
            u |= i["_mask"]
        frac = float(u.sum() / total)
        for i in c["instances"]:
            i.pop("_mask", None)
        c["area_frac"] = round(frac, 5)
        c["n_instances"] = len(c["instances"])
        if not c["instances"]:
            reason = "no SAM3 hit"
        elif not good:
            reason = "all instances degenerate (bbox > 40% of image)"
        elif frac > MAX_AREA:
            reason = f"union mask {frac:.1%} > {MAX_AREA:.0%} (degenerate)"
        elif frac < MIN_AREA:
            reason = f"union mask {frac:.2%} < {MIN_AREA:.1%} (too small)"
        else:
            reason = None
        if reason:
            dropped.append({"cue": c["cue"], "category": c["category"],
                            "n_instances": c["n_instances"],
                            "area_frac": c["area_frac"], "reason": reason})
            if verbose:
                print(f"    - drop {c['cue']:20s} ({reason})", flush=True)
        else:
            c["instances"] = [dict(i) for i in c["instances"] if not i["degenerate"]]
            c["degenerate"] = False
            c["maskable"] = True
            cues.append(c)
            if verbose:
                print(f"    + {c['cue']:20s} n={len(c['instances']):2d} "
                      f"area={frac*100:5.2f}%  [{c['used_query']}]", flush=True)
    return cues, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="+", required=True,
                    help="image_id 或其前缀(如 158307292)")
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--thresh", type=float, default=SEG_THRESH)
    ap.add_argument("--force", action="store_true", help="重算已存在的结果")
    args = ap.parse_args()

    outdir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    subset = load_subsets()
    ids = resolve_ids(args.ids, subset)

    t0 = time.time()
    for k, iid in enumerate(ids, 1):
        dst = os.path.join(outdir, iid + ".json")
        if os.path.exists(dst) and not args.force:
            print(f"[{k}/{len(ids)}] cached {iid[:22]}", flush=True)
            continue
        p = subset[iid]["path"]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).convert("RGB")
        W, H = target_size(iid, img)
        img = img.resize((W, H))
        print(f"[{k}/{len(ids)}] {iid[:22]} {W}x{H}", flush=True)
        cues, dropped = extract_one(img, thresh=args.thresh)
        rec = {"image_id": iid, "image_size": [W, H],
               "source": "fixed-vocabulary + SAM3", "vocab_version": VOCAB_VERSION,
               "seg_threshold": args.thresh,
               "vocab": [v["cue"] for v in VOCAB],
               "geo_privacy_cues": cues, "dropped_cues": dropped,
               "n_unlocalized": sum(1 for d in dropped if d["reason"] == "no SAM3 hit")}
        json.dump(rec, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"    -> {len(cues)}/{len(VOCAB)} cues kept ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
