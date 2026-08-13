"""逐线索 mPL 消融 v2:用 cue_extract 的干净标注 + **不规则 SAM 掩码**(而非方框)。

先验 = 用纯灰按掩码涂掉子集 S 的线索本体(其余可见)→ gallery 打分;
后验 = 完整原图信念(优先复用 cache_post_hires/,缺则现算,自洽);
mPL(先验_S → 后验) = 子集 S 的泄露贡献。
只对 maskable + 非退化 + 有 mask 的线索消融;排除 0 可遮线索的图(g=0 整幅地标图)。
gallery = subset100 全部唯一 GT 标签。
用法：python -m clue_leak.run_combo2 --ids id1,id2,...
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

from geobayes.analysis.belief import entropy_bits, kl_bits
from clue_leak.combo import nonempty_subsets
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from run import build_client, load_config

LABELS = os.path.join(ROOT, "data", "gt_labels_cache.json")
# 默认走 route-B + SAM3 高清管线(可用 --cue_dir/--out_dir/--post_dir 覆盖)
CUEDIR = os.path.join(ROOT, "cue_extract", "results_sam3")
POSTDIR = os.path.join(os.path.dirname(__file__), "cache_post_hires")   # 原图后验缓存(可选,缺则现算)
OUTDIR = os.path.join(os.path.dirname(__file__), "combo2_sam3_results")
MAX_FULL = 6      # m<=6 枚举全部非空子集;更大只跑单条+全集(避免 2^m 爆炸)


def load_subsets():
    """合并 data/ 下所有 subset*.jsonl → {image_id: item}(样例集或全量都能定位图片路径)。"""
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            out[it["image_id"]] = it   # 覆盖:后加载(subset_sample 高清)优先
    return out


def maskable_cues(rec):
    out = []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        masks = [i["mask_rle"] for i in c.get("instances", [])
                 if (not i.get("degenerate")) and i.get("mask_rle")]
        if masks:
            out.append({"cue": c["cue"], "category": c.get("category"),
                        "risk": c.get("risk_level"), "masks": masks})
    return out


def subsets_for(m):
    if m <= MAX_FULL:
        return nonempty_subsets(m)
    singles = [(k,) for k in range(m)]
    loo = [tuple(j for j in range(m) if j != k) for k in range(m)]   # leave-one-out
    return singles + loo + [tuple(range(m))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--cue_dir", default=CUEDIR, help="线索标注目录(默认旧 5 阶段;SAM3 用 results_sam3)")
    ap.add_argument("--out_dir", default=OUTDIR, help="消融结果输出目录")
    ap.add_argument("--post_dir", default=POSTDIR, help="原图后验缓存目录;换分辨率时指到新目录以重算")
    args = ap.parse_args()
    ids = [x for x in args.ids.split(",") if x]
    _abs = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    cue_dir, out_dir, post_dir = _abs(args.cue_dir), _abs(args.out_dir), _abs(args.post_dir)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(post_dir, exist_ok=True)

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    subset = load_subsets()                 # 合并所有 data/subset*.jsonl(样例或全量都能找到图)
    labels_cache = json.load(open(LABELS, encoding="utf-8"))
    label_of = lambda it: labels_cache.get(f"{it['lat']:.5f},{it['lon']:.5f}")
    # gallery(候选集)固定:优先读固化文件,保证样例与全量用同一 77 标签集 → 结果可比
    gfile = os.path.join(ROOT, "data", "gallery_labels.json")
    if os.path.exists(gfile):
        gallery = json.load(open(gfile, encoding="utf-8"))
    else:
        gallery = sorted({label_of(it) for it in subset.values() if label_of(it)})
    print(f"gallery {len(gallery)} labels; {len(ids)} images", flush=True)

    t0 = time.time()
    for i, iid in enumerate(ids, 1):
        out = os.path.join(out_dir, iid + ".json")
        if os.path.exists(out):
            print(f"[{i}] cached {iid[:22]}", flush=True); continue
        rec = json.load(open(os.path.join(cue_dir, iid + ".json"), encoding="utf-8"))
        cues = maskable_cues(rec)
        m = len(cues)
        if m < 1:
            print(f"[{i}] skip {iid[:22]} (0 maskable)", flush=True); continue
        p = subset[iid]["path"]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = client.prepare(Image.open(p))
        tl = label_of(subset[iid])
        # 原图后验:优先复用缓存(须同分辨率!);缺失则现算并存
        post_cache = os.path.join(post_dir, iid + ".json")
        if os.path.exists(post_cache):
            posterior = json.load(open(post_cache, encoding="utf-8"))["posterior"]
        else:
            posterior = client.score_candidates(img, gallery)["prior"]
            json.dump({"posterior": posterior, "true_label": tl},
                      open(post_cache, "w", encoding="utf-8"), ensure_ascii=False)
        # 预解码每条线索的掩码并集(图尺寸)
        decoded = [[rle_to_mask(r) for r in c["masks"]] for c in cues]

        combos = []
        for S in subsets_for(m):
            masks = [mm for k in S for mm in decoded[k]]
            masked = mask_solid_from_masks(img, masks)
            prior = client.score_candidates(masked, gallery)["prior"]
            combos.append({"subset": list(S),
                           "cues": [cues[k]["cue"] for k in S],
                           "prior": prior, "kl_bits": kl_bits(posterior, prior),
                           "prior_prob_true": prior.get(tl, 0.0)})
            print(f"  [{i}/{len(ids)}] S={list(S)} KL={combos[-1]['kl_bits']:.2f}b "
                  f"p_true {combos[-1]['prior_prob_true']:.2f}->{posterior.get(tl,0):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        json.dump({"image_id": iid, "true_label": tl, "n_maskable": m,
                   "cue_meta": [{"cue": c["cue"], "category": c["category"], "risk": c["risk"]}
                                for c in cues],
                   "posterior": posterior, "post_prob_true": posterior.get(tl, 0.0),
                   "mask_type": "sam_irregular", "combos": combos},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
