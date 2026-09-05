"""预计算 LaMa 修复图变体(单线索 / 成对 / 全体 / 等面积对照),供后续 GPU 打分消费。

打分脚本在另一个 venv(带 GeoRanker),这里只负责生成像素,不做任何打分。
产物目录:belief_elicit/inpaint_cache/<image_id>/
    s<k>.png        第 k 条线索单独修复(0-based,顺序 = 源 JSON 过滤后的线索顺序)
    p<k>-<l>.png    第 k,l 两条线索一起修复(k<l,仅 n_cues>=2)
    all.png         全部线索一起修复
    c<k>-<j>.png    第 k 条线索的等面积对照(同形状平移到非线索区)第 j 个放置 —— 修复版
    cg<k>-<j>.png   同一放置的 **灰块** 版(灰 vs 修复可在同一放置上直接对比)
    manifest.json   线索表 + 每个文件的规格 / 涉及线索 / 对照掩码 RLE / 重叠率 / 覆盖率

运行(用 cue_extract 的 venv,里面有 torch+CUDA+simple_lama_inpainting):
    cue_extract/.venv/Scripts/python.exe -m belief_elicit.precompute_inpaint
换线索源:
    ... -m belief_elicit.precompute_inpaint --src cue_extract/results_vocab \
        --out belief_elicit/inpaint_cache_vocab
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.inpaint_ops import DILATE_PX, inpaint_from_masks
from clue_leak.masking import mask_solid_from_masks
SWEEP = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")


def mask_to_rle(mask):
    """与 cue_extract.rle.mask_to_rle 输出完全一致的向量化版本(那个是逐像素 Python 循环,
    百万像素要 0.4s;这里几毫秒)。counts = 交替段长,从 False 段起(可为 0)。"""
    m = np.asarray(mask, dtype=bool)
    flat = m.ravel(order="C")
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {"size": list(m.shape), "counts": [int(c) for c in counts]}


# ---- 以下两个函数从 belief_elicit/run_georanker_control.py 原样复制 ----
# (那个文件 import GeoRanker/peft,在本 venv 里装不上,所以复制而非 import)
def translate_mask(mask, dx, dy):
    H, W = mask.shape
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    out[ny[ok], nx[ok]] = True
    return out


def sample_control(cue_mask, cue_union, rng, tries=40):
    """随机平移:与线索并集重叠尽量小、越界不超 10%。返回 (mask, overlap_frac)。"""
    H, W = cue_mask.shape
    ys, xs = np.nonzero(cue_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    area = int(cue_mask.sum())
    best, best_bad = None, 10 ** 18
    for _ in range(tries):
        dx = int(rng.integers(-x0, W - 1 - x1)) if x1 - x0 < W - 1 else 0
        dy = int(rng.integers(-y0, H - 1 - y1)) if y1 - y0 < H - 1 else 0
        t = translate_mask(cue_mask, dx, dy)
        if t.sum() < 0.9 * area:
            continue
        ov = int((t & cue_union).sum())
        if ov < best_bad:
            best_bad, best = ov, t
        if ov == 0:
            break
    return best, (best_bad / max(area, 1) if best is not None else None)


def cue_masks_of(src, iid):
    """从 <src>/<iid>.json 读线索掩码(与 run_georanker_control.cue_masks_of 同过滤规则)。

    返回 (cues, categories, masks, (W, H));源文件不存在时返回 None。
    """
    from cue_extract.rle import rle_to_mask
    p = os.path.join(src, iid + ".json")
    if not os.path.exists(p):
        return None
    rec = json.load(open(p, encoding="utf-8"))
    W, H = rec["image_size"]
    cues, cats, masks = [], [], []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        good = [i for i in c["instances"] if not i.get("degenerate") and i.get("mask_rle")]
        if not good:
            continue
        u = np.zeros((H, W), bool)
        for i in good:
            m = rle_to_mask(i["mask_rle"])
            if m.shape == (H, W):
                u |= m
        if not u.any():
            continue
        cues.append(c["cue"]); cats.append(c.get("category")); masks.append(u)
    return cues, cats, masks, (W, H)


def load_subset_paths():
    """image_id -> 原图路径(与 run_georanker_control.py 完全相同的 glob/覆盖顺序)。"""
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            subset[it["image_id"]] = it
    return subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("cue_extract", "results_sam3"),
                    help="线索来源目录(<image_id>.json,schema 同 results_sam3)")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="只跑这些 image_id 前缀(默认:sweep 里 n_cues>=1 的全部图)")
    ap.add_argument("--nctrl", type=int, default=2, help="每线索等面积对照放置数")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--dilate", type=int, default=DILATE_PX, help="修复前掩码膨胀像素")
    ap.add_argument("--out", default=os.path.join("belief_elicit", "inpaint_cache"))
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    out_root = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    sweep = json.load(open(SWEEP, encoding="utf-8"))
    all_ids = [r["image_id"] for r in sweep if r["n_cues"] >= 1]
    if args.ids:
        ids = [i for i in all_ids if any(i.startswith(p) for p in args.ids)]
        missing = [p for p in args.ids if not any(i.startswith(p) for i in all_ids)]
        if missing:
            print(f"警告:这些前缀在 sweep 里没匹配到:{missing}", flush=True)
    else:
        ids = all_ids
    print(f"src={src}\nout={out_root}\n待处理 {len(ids)} 张,nctrl={args.nctrl} "
          f"seed={args.seed} dilate={args.dilate}px", flush=True)

    subset = load_subset_paths()
    t0, n_new, n_skip, n_img = time.time(), 0, 0, 0
    for ii, iid in enumerate(ids):
        got = cue_masks_of(src, iid)
        if got is None:
            print(f"[{ii+1}/{len(ids)}] {iid[:24]} 源 JSON 缺失,跳过", flush=True)
            continue
        cues, cats, masks, (W, H) = got
        m = len(cues)
        if m == 0:
            print(f"[{ii+1}/{len(ids)}] {iid[:24]} 无可遮蔽线索,跳过", flush=True)
            continue
        d = os.path.join(out_root, iid)
        os.makedirs(d, exist_ok=True)
        p = subset[iid]["path"]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")
        union = np.zeros((H, W), bool)
        for m_ in masks:
            union |= m_
        npx = float(W * H)

        # 每图独立、由 (seed, image_id) 决定的 rng —— 保证断点续跑时放置不变
        rng = np.random.default_rng([args.seed, zlib.crc32(iid.encode("utf-8"))])

        files = []
        i_new = 0

        def emit(name, sel_masks, entry, gray=False):
            """写一个变体(已存在则跳过),并登记 manifest 条目。"""
            nonlocal n_new, n_skip, i_new
            fp = os.path.join(d, name)
            if os.path.exists(fp):
                n_skip += 1
            else:
                op = mask_solid_from_masks if gray else \
                    (lambda im, ms: inpaint_from_masks(im, ms, dilate_px=args.dilate))
                op(img, sel_masks).save(fp, format="PNG")
                n_new += 1; i_new += 1
            files.append(entry)

        # --- 单线索 / 成对 / 全体 ---
        for k in range(m):
            emit(f"s{k}.png", [masks[k]],
                 {"file": f"s{k}.png", "spec": "single", "op": "inpaint", "cues": [k],
                  "cov": float(masks[k].sum() / npx)})
        for k in range(m):
            for l in range(k + 1, m):
                mm = masks[k] | masks[l]
                emit(f"p{k}-{l}.png", [masks[k], masks[l]],
                     {"file": f"p{k}-{l}.png", "spec": "pair", "op": "inpaint",
                      "cues": [k, l], "cov": float(mm.sum() / npx)})
        emit("all.png", masks,
             {"file": "all.png", "spec": "all", "op": "inpaint",
              "cues": list(range(m)), "cov": float(union.sum() / npx)})

        # --- 等面积对照:同形状平移到非线索位置;修复版 + 同放置灰块版 ---
        for k in range(m):
            for j in range(args.nctrl):
                t, ovf = sample_control(masks[k], union, rng)
                if t is None:
                    print(f"    {iid[:20]} cue{k} ctrl{j}: 采样失败,跳过", flush=True)
                    continue
                emit(f"c{k}-{j}.png", [t],
                     {"file": f"c{k}-{j}.png", "spec": "control", "op": "inpaint",
                      "cues": [k], "placement": j, "overlap_frac": ovf,
                      "cov": float(t.sum() / npx), "mask_rle": mask_to_rle(t)})
                emit(f"cg{k}-{j}.png", [t],
                     {"file": f"cg{k}-{j}.png", "spec": "control", "op": "gray",
                      "cues": [k], "placement": j, "overlap_frac": ovf,
                      "cov": float(t.sum() / npx), "mask_from": f"c{k}-{j}.png"},
                     gray=True)

        json.dump({"image_id": iid, "src": os.path.relpath(src, ROOT).replace("\\", "/"),
                   "image_path": os.path.relpath(p, ROOT).replace("\\", "/"),
                   "image_size": [W, H], "n_cues": m, "nctrl": args.nctrl,
                   "seed": args.seed, "rng": "default_rng([seed, crc32(image_id)])",
                   "dilate_px": args.dilate,
                   "cues": [{"index": k, "cue": cues[k], "category": cats[k],
                             "area_frac": float(masks[k].sum() / npx)} for k in range(m)],
                   "files": files},
                  open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        n_img += 1
        el = time.time() - t0
        print(f"[{ii+1}/{len(ids)}] {iid[:28]:28s} m={m} 文件{len(files)} "
              f"(新{i_new}) 累计新{n_new}/跳{n_skip} {el/60:.1f}min", flush=True)

    el = time.time() - t0
    print(f"\n完成:{n_img} 张图,新写 {n_new} 个 PNG,跳过已存在 {n_skip} 个,"
          f"用时 {el/60:.2f} min ({el/max(n_new,1):.2f}s/新文件)", flush=True)


if __name__ == "__main__":
    main()
