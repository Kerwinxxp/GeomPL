"""【实验性 · 可整体删除】等面积对照(GeoRanker 版):测"遮蔽伪影地板"。

对每条真线索:把它的 SAM3 掩码平移到随机非线索位置(形状/面积不变,避开全部线索并集),
同管线(变体 B / gallery_v2 / 2km 几何)打分 → 对照 mPL。
若 对照 ≈ 真线索,则 mPL 量的是"遮了一块"的伪影而非该线索的信息 —— 这是当年在
GPT-4o 仪器上实测到的问题,本脚本在 GeoRanker 上闭环验证。

选图:按 v(N) 分位数均匀取 10 张 + 强制含 NY/Bled/Cuba 案例图;每线索 nctrl=3 个放置。
增量保存 + 断点续跑。真线索 mPL 直接复用 sweep(同后验,零额外打分)。
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_control
"""
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

from belief_elicit.georanker_belief import score_labels
from belief_elicit.run_georanker_check import build_geometry, mpl
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask

SWEEP = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")
OUT = os.path.join(os.path.dirname(__file__), "georanker_control_results.json")
MERGE_KM, SEED, NPICK = 2.0, 42, 10
FORCE = ["158307292", "754780171", "370717727"]      # NY / Bled / Cuba 案例图


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


def cue_masks_of(iid):
    rec = json.load(open(os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json"),
                         encoding="utf-8"))
    W, H = rec["image_size"]
    cues, masks = [], []
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
        cues.append(c["cue"]); masks.append(u)
    return cues, masks, (W, H)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="跑全部 maskable 图(否则分位抽样)")
    ap.add_argument("--nctrl", type=int, default=3, help="每线索随机放置数")
    args = ap.parse_args()
    NCTRL = args.nctrl
    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    sweep = {r["image_id"]: r for r in json.load(open(SWEEP, encoding="utf-8"))}

    # ---- 选图:按 v(N) 排序取分位 + 强制案例图 ----
    cand = sorted((r for r in sweep.values() if r["n_cues"] >= 1),
                  key=lambda r: r["mpl_all"])
    if args.all:
        picks = [r["image_id"] for r in cand]
        print(f"全量模式:{len(picks)} 张", flush=True)
    else:
        idx = np.linspace(0, len(cand) - 1, NPICK).round().astype(int)
        picks = [cand[i]["image_id"] for i in idx]
        for pref in FORCE:
            iid = next((k for k in sweep if k.startswith(pref)), None)
            if iid and iid not in picks:
                picks.append(iid)
        print(f"选图 {len(picks)} 张(按 v(N) 分位 + 案例图)", flush=True)

    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["image_id"]] = r
        print(f"resume: 已有 {len(done)} 张", flush=True)

    rng = np.random.default_rng(SEED)
    t0, nsc = time.time(), 0
    total = sum(sweep[i]["n_cues"] * NCTRL for i in picks if i not in done)
    for iid in picks:
        if iid in done:
            continue
        r = sweep[iid]
        cues, masks, (W, H) = cue_masks_of(iid)
        assert cues == [pc["cue"] for pc in r["per_cue"]], f"cue 顺序不一致 {iid}"
        union = np.zeros((H, W), bool)
        for m_ in masks:
            union |= m_
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")
        post = r["posterior"]

        rec_out = {"image_id": iid, "true_label": r["true_label"],
                   "country_hit": r["country_hit"], "n_cues": r["n_cues"],
                   "vN": r["mpl_all"], "cues": []}
        for k, (name, m_) in enumerate(zip(cues, masks)):
            ctrls = []
            for j in range(NCTRL):
                t, ovf = sample_control(m_, union, rng)
                if t is None:
                    continue
                pri, _ = score_labels(mask_solid_from_masks(img, [t]), gv, variant="B",
                                      batch_size=4)
                ctrls.append({"mpl": mpl(pri, post, rep, clusters, dist),
                              "overlap_frac": ovf,
                              "cov": float(t.sum() / (W * H))})
                nsc += 1
                el = time.time() - t0
                print(f"[{nsc}/{total}] {r['true_label'].split(',')[0][:12]:12s} "
                      f"cue{k+1} ctrl{j+1} mPL={ctrls[-1]['mpl']:.3f} "
                      f"(real={r['per_cue'][k]['mpl']:.3f}) "
                      f"({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)
            rec_out["cues"].append({"cue": name, "category": r["per_cue"][k]["category"],
                                    "area_frac": float(m_.sum() / (W * H)),
                                    "real_mpl": r["per_cue"][k]["mpl"],
                                    "controls": ctrls})
        done[iid] = rec_out
        json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
