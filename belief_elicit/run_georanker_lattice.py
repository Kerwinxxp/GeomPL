"""【实验性 · 可整体删除】补全 3-5 线索图的全子集格(为完整 Shapley)。

sweep 已有:原图后验、全部单条先验、全遮先验(均含完整分布)。
本脚本只补中间尺寸子集(2 <= |S| <= m-1),完成后每张多线索图具备全部 2^m-1 个 v(S)。
与 sweep 严格对齐:同 gallery/几何(2km)、同掩码顺序(断言校验 cue 名逐一相同)、
复用 sweep 存储的后验(模型确定性保证一致)。增量保存 + 断点续跑。
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_lattice
"""
import glob
import itertools
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
OUT = os.path.join(os.path.dirname(__file__), "georanker_lattice_results.json")
MERGE_KM = 2.0


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
    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    sweep = {r["image_id"]: r for r in json.load(open(SWEEP, encoding="utf-8"))}

    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["image_id"]] = r
        print(f"resume: 已有 {len(done)} 张(可能部分完成)", flush=True)

    targets = [r for r in sweep.values() if r["n_cues"] >= 3]
    total_scorings = sum(2 ** r["n_cues"] - 2 - r["n_cues"] for r in targets)
    print(f"待补 {len(targets)} 张(m>=3),共 {total_scorings} 次打分", flush=True)

    t0 = time.time()
    n_done_scorings = 0
    for r in sorted(targets, key=lambda x: x["n_cues"]):
        iid = r["image_id"]
        m = r["n_cues"]
        cues, masks, (W, H) = cue_masks_of(iid)
        assert [c for c in cues] == [pc["cue"] for pc in r["per_cue"]], \
            f"cue 顺序不一致: {iid}"
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")
        tl = r["true_label"]
        post = r["posterior"]                       # 复用 sweep 后验(确定性)

        rec_out = done.get(iid, {"image_id": iid, "true_label": tl, "n_cues": m,
                                 "cue_names": cues, "merge_km": MERGE_KM, "combos": []})
        have = {tuple(c["subset"]) for c in rec_out["combos"]}
        need = [S for size in range(2, m)
                for S in itertools.combinations(range(m), size)
                if S not in have]
        if not need:
            done[iid] = rec_out; continue

        for S in need:
            u = np.zeros((H, W), bool)
            for k in S:
                u |= masks[k]
            pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant="B",
                                  batch_size=4)
            rec_out["combos"].append({"subset": list(S),
                                      "cov": float(u.sum() / (W * H)),
                                      "p_true": pri.get(tl, 0.0),
                                      "mpl": mpl(pri, post, rep, clusters, dist),
                                      "prior": pri})
            done[iid] = rec_out
            json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            n_done_scorings += 1
            el = time.time() - t0
            eta = el / n_done_scorings * (total_scorings - n_done_scorings) / 60
            print(f"[{n_done_scorings}/{total_scorings}] {tl.split(',')[0][:14]:14s} "
                  f"S={list(S)} mPL={rec_out['combos'][-1]['mpl']:.3f} "
                  f"({el/60:.0f}min, 剩~{eta:.0f}min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
