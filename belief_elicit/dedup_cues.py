"""De-duplicate the GPT-4o cue list by **mask geometry** (a necessary clean-up before attribution).

The problem: GPT-4o gives cues different names but SAM 3 segments (nearly) the same pixels
for them. For example Seville has 3 cues with pairwise IoU = 1.000; in Bangkok "Thai text on
storefront signs" is the same region as "Fujifilm signage"; in Tinum "Mayan architectural
style" ~ "Ruined stone structures" (IoU 0.998). Duplicate cues (1) inflate m, (2) split one
region's credit across several phi, and (3) manufacture fake "overlap" interactions (I ~ -v).

Method: per image, compute pairwise IoU of the cue union masks and union-find them into
"merged players" at --iou (default 0.90). Containment is **reported but not merged**:
recall(A subset B) = |A and B| / |A| >= --recall (default 0.95) while IoU < the threshold.

Masks come from belief_elicit.cues.cue_masks_of (the single reader), and the cue order must
match per_cue in georanker_sweep_results.json (asserted, as in run_georanker_control.py).

Output: belief_elicit/cue_dedup_groups.json
Run: python -m belief_elicit.dedup_cues            # default --iou 0.9
     python -m belief_elicit.dedup_cues --iou 0.95
"""
import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from belief_elicit.cues import cue_masks_of
from belief_elicit.results import DEDUP_GROUPS as OUT, SAM3_DIR as SRC, SWEEP

HERE = os.path.dirname(os.path.abspath(__file__))

IOU_DEFAULT = 0.90
RECALL_DEFAULT = 0.95


# ---------------- geometry ----------------

def pairwise_geometry(masks):
    """-> (iou[m,m], recall[m,m]); recall[i,j] = |Mi and Mj| / |Mi| (how much of i lies in j)."""
    m = len(masks)
    flat = [mk.ravel() for mk in masks]
    area = np.array([float(f.sum()) for f in flat])
    iou = np.eye(m)
    rec = np.eye(m)
    for i, j in itertools.combinations(range(m), 2):
        inter = float(np.count_nonzero(flat[i] & flat[j]))
        union = area[i] + area[j] - inter
        iou[i, j] = iou[j, i] = inter / union if union > 0 else 0.0
        rec[i, j] = inter / area[i] if area[i] > 0 else 0.0
        rec[j, i] = inter / area[j] if area[j] > 0 else 0.0
    return iou, rec


def union_find_groups(iou, thr):
    """Union-find over pairs with IoU >= thr -> groups (list of lists) sorted by lowest member."""
    m = iou.shape[0]
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in itertools.combinations(range(m), 2):
        if iou[i, j] >= thr:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    buckets = {}
    for i in range(m):
        buckets.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in buckets.values()), key=lambda g: g[0])


def dedup_image(cues, cats, masks, iou_thr=IOU_DEFAULT, recall_thr=RECALL_DEFAULT):
    """De-duplicate one image -> dict(groups / index_map / containment / n_cues / n_merged / affected)."""
    m = len(cues)
    iou, rec = pairwise_geometry(masks)
    groups = union_find_groups(iou, iou_thr)
    index_map = [0] * m
    out_groups = []
    for gi, g in enumerate(groups):
        for k in g:
            index_map[k] = gi
        # quantify the approximation: the merged player is the members' union, so record
        # how many pixels the worst member differs from that union by
        u = np.zeros_like(masks[0])
        for k in g:
            u |= masks[k]
        ua = int(u.sum())
        diff = max(int((u ^ masks[k]).sum()) for k in g)
        out_groups.append({
            "members": g,
            "names": [cues[k] for k in g],
            "categories": [cats[k] for k in g],
            "name": " | ".join(cues[k] for k in g),
            "category": cats[g[0]],
            "pair_ious": [[a, b, float(iou[a, b])] for a, b in itertools.combinations(g, 2)],
            "areas": [float(masks[k].sum()) for k in g],
            "union_area_px": ua,
            "max_member_diff_px": diff,
            "max_member_diff_frac": (diff / ua if ua else 0.0),
        })
    containment = []
    for i, j in itertools.combinations(range(m), 2):
        if iou[i, j] >= iou_thr:
            continue
        for a, b in ((i, j), (j, i)):
            if rec[a, b] >= recall_thr:
                containment.append({"inner": a, "outer": b,
                                    "inner_name": cues[a], "outer_name": cues[b],
                                    "recall_inner_in_outer": float(rec[a, b]),
                                    "iou": float(iou[i, j])})
    return {"n_cues": m, "n_merged": len(groups),
            "affected": len(groups) < m,
            "groups": out_groups, "index_map": index_map,
            "containment": containment,
            "max_offdiag_iou": float(max((iou[a, b] for a, b in
                                          itertools.combinations(range(m), 2)), default=0.0))}


# ---------------- batch ----------------

def build_all(sweep_path=SWEEP, src=SRC, iou_thr=IOU_DEFAULT, recall_thr=RECALL_DEFAULT,
              verbose=True):
    sweep = json.load(open(sweep_path, encoding="utf-8"))
    images, missing = {}, []
    for r in sweep:
        iid, m = r["image_id"], r["n_cues"]
        if m < 1:
            continue
        got = cue_masks_of(src, iid)
        if got is None:
            missing.append(iid)
            continue
        cues, cats, masks, _ = got
        assert cues == [pc["cue"] for pc in r["per_cue"]], f"cue 顺序不一致 {iid}"
        assert len(cues) == m, f"cue 数不一致 {iid}: {len(cues)} vs {m}"
        d = dedup_image(cues, cats, masks, iou_thr, recall_thr)
        d["image_id"] = iid
        d["place"] = r["true_label"].split(",")[0]
        images[iid] = d
        if verbose and d["affected"]:
            print(f"  {d['place'][:18]:18s} m {m} -> {d['n_merged']}", flush=True)
    return {"meta": {"iou_threshold": iou_thr, "recall_threshold": recall_thr,
                     "src": os.path.relpath(src, ROOT).replace("\\", "/"),
                     "n_images": len(images), "missing_source_json": missing},
            "images": images}


def load_groups(path=OUT):
    """Read cue_dedup_groups.json back (for shapley_v3 / dedup_report / plot_dedup)."""
    return json.load(open(path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="de-duplicate cues by mask geometry")
    ap.add_argument("--iou", type=float, default=IOU_DEFAULT, help="merge threshold (union-find)")
    ap.add_argument("--recall", type=float, default=RECALL_DEFAULT,
                    help="containment reporting threshold (reported, never merged)")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    print(f"=== 掩码几何去重(IoU>={a.iou:.2f} 合并;包含 recall>={a.recall:.2f} 仅报告)===")
    data = build_all(a.sweep, a.src, a.iou, a.recall)
    json.dump(data, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    imgs = data["images"]
    aff = [d for d in imgs.values() if d["affected"]]
    n_cues_before = sum(d["n_cues"] for d in imgs.values())
    n_cues_after = sum(d["n_merged"] for d in imgs.values())
    merged_groups = [(d, g) for d in imgs.values() for g in d["groups"] if len(g["members"]) > 1]
    print(f"\n图 {len(imgs)} 张(源 JSON 缺失 {len(data['meta']['missing_source_json'])})")
    print(f"受影响的图: {len(aff)} | 合并组: {len(merged_groups)} | "
          f"线索 {n_cues_before} -> {n_cues_after}(删掉重复 {n_cues_before - n_cues_after})")

    print(f"\n--- 全部合并组({len(merged_groups)} 组)---")
    for d, g in sorted(merged_groups, key=lambda x: -max(i[2] for i in x[1]["pair_ious"])):
        ious = ", ".join(f"{a_}-{b_}:{v:.3f}" for a_, b_, v in g["pair_ious"])
        print(f"  [{d['place'][:16]:16s}] m={d['n_cues']} 组 {g['members']}  IoU {ious}"
              f"  | 成员 vs 并集最大差 {g['max_member_diff_px']} px "
              f"({g['max_member_diff_frac']*100:.2f}%)")
        for k, nm in zip(g["members"], g["names"]):
            print(f"        {k}: {nm}")

    cont = [(d, c) for d in imgs.values() for c in d["containment"]]
    print(f"\n--- 包含关系(recall>={a.recall:.2f} 且 IoU<{a.iou:.2f};**不合并**,仅报告)"
          f":{len(cont)} 对 ---")
    for d, c in sorted(cont, key=lambda x: -x[1]["recall_inner_in_outer"])[:25]:
        print(f"  [{d['place'][:16]:16s}] recall={c['recall_inner_in_outer']:.3f} "
              f"IoU={c['iou']:.3f}  '{c['inner_name'][:34]}' ⊂ '{c['outer_name'][:34]}'")
    if len(cont) > 25:
        print(f"  ...(另有 {len(cont)-25} 对)")

    from collections import Counter
    hist = Counter((d["n_cues"], d["n_merged"]) for d in imgs.values())
    print("\n--- m 变化直方图(before -> after: 图数)---")
    for (b, a2), n in sorted(hist.items()):
        flag = "  <-- 合并" if a2 < b else ""
        print(f"  {b} -> {a2}: {n}{flag}")
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
