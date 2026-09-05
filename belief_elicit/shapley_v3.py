"""**几何去重之后**的归因(shapley_v2 的替代口径)。

先用 belief_elicit/dedup_cues.py 把两两 IoU >= 0.9 的线索并成"合并玩家",再在合并后的
博弈上算精确 Shapley + SII。

无需重新打分(关键点):合并玩家 A = {a1..aj} 的成员掩码几乎逐像素相同,所以
    "遮住 S' 中每个合并玩家" 的图 ≈ "遮住 ⋃_{A∈S'} members(A)" 的图,
两者只差极少数边界像素(IoU=1.000 的组完全相同;IoU=0.903/0.998 的组差几个像素)。
灰块 2^m 全格已有,故直接令
    v'(S') := v(⋃_{A∈S'} members(A))
从已有数据里读出来。**这是本脚本唯一的近似**,报告里必须写明。

修复(inpaint)口径只有 单条 s<k> / 成对 p<k>-<l> / 全遮 all,合并之后需要的
"⋃members" 子集不一定在其中:
  · 逐玩家的 v'_inp({A}) 只要 ⋃members(A) 在里面就能读(1 个成员 -> s;2 个 -> p;整图 -> all);
  · 全图二阶锚定 φ 还需要全部合并成对,缺任一个就整图记为 inpaint-incomputable 并计数。

产物:belief_elicit/shapley_v3_results.json(schema 与 shapley_v2_results.json 一致,
      cues[] 额外带 members / member_indices / 修复口径字段),下游脚本可直接指向它。
运行:python -m belief_elicit.shapley_v3
"""
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from belief_elicit.dedup_cues import OUT as GROUPS_DEFAULT
from belief_elicit.order2_shapley import order2_shapley
from belief_elicit.shapley_v2 import build_v, shapley, sii

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "georanker_sweep_results.json")
LATTICE = os.path.join(HERE, "georanker_lattice_results.json")
INPAINT = os.path.join(HERE, "georanker_inpaint_results.json")
INPAINT_CTRL = os.path.join(HERE, "georanker_inpaint_control_results.json")
GRAY_CTRL = os.path.join(HERE, "georanker_control_results.json")
OUT = os.path.join(HERE, "shapley_v3_results.json")


# ---------------- 合并博弈 ----------------

def merged_v(v, groups):
    """原始 v(键 = 原线索下标 frozenset)→ 合并博弈 v'(键 = 合并玩家下标 frozenset)。"""
    M = len(groups)
    vm = {}
    for size in range(M + 1):
        for S in itertools.combinations(range(M), size):
            orig = frozenset(k for gi in S for k in groups[gi]["members"])
            if orig not in v:
                return None
            vm[frozenset(S)] = v[orig]
    return vm


def inpaint_lookup(rec):
    """inpaint 结果记录 → {frozenset(原线索下标): mpl};只含 单条 / 成对 / 全遮。"""
    out = {frozenset(): 0.0}
    m = rec["n_cues"]
    for v in rec.get("variants", []):
        sp = v["spec"]
        if sp.startswith("s") and sp[1:].isdigit():
            out[frozenset([int(sp[1:])])] = v["mpl"]
        elif sp.startswith("p") and "-" in sp:
            a, b = sp[1:].split("-")
            out[frozenset([int(a), int(b)])] = v["mpl"]
        elif sp == "all":
            out[frozenset(range(m))] = v["mpl"]
    return out


def merged_inpaint(rec, groups):
    """合并后二阶锚定 φ 所需的 v'。缺任一子集返回 (None, 缺的子集列表)。"""
    lut = inpaint_lookup(rec)
    M = len(groups)
    need, miss = {}, []
    for size in (1, 2):
        for S in itertools.combinations(range(M), size):
            orig = frozenset(k for gi in S for k in groups[gi]["members"])
            if orig in lut:
                need[frozenset(S)] = lut[orig]
            else:
                miss.append(sorted(orig))
    allset = frozenset(range(M))
    orig_all = frozenset(k for g in groups for k in g["members"])
    if orig_all in lut:
        need[allset] = lut[orig_all]
    else:
        miss.append(sorted(orig_all))
    return (None, miss) if miss else (need, [])


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="几何去重后的 Shapley/SII 归因")
    ap.add_argument("--groups", default=GROUPS_DEFAULT)
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--lattice", default=LATTICE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    G = json.load(open(a.groups, encoding="utf-8"))["images"]
    sweep = json.load(open(a.sweep, encoding="utf-8"))
    lattice = ({r["image_id"]: r for r in json.load(open(a.lattice, encoding="utf-8"))}
               if os.path.exists(a.lattice) else {})
    inpaint = {r["image_id"]: r for r in json.load(open(INPAINT, encoding="utf-8"))} \
        if os.path.exists(INPAINT) else {}
    ictrl = {r["image_id"]: r for r in json.load(open(INPAINT_CTRL, encoding="utf-8"))} \
        if os.path.exists(INPAINT_CTRL) else {}
    gctrl = {r["image_id"]: r for r in json.load(open(GRAY_CTRL, encoding="utf-8"))} \
        if os.path.exists(GRAY_CTRL) else {}

    rows, skipped, n_inpaint_ok, n_inpaint_bad, bad_ids = [], 0, 0, 0, []
    for r in sweep:
        m = r["n_cues"]
        if m < 1:
            continue
        iid = r["image_id"]
        d = G.get(iid)
        if d is None:
            skipped += 1; continue
        assert d["n_cues"] == m, f"n_cues 不一致 {iid}"
        groups = d["groups"]
        M = len(groups)
        v, ok = build_v(r, lattice)
        if not ok:
            skipped += 1; continue
        vm = merged_v(v, groups)
        assert vm is not None, f"灰块格不完整,合并子集缺失 {iid}"

        phis = shapley(vm, M)
        vN = vm[frozenset(range(M))]
        if M >= 2:
            assert abs(sum(phis) - vN) < 1e-9, f"效率失败 {iid}"
        inter_sii = {f"{k},{l}": sii(vm, M, k, l)
                     for k, l in itertools.combinations(range(M), 2)}
        inter_empty = {f"{k},{l}": vm[frozenset([k, l])] - vm[frozenset([k])] - vm[frozenset([l])]
                       for k, l in itertools.combinations(range(M), 2)}

        # ---- 修复口径:合并后的二阶锚定 φ(可算时)----
        inp_phi, inp_single, inp_info = None, None, {"computable": False, "reason": "no inpaint data"}
        irec = inpaint.get(iid)
        if irec is not None:
            # 逐玩家的 v'_inp({A}) 独立于全图二阶 φ 是否可算(能读到就读)
            lut = inpaint_lookup(irec)
            inp_single = [lut.get(frozenset(g["members"])) for g in groups]
            need, miss = merged_inpaint(irec, groups)
            if need is None:
                n_inpaint_bad += 1; bad_ids.append(iid)
                inp_info = {"computable": False, "reason": "merged subset not scored under inpaint",
                            "missing_subsets": miss}
            else:
                singles = [need[frozenset([k])] for k in range(M)]
                pairs = {(k, l): need[frozenset([k, l])]
                         for k, l in itertools.combinations(range(M), 2)}
                o2 = order2_shapley(singles, pairs, need[frozenset(range(M))])
                inp_phi = o2["phi"]
                inp_info = {"computable": True,
                            "residual_share": o2["residual_share"],
                            "vN": need[frozenset(range(M))]}
                n_inpaint_ok += 1

        # ---- 可分辨性(合并玩家:real = v'({A});对照 = 各成员自己的对照最大值)----
        gc = gctrl.get(iid)
        gmax_by_k = {}
        if gc:
            for k, c in enumerate(gc["cues"]):
                cm = [x["mpl"] for x in c["controls"]]
                gmax_by_k[k] = max(cm) if cm else None
        ic = ictrl.get(iid)
        imax_by_k = defaultdict(list)
        if ic:
            for vv in ic.get("variants", []):
                sp = vv["spec"]
                if sp.startswith("c") and "-" in sp and sp[1:].split("-")[0].isdigit():
                    imax_by_k[int(sp[1:].split("-")[0])].append(vv["mpl"])

        cues_out = []
        for gi, g in enumerate(groups):
            mem = g["members"]
            gm = [gmax_by_k.get(k) for k in mem]
            gm = [x for x in gm if x is not None]
            im = [x for k in mem for x in imax_by_k.get(k, [])]
            v_single = vm[frozenset([gi])]
            cues_out.append({
                "cue": g["name"], "category": g["category"],
                "v_single": v_single, "phi": phis[gi],
                "members": g["names"], "member_indices": mem,
                "n_members": len(mem),
                "categories": g["categories"],
                "ctrl_gray_max": (max(gm) if gm else None),
                "resolvable_gray": (v_single > max(gm)) if gm else None,
                "v_single_inpaint": (inp_single[gi] if inp_single else None),
                "phi_inpaint": (inp_phi[gi] if inp_phi else None),
                "ctrl_inpaint_max": (max(im) if im else None),
                "resolvable_inpaint": ((inp_single[gi] > max(im))
                                       if (inp_single and inp_single[gi] is not None and im)
                                       else None),
            })

        rows.append({"image_id": iid, "place": r["true_label"].split(",")[0],
                     "country_hit": r["country_hit"], "km_error": r["km_error"],
                     "n_cues": M, "n_cues_orig": m, "merged": d["affected"],
                     "vN": vN,
                     "cues": cues_out,
                     "sii": inter_sii, "empty_interaction": inter_empty,
                     "inpaint": inp_info})
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------- 打印 ----------------
    n_before = sum(x["n_cues_orig"] for x in rows)
    n_after = sum(x["n_cues"] for x in rows)
    aff = [x for x in rows if x["merged"]]
    print(f"=== 去重后归因:{len(rows)} 张图(跳过 {skipped})===")
    print(f"线索 {n_before} -> {n_after}(合并掉 {n_before-n_after});受影响的图 {len(aff)}")
    print(f"效率(m>=2):全部通过")
    print(f"修复口径二阶 φ:可算 {n_inpaint_ok} 张 | 不可算(合并子集未打分) {n_inpaint_bad} 张")
    if bad_ids:
        by_place = [x["place"] for x in rows if x["image_id"] in set(bad_ids)]
        print(f"  不可算的图: {by_place}")

    cphi, cv = defaultdict(list), defaultdict(list)
    for x in rows:
        for c in x["cues"]:
            cphi[c["category"] or "unknown"].append(c["phi"])
            cv[c["category"] or "unknown"].append(c["v_single"])
    print("\n逐类别(去重后;φ=Shapley, v=single-cue):")
    print(f"{'类别':24s} {'φ中位':>8s} {'v中位':>8s} {'n':>4s}")
    rank = sorted(cphi, key=lambda x: -np.median(cphi[x]))
    for k in rank:
        print(f"{k:24s} {np.median(cphi[k]):8.4f} {np.median(cv[k]):8.4f} {len(cphi[k]):4d}")
    print(f"  φ 排名: {[k[:14] for k in rank]}")

    sii_all = np.array([x for r_ in rows for x in r_["sii"].values()])
    if len(sii_all):
        print(f"\nSII({len(sii_all)} 对): 重叠<-.01 {int((sii_all<-0.01).sum())} | "
              f"~0 {int((np.abs(sii_all)<=0.01).sum())} | 备份>.01 {int((sii_all>0.01).sum())} "
              f"| 中位 {np.median(sii_all):+.3f}")
    rg = [c["resolvable_gray"] for x in rows for c in x["cues"] if c["resolvable_gray"] is not None]
    ri = [c["resolvable_inpaint"] for x in rows for c in x["cues"]
          if c["resolvable_inpaint"] is not None]
    if rg:
        print(f"可分辨(灰块,real>成员对照最大): {np.mean(rg)*100:.1f}% (n={len(rg)})")
    if ri:
        print(f"可分辨(修复,real>成员对照最大): {np.mean(ri)*100:.1f}% (n={len(ri)})")
    print("saved", a.out)


if __name__ == "__main__":
    main()
