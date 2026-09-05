"""【实验性 · 可整体删除】词表(vocab)口径修复缓存的 GeoRanker 打分 —— run_georanker_inpaint.py 的薄包装。

**为什么需要它**:run_georanker_inpaint.py 的线索元数据全部取自 sweep 的 `per_cue`
(`r["per_cue"][k]["cue"]` / `["category"]` / `["mpl"]`、`r["n_cues"]`、`r["mpl_all"]`)。
inpaint_cache_vocab 来自 cue_extract/results_vocab,**线索表与 sweep 完全不同**
(10 张图逐一核对:sweep 3 条 vs 词表 7 条、4 vs 6、3 vs 1、1 vs 2 ……,首条线索名也全不一样)。
直接跑它会:

  1. `cue_names` / `cue_categories` / `n_cues` 用 sweep 的线索表 → **静默张冠李戴**;
  2. `gray_mpl` 写成**另一条线索**的灰块 mPL(词表线索数 ≤ sweep 时);
  3. 词表线索数 > sweep 时,`if 0 <= k < len(r["per_cue"])` 让 cue_names 静默变空/变短,
     不报错;
  4. 默认 `--out` 仍是 georanker_inpaint_results.json → **与正在跑的主作业撞车**
     (同一 image_id 的 variants 会被混进同一条记录,spec 名还一样)。

本包装复用 run_georanker_inpaint 的缓存解析(parse_stem / list_specs / PART_OF)与打分/几何,
只改三件事:
  * 线索名/类别/面积一律读 `<cache>/<image_id>/manifest.json` 的 `cues[]`;
  * 不写 `gray_mpl` / `mpl_all_gray`(该口径没有灰块基线),`n_cues` 用 manifest 的;
  * 输出默认 georanker_inpaint_vocab_results.json / ..._vocab_control_results.json。
sweep 仍被读取,但**只用来取 posterior**(仪器确定性:同一张原图的后验不必重打)以及
true_label / country_hit / km_error 这些与线索表无关的图级字段。

下游分析:
  python -m belief_elicit.inpaint_report --results belief_elicit/georanker_inpaint_vocab_results.json \\
         --controls belief_elicit/georanker_inpaint_vocab_control_results.json \\
         --cues-from manifest --cache belief_elicit/inpaint_cache_vocab

运行(GPU 环境,注意 offline 变量):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint_vocab --part main
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

from belief_elicit.georanker_belief import score_labels
from belief_elicit.run_georanker_check import build_geometry, mpl
from belief_elicit.run_georanker_inpaint import (BATCH_SIZE, MERGE_KM, PART_OF,
                                                 VARIANT, list_specs)

HERE = os.path.dirname(os.path.abspath(__file__))


def read_cue_table(dirpath):
    """manifest.json → [{cue, category, area_frac}, ...];缺失/损坏返回 None。"""
    p = os.path.join(dirpath, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] manifest 解析失败 {p}: {e}", flush=True)
        return None
    cs = d.get("cues")
    if not isinstance(cs, list) or not cs:
        print(f"  [warn] manifest 里没有 cues[]:{p}", flush=True)
        return None
    out = []
    for i, c in enumerate(cs):
        if not isinstance(c, dict) or not c.get("cue"):
            print(f"  [warn] manifest cues[{i}] 形态不对:{p}", flush=True)
            return None
        out.append({"cue": c["cue"], "category": c.get("category") or "unknown",
                    "area_frac": c.get("area_frac")})
    n = d.get("n_cues")
    if isinstance(n, int) and n != len(out):
        print(f"  [warn] manifest n_cues={n} 与 cues[] 长度 {len(out)} 不一致:{p}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "inpaint_cache_vocab"))
    ap.add_argument("--part", choices=["main", "control", "all"], default="main")
    ap.add_argument("--ops", choices=["inpaint", "gray", "both"], default="both",
                    help="只对**对照**生效:inpaint = 只打 c*;gray = 只打 cg*;both = 两者")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sweep", default=os.path.join(HERE, "georanker_sweep_results.json"),
                    help="只用于取 posterior 与图级元数据,线索表一律来自 manifest")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        HERE, "georanker_inpaint_vocab_control_results.json" if args.part == "control"
        else "georanker_inpaint_vocab_results.json")
    if os.path.abspath(out_path) in (
            os.path.abspath(os.path.join(HERE, "georanker_inpaint_results.json")),
            os.path.abspath(os.path.join(HERE, "georanker_inpaint_control_results.json"))):
        print("[fatal] --out 指向了 SAM3 口径的主结果文件;词表口径必须写到单独的文件"
              "(两者 spec 名相同、线索表不同,混在一起就废了)")
        return 1

    sweep = {r["image_id"]: r for r in json.load(open(args.sweep, encoding="utf-8"))}
    try:
        order = sorted(d for d in os.listdir(args.cache)
                       if os.path.isdir(os.path.join(args.cache, d)))
    except OSError as e:
        print(f"[fatal] 读不了缓存目录 {args.cache}: {e}")
        return 1
    if args.ids:
        order = [i for i in order if any(i.startswith(p) for p in args.ids)]
    if not order:
        print(f"缓存目录里没有可用的图:{args.cache}")
        return 1

    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)

    # ---- 断点续跑:spec 级 ----
    done = {}
    if os.path.exists(out_path):
        try:
            for r in json.load(open(out_path, encoding="utf-8")):
                done[r["image_id"]] = r
        except Exception as e:
            print(f"[warn] 旧结果不可读({e}),从头开始", flush=True)
            done = {}
    done_specs = {i: {v["spec"] for v in r.get("variants", [])} for i, r in done.items()}
    if done:
        print(f"resume: 已有 {len(done)} 张 / "
              f"{sum(len(s) for s in done_specs.values())} 个 spec", flush=True)

    def save():
        recs = [done[i] for i in order if i in done]
        recs += [r for i, r in done.items() if i not in order]
        tmp = out_path + ".tmp"
        json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)

    # ---- 盘点待办 ----
    todo, cue_tables, n_skip = [], {}, 0
    for iid in order:
        d = os.path.join(args.cache, iid)
        cues = read_cue_table(d)
        if cues is None:                       # 没 manifest = 缓存可能还在生成中
            n_skip += 1
            continue
        if iid not in sweep:                   # 没 posterior 就算不出 mPL
            print(f"  [warn] {iid[:16]} 不在 sweep 里(没有 posterior),跳过", flush=True)
            n_skip += 1
            continue
        cue_tables[iid] = cues
        for it in list_specs(d, args.part, args.ops):
            if it["spec"] in done_specs.get(iid, set()):
                continue
            bad = [k for k in (it["cues"] or []) if not (0 <= k < len(cues))]
            if bad:
                print(f"  [warn] {iid[:16]} {it['spec']}: 线索索引 {bad} 超出 manifest "
                      f"的 {len(cues)} 条,跳过", flush=True)
                continue
            todo.append((iid, d, it))
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    if n_skip:
        print(f"[warn] {n_skip} 张图无 manifest / 无 posterior,本次跳过", flush=True)
    ops_note = "" if args.ops == "both" else f", ops={args.ops}"
    print(f"待打分 {total} 个变体(part={args.part}{ops_note}, cache={args.cache}) "
          f"→ {out_path}", flush=True)
    if total == 0:
        print("done (无待办)", flush=True)
        return 0

    t0, nsc, cur_iid = time.time(), 0, None
    for iid, d, it in todo:
        r = sweep[iid]
        cues_meta = cue_tables[iid]
        if iid not in done:
            done[iid] = {
                "image_id": iid, "true_label": r["true_label"],
                "country_hit": r["country_hit"], "km_error": r["km_error"],
                "n_cues": len(cues_meta), "merge_km": MERGE_KM, "variant": VARIANT,
                "cue_source": "manifest", "cache": os.path.abspath(args.cache),
                "cue_names": [c["cue"] for c in cues_meta],
                "cue_categories": [c["category"] for c in cues_meta],
                "cue_area_fracs": [c["area_frac"] for c in cues_meta],
                # 词表口径没有灰块基线:不写 mpl_all_gray / gray_mpl
                "variants": [],
            }
            done_specs.setdefault(iid, set())
        if iid != cur_iid:
            cur_iid = iid
            print(f"\n#### {r['true_label'].split(',')[0][:20]} ({iid[:12]}) "
                  f"{len(cues_meta)} vocab cues ####", flush=True)

        img = Image.open(os.path.join(d, it["file"])).convert("RGB")
        ts = time.time()
        prior, _ = score_labels(img, gv, variant=VARIANT, batch_size=BATCH_SIZE)
        secs = time.time() - ts
        val = mpl(prior, r["posterior"], rep, clusters, dist)

        ks = it["cues"] or []
        rec = {"spec": it["spec"], "file": it["file"], "kind": it["kind"],
               "op": it["op"], "cues": ks,
               "cue_names": [cues_meta[k]["cue"] for k in ks],
               "cue_categories": [cues_meta[k]["category"] for k in ks],
               "mpl": val, "seconds": secs, "prior": prior}
        if it["coverage"] is not None:
            rec["coverage"] = it["coverage"]
        if it["placement"] is not None:
            rec["placement"] = it["placement"]
        if PART_OF[it["kind"]] == "control":
            rec["overlap_frac"] = it["overlap_frac"]

        done[iid]["variants"].append(rec)
        done_specs[iid].add(it["spec"])
        save()

        nsc += 1
        el = time.time() - t0
        print(f"[{nsc}/{total}] {it['spec']:10s} {it['kind']:12s} mPL={val:.4f} "
              f"{secs:.0f}s ({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)

    save()
    print(f"\ndone → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
