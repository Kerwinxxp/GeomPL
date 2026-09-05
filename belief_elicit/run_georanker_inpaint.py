"""LaMa 修复(inpaint)版逐线索 mPL:对预计算图像变体批量打分。

灰块遮蔽会引入"篡改伪影"(等面积对照实验:涂灰一块无关区域也能大幅改变后验)。
本脚本改用 **预先修复好的图像变体**(由 inpaint 缓存包生成)重跑同一口径:
  变体图 → GeoRanker 变体 B / gallery_v2 → prior;posterior 直接复用 sweep(仪器确定性,
  原图零重复打分)→ 2km 几何 mPL。

缓存契约(belief_elicit/inpaint_cache/<image_id>/):
  s<k>.png       单线索 k 被删
  p<k>-<l>.png   线索对 (k,l) 被删
  all.png        全部线索被删
  c<k>-<j>.png   线索 k 的等面积对照放置 j(修复填补)
  cg<k>-<j>.png  同一放置的灰块填补版(修复 vs 灰块的直接对照)
  manifest.json  逐文件元数据(spec / cue 索引 / 对照掩码 RLE / overlap_frac / coverage)
manifest 缺失或不全时按文件名回退解析;磁盘上没有的文件跳过(允许缓存包仍在跑)。

输出:逐变体记录 **完整 prior 分布**({label: prob},138 项)——分布必须落盘,
下游 Shapley / 分布层面分析靠它,不能只存标量。每次打分后即写盘;重启按 **spec 级**
断点续跑(同一张图已打过的 spec 不重复打)。

运行(注意 offline 环境变量,避免 HF 联网卡住):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint --part main
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image

from belief_elicit.geometry import build_geometry, mpl
from belief_elicit.georanker_belief import score_labels

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE_KM = 2.0
VARIANT = "B"
BATCH_SIZE = 4

RE_ALL = re.compile(r"^all$")
RE_S = re.compile(r"^s(\d+)$")
RE_P = re.compile(r"^p(\d+)-(\d+)$")
RE_C = re.compile(r"^c(\d+)-(\d+)$")
RE_CG = re.compile(r"^cg(\d+)-(\d+)$")


# ---------------- 缓存清单解析 ----------------

def parse_stem(stem):
    """文件名 → (kind, cue 索引列表, 放置序号)。无法识别返回 None。

    kind ∈ {all, single, pair, ctrl_inpaint, ctrl_gray};前三者属 main,后两者属 control。
    """
    m = RE_ALL.match(stem)
    if m:
        return "all", None, None
    m = RE_S.match(stem)
    if m:
        return "single", [int(m.group(1))], None
    m = RE_P.match(stem)
    if m:
        return "pair", [int(m.group(1)), int(m.group(2))], None
    m = RE_CG.match(stem)                      # 必须先于 RE_C 试(cg 也匹配 c 前缀)
    if m:
        return "ctrl_gray", [int(m.group(1))], int(m.group(2))
    m = RE_C.match(stem)
    if m:
        return "ctrl_inpaint", [int(m.group(1))], int(m.group(2))
    return None


PART_OF = {"all": "main", "single": "main", "pair": "main",
           "ctrl_inpaint": "control", "ctrl_gray": "control"}


def load_manifest(dirpath):
    """读 manifest.json,归一化成 {filename: entry dict}。缺失/损坏 → {}。"""
    p = os.path.join(dirpath, "manifest.json")
    if not os.path.exists(p):
        return {}
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] manifest 解析失败 {p}: {e}", flush=True)
        return {}
    entries = None
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("files", "variants", "entries", "items"):
            if isinstance(data.get(key), list):
                entries = data[key]
                break
            if isinstance(data.get(key), dict):       # {"s0.png": {...}} 形态
                entries = [dict(v, file=k) for k, v in data[key].items()]
                break
        if entries is None:                            # 顶层直接按文件名键
            cand = [(k, v) for k, v in data.items() if isinstance(v, dict) and "." in k]
            entries = [dict(v, file=k) for k, v in cand]
    out = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        fn = None
        for key in ("file", "filename", "name", "path", "png"):
            if isinstance(e.get(key), str):
                fn = os.path.basename(e[key])
                break
        if fn is None and isinstance(e.get("spec"), str):
            fn = e["spec"] + ".png"
        if fn:
            out[fn] = e
    return out


def pick(entry, keys, default=None):
    for k in keys:
        if entry.get(k) is not None:
            return entry[k]
    return default


def list_specs(dirpath, part, ops="both"):
    """列出该图缓存目录中属于 part 的变体,按稳定顺序返回条目列表。

    ops ∈ {inpaint, gray, both}:只过滤 **对照**(c*/cg*)。对照的 op 优先取 manifest 的
    `op` 字段,缺失时按文件名前缀回退(cg → gray,c → inpaint)。main 部分(s*/p*/all)
    不受 ops 影响。
    """
    manifest = load_manifest(dirpath)
    try:
        files = sorted(os.listdir(dirpath))
    except OSError:
        return []
    items = []
    for fn in files:
        if not fn.lower().endswith(".png"):
            continue
        stem = os.path.splitext(fn)[0]
        parsed = parse_stem(stem)
        if parsed is None:
            print(f"  [warn] 无法识别的缓存文件名,跳过: {fn}", flush=True)
            continue
        kind, cues, place = parsed
        if part != "all" and PART_OF[kind] != part:
            continue
        e = manifest.get(fn, {})
        cues_m = pick(e, ["cues", "cue_indices", "cue_idx", "cue_ids", "k"])
        if isinstance(cues_m, int):
            cues_m = [cues_m]
        if isinstance(cues_m, list) and cues_m:
            cues = [int(c) for c in cues_m]
        # manifest 的 "spec" 是**类别**(single/pair/all/control),多个文件共用 →
        # 唯一 id 只能用文件名 stem(断点续跑按它去重)。
        op = pick(e, ["op"])
        if PART_OF[kind] == "control" and op in ("gray", "inpaint"):
            kind = "ctrl_gray" if op == "gray" else "ctrl_inpaint"
        if (PART_OF[kind] == "control" and ops != "both"
                and ("gray" if kind == "ctrl_gray" else "inpaint") != ops):
            continue
        items.append({
            "spec": stem,
            "file": fn,
            "kind": kind,
            "spec_kind": pick(e, ["spec"]),
            "op": op or ("gray" if kind == "ctrl_gray" else "inpaint"),
            "cues": cues,
            "placement": pick(e, ["placement", "j", "ctrl_idx"], place),
            "coverage": pick(e, ["coverage", "cov", "area_frac"]),
            "overlap_frac": pick(e, ["overlap_frac", "overlap"]),
        })
    order = {"single": 0, "pair": 1, "all": 2, "ctrl_inpaint": 3, "ctrl_gray": 3}

    def key(it):
        c = it["cues"] or []
        return (order[it["kind"]], c, it["placement"] if it["placement"] is not None else -1,
                0 if it["kind"] != "ctrl_gray" else 1, it["spec"])

    return sorted(items, key=key)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "inpaint_cache"),
                    help="修复缓存根目录(内含 <image_id>/ 子目录)")
    ap.add_argument("--part", choices=["main", "control", "all"], default="main",
                    help="main = s*/p*/all;control = c*/cg*;all = 两者")
    ap.add_argument("--ops", choices=["inpaint", "gray", "both"], default="both",
                    help="只对**对照**生效:inpaint = 只打 c*;gray = 只打 cg*;"
                         "both(默认)= 两者都打。对照打分量减半时用它")
    ap.add_argument("--ids", nargs="*", default=None, help="按 image_id 前缀筛选")
    ap.add_argument("--out", default=None, help="结果 JSON(默认按 --part 选择)")
    ap.add_argument("--limit", type=int, default=0, help="本次最多打分次数(0 = 不限,冒烟用)")
    ap.add_argument("--sweep", default=os.path.join(HERE, "georanker_sweep_results.json"),
                    help="sweep 结果:提供 posterior 与逐线索元数据")
    ap.add_argument("--no-manifest-ok", action="store_true",
                    help="缺 manifest.json 的图也照跑(默认跳过:该图可能仍在生成中)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        HERE, "georanker_inpaint_control_results.json" if args.part == "control"
        else "georanker_inpaint_results.json")

    sweep = json.load(open(args.sweep, encoding="utf-8"))
    order = [r["image_id"] for r in sweep]
    sw = {r["image_id"]: r for r in sweep}
    if args.ids:
        order = [i for i in order if any(i.startswith(p) for p in args.ids)]
        if not order:
            print(f"没有匹配 --ids {args.ids} 的图", flush=True)
            return

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
        recs += [r for i, r in done.items() if i not in order]     # 保留其它图的旧结果
        tmp = out_path + ".tmp"
        json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)

    # ---- 盘点待办 ----
    todo = []
    n_nomani = 0
    for iid in order:
        d = os.path.join(args.cache, iid)
        if not os.path.isdir(d):
            continue
        # precompute 是**最后**才写 manifest.json 的:没有 manifest = 该图还在生成中,
        # 此时 PNG 可能只写了一半 → 跳过(缓存包跑完后重跑本脚本即可补上)。
        if not args.no_manifest_ok and not os.path.exists(os.path.join(d, "manifest.json")):
            n_nomani += 1
            continue
        for it in list_specs(d, args.part, args.ops):
            if it["spec"] in done_specs.get(iid, set()):
                continue
            todo.append((iid, d, it))
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    if n_nomani:
        print(f"[warn] {n_nomani} 张图无 manifest.json(缓存仍在生成?),本次跳过", flush=True)
    ops_note = "" if args.ops == "both" else f", ops={args.ops}"
    print(f"待打分 {total} 个变体(part={args.part}{ops_note}, cache={args.cache})",
          flush=True)
    if total == 0:
        print("done (无待办)", flush=True)
        return

    t0, nsc = time.time(), 0
    cur_iid = None
    for iid, d, it in todo:
        r = sw[iid]
        if iid not in done:
            done[iid] = {
                "image_id": iid, "true_label": r["true_label"],
                "country_hit": r["country_hit"], "km_error": r["km_error"],
                "n_cues": r["n_cues"], "merge_km": MERGE_KM, "variant": VARIANT,
                "cue_names": [pc["cue"] for pc in r["per_cue"]],
                "cue_categories": [pc["category"] for pc in r["per_cue"]],
                "mpl_all_gray": r["mpl_all"],
                "variants": [],
            }
            done_specs.setdefault(iid, set())
        if iid != cur_iid:
            cur_iid = iid
            print(f"\n#### {r['true_label'].split(',')[0][:20]} ({iid[:12]}) "
                  f"{r['n_cues']} cues ####", flush=True)

        img = Image.open(os.path.join(d, it["file"])).convert("RGB")
        ts = time.time()
        prior, _ = score_labels(img, gv, variant=VARIANT, batch_size=BATCH_SIZE)
        secs = time.time() - ts
        val = mpl(prior, r["posterior"], rep, clusters, dist)

        cues = it["cues"] or []
        rec = {
            "spec": it["spec"], "file": it["file"], "kind": it["kind"],
            "op": it["op"],
            "cues": cues,
            "cue_names": [r["per_cue"][k]["cue"] for k in cues
                          if 0 <= k < len(r["per_cue"])],
            "mpl": val,
            "seconds": secs,
            "prior": prior,                      # 完整分布,必须落盘
        }
        if it["coverage"] is not None:
            rec["coverage"] = it["coverage"]
        if it["placement"] is not None:
            rec["placement"] = it["placement"]
        if PART_OF[it["kind"]] == "control":
            rec["overlap_frac"] = it["overlap_frac"]
        # 灰块基线:单线索时给出 sweep 的灰块 mPL,便于直接对照
        if it["kind"] == "single" and 0 <= cues[0] < len(r["per_cue"]):
            rec["gray_mpl"] = r["per_cue"][cues[0]]["mpl"]

        done[iid]["variants"].append(rec)
        done_specs[iid].add(it["spec"])
        save()                                   # 每次打分后即写盘(spec 级续跑才有意义)

        nsc += 1
        el = time.time() - t0
        ref = f" (gray={rec['gray_mpl']:.3f})" if "gray_mpl" in rec else ""
        print(f"[{nsc}/{total}] {it['spec']:10s} {it['kind']:12s} "
              f"mPL={val:.4f}{ref} {secs:.0f}s "
              f"({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)

    save()
    print(f"\ndone → {out_path}", flush=True)


if __name__ == "__main__":
    main()
