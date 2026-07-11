"""扩到 50 张:选出所有有 >=1 可遮线索的图(优先线索多的),跑 combo2 消融,凑够 target 张。
仅编排,不改测量。用法：python -m clue_leak.run_50 [--target 50]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from clue_leak.run_combo2 import maskable_cues, POSTDIR

CUEDIR = os.path.join(ROOT, "cue_extract", "results")
OUTDIR = os.path.join(os.path.dirname(__file__), "combo2_results")


def eligible(max_m: int = 6):
    """返回 [(m, image_id)]:1<=可遮线索<=max_m 且有原图后验的图,按 m 降序。
    max_m 排除 m 过大的文字巨怪图(单张 2^m 爆炸 + 柱图不可读)。"""
    out = []
    for f in glob.glob(os.path.join(CUEDIR, "*.json")):
        rec = json.load(open(f, encoding="utf-8"))
        iid = rec["image_id"]
        if not os.path.exists(os.path.join(POSTDIR, iid + ".json")):
            continue
        m = len(maskable_cues(rec))
        if 1 <= m <= max_m:
            out.append((m, iid))
    out.sort(key=lambda x: -x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50)
    args = ap.parse_args()
    elig = eligible()
    done = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(OUTDIR, "*.json"))}
    print(f"{len(elig)} eligible images (>=1 maskable cue); {len(done)} already ablated")
    picked = [iid for _, iid in elig][: args.target]
    # 处理顺序按 m 升序:小图先出、快反馈(pick 已按 m 降序取够 target)
    m_of = {iid: m for m, iid in elig}
    todo = sorted([iid for iid in picked if iid not in done], key=lambda i: m_of[i])
    print(f"target {args.target}; {len(todo)} to run now (ascending m)")
    if todo:
        cmd = [sys.executable, "-m", "clue_leak.run_combo2", "--ids", ",".join(todo)]
        subprocess.run(cmd, cwd=ROOT, check=False)
    # 生成图
    from importlib import import_module
    plot = import_module("clue_leak.plot_one_mpl")
    made = 0
    for iid in picked:
        if os.path.exists(os.path.join(OUTDIR, iid + ".json")):
            sys.argv = ["plot_one_mpl", iid.split("_")[0]]
            try:
                plot.main(); made += 1
            except Exception as e:
                print(f"  plot fail {iid[:22]}: {e}")
    print(f"done: {made} figures for {len(picked)} target images")


if __name__ == "__main__":
    main()
