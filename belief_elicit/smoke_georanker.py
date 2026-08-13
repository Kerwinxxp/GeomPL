"""【实验性 · 可整体删除】GeoRanker 冒烟测:方向 / 确定性 / 显存 / 速度。

Okazaki 原图 × 2 候选(Okazaki,Japan vs City of Westminster,UK):
  ① reward(冈崎) 应显著 > reward(伦敦)   —— 方向对
  ② 同输入跑 2 遍,reward 差 < 1e-3        —— 确定性
  ③ 报显存峰值与单前向耗时                 —— 定 batch_size
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.smoke_georanker
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from PIL import Image

from belief_elicit.georanker_belief import score_rewards

IMG = os.path.join(ROOT, "data", "sample_images",
                   "261517384_292417efcc_117_60558526@N00.jpg")
CANDS = [("Okazaki, Japan", (34.9551, 137.1740)),
         ("City of Westminster, UK", (51.4973, -0.1372))]


def main():
    img = Image.open(IMG)
    gps = [c[1] for c in CANDS]
    texts = [c[0] for c in CANDS]

    t0 = time.time()
    r1 = score_rewards(img, gps, cand_texts=texts, batch_size=2)
    t_first = time.time() - t0            # 含模型加载
    t0 = time.time()
    r2 = score_rewards(img, gps, cand_texts=texts, batch_size=2)
    t_second = time.time() - t0           # 纯前向(2 条)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"reward(Okazaki)  = {r1[0]:+.4f}")
    print(f"reward(London)   = {r1[1]:+.4f}")
    print(f"① 方向: {'PASS' if r1[0] > r1[1] else 'FAIL'}  (差 {r1[0]-r1[1]:+.4f})")
    dmax = max(abs(a - b) for a, b in zip(r1, r2))
    print(f"② 确定性: {'PASS' if dmax < 1e-3 else 'FAIL'}  (两次最大差 {dmax:.2e})")
    print(f"③ 显存峰值 {peak:.1f} GB | 首跑(含加载) {t_first:.0f}s | 二跑(2条前向) {t_second:.1f}s "
          f"→ ~{t_second/2:.2f}s/候选")
    est = t_second / 2 * 138
    print(f"   估算:138 候选一次打分 ≈ {est:.0f}s;Okazaki 体检(3变体×16打分) ≈ {est*48/60:.0f} min")


if __name__ == "__main__":
    main()
