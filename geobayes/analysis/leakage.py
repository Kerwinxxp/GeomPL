"""Metric-normalized Posterior Leakage (mPL), Chen et al. 2026 (Def 2.2/2.3).

把闭集的先验/后验信念分布 + 候选位置间的地理距离，映射为距离归一化的后验泄露：
  LLR_k = ln(post_k / prior_k)                              # 每候选的信念对数移动
  mPL(xi,xj) = |LLR_i − LLR_j| / d_ij                       # 一对候选的距离归一化几率漂移
  一张图的 mPL = sup over pairs（Def 2.3）                  # 最"泄露"的那对
单位：nats / (距离单位)。距离用 km → nats/km。近距离对的大几率漂移 = 强细粒度泄露。
"""
import math


def log_belief_shift(prior: dict, posterior: dict) -> dict:
    """LLR_k = ln(post_k / prior_k)。要求同支撑、无零（闭集打分已平滑保证）。"""
    return {k: math.log(posterior[k] / prior[k]) for k in prior}


def mpl_pair(prior, posterior, i, j, d_ij) -> float:
    """一对候选的 mPL：|Δ log posterior-odds − Δ log prior-odds| / d_ij。"""
    llr = log_belief_shift(prior, posterior)
    return abs(llr[i] - llr[j]) / d_ij


def mpl_sup(prior, posterior, distance) -> dict:
    """图级 mPL = sup over 候选对（Def 2.3）。

    distance(i, j) → 距离或 None（无法定位则跳过该对）。
    返回 {mpl, pair, n_pairs, per_pair_max_d}。无可用对时 mpl=None。
    """
    llr = log_belief_shift(prior, posterior)
    keys = list(prior)
    best_val, best_pair, n = None, None, 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            d = distance(i, j)
            if not d or d <= 0:
                continue
            n += 1
            v = abs(llr[i] - llr[j]) / d
            if best_val is None or v > best_val:
                best_val, best_pair = v, (i, j)
    return {"mpl": best_val, "pair": best_pair, "n_pairs": n}
