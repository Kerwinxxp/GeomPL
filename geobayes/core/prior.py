"""Eq.5: P0(li) = exp(min(si, tau_p)/T) / sum_j exp(min(sj, tau_p)/T).  [paper]

tau_p=0.6, T=1.5（论文实现细节）。si 先钳制到 [0,1] [assumption, map §3.1]。
调用方（controller）负责把原始 si 落盘为 raw_scores。
"""
import math

TAU_P = 0.6
TEMPERATURE = 1.5


def compute_prior(raw_scores: dict, tau_p: float = TAU_P, temperature: float = TEMPERATURE) -> dict:
    if not raw_scores:
        raise ValueError("raw_scores must contain at least one candidate")
    clipped = {
        loc: min(min(1.0, max(0.0, float(s))), tau_p) / temperature
        for loc, s in raw_scores.items()
    }
    exps = {loc: math.exp(v) for loc, v in clipped.items()}
    z = sum(exps.values())
    return {loc: v / z for loc, v in exps.items()}
