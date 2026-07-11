"""隐私侧信念度量：熵、KL、逐线索证据权重（bits）。

weight_of_evidence = log2 W —— 每条线索对每个假设的信念推动量（bit），
Eq.6 的直接副产品，用于 per-clue 泄露量分解。
"""
import math


def entropy_bits(p: dict) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def kl_bits(p: dict, q: dict) -> float:
    """KL(p‖q)，要求同支撑。p 中零概率项贡献 0。"""
    if set(p) != set(q):
        raise KeyError("KL requires identical supports")
    total = 0.0
    for k, pv in p.items():
        if pv > 0:
            if q[k] <= 0:
                raise ValueError(f"q[{k}]=0 while p[{k}]>0: KL undefined")
            total += pv * math.log2(pv / q[k])
    return total


def weight_of_evidence_bits(weights: dict) -> dict:
    return {loc: math.log2(w) for loc, w in weights.items()}
