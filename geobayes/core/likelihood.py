"""Eq.6: W(et|li) = exp[alpha * beta * (c - 3)], beta = ln2.  [paper]

c ∈ {1..5}（1=强矛盾, 3=中性, 5=强支持），alpha ∈ [0,1]。
越界输入钳制（judge 输出容错）[assumption, map §3.3]。
"""
import math

BETA = math.log(2)


def support_score(c: float, alpha: float) -> float:
    c = min(5, max(1, int(round(float(c)))))
    alpha = min(1.0, max(0.0, float(alpha)))
    return math.exp(alpha * BETA * (c - 3))
