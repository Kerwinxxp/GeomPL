"""Eq.7: P(l|E1:t) ∝ P(l|E1:t-1) * W(et|l)，每步归一化。  [paper, 由 Fig.3 全链定死]

不剪枝：键集不变，小概率假设保留（Fig.3 中 SE 以 0.019 带到层末）。
"""


def bayes_step(posterior: dict, weights: dict) -> dict:
    if set(posterior) != set(weights):
        raise KeyError(
            f"hypothesis keys mismatch: {sorted(posterior)} vs {sorted(weights)}"
        )
    unnorm = {loc: posterior[loc] * weights[loc] for loc in posterior}
    z = sum(unnorm.values())
    if z <= 0:
        raise ValueError("total probability mass is zero after update")
    return {loc: v / z for loc, v in unnorm.items()}
