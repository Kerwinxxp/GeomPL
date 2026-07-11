"""mPL（Metric-normalized Posterior Leakage, Chen et al. 2026, Def 2.2/2.3）。

mPL(xi,xj) = |ln(post_i/post_j) − ln(prior_i/prior_j)| / d_ij
一张图的 mPL = sup over pairs = max_ij |LLR_i − LLR_j| / d_ij，LLR_k=ln(post_k/prior_k)。
"""
import math

import pytest

from geobayes.analysis.leakage import log_belief_shift, mpl_pair, mpl_sup


def test_log_belief_shift_is_log_post_over_prior():
    llr = log_belief_shift({"A": 0.5, "B": 0.5}, {"A": 0.8, "B": 0.2})
    assert llr["A"] == pytest.approx(math.log(0.8 / 0.5))
    assert llr["B"] == pytest.approx(math.log(0.2 / 0.5))


def test_mpl_pair_matches_hand_computation():
    # prior {A:0.5,B:0.5}, post {A:0.8,B:0.2}, d=2
    # Δlog-odds = ln((0.8/0.2)/(0.5/0.5)) = ln4 = 1.3863 ; /2 = 0.6931
    prior = {"A": 0.5, "B": 0.5}
    post = {"A": 0.8, "B": 0.2}
    assert mpl_pair(prior, post, "A", "B", 2.0) == pytest.approx(math.log(4) / 2)


def test_mpl_pair_is_symmetric_and_nonneg():
    prior = {"A": 0.5, "B": 0.5}
    post = {"A": 0.8, "B": 0.2}
    assert mpl_pair(prior, post, "A", "B", 2.0) == pytest.approx(mpl_pair(prior, post, "B", "A", 2.0))
    assert mpl_pair(prior, post, "A", "B", 2.0) >= 0


def test_mpl_pair_zero_when_no_belief_change():
    p = {"A": 0.5, "B": 0.5}
    assert mpl_pair(p, p, "A", "B", 3.0) == pytest.approx(0.0)


def test_mpl_sup_picks_max_normalized_pair():
    # 3 候选：A 涨、C 跌最多；近距离对给出更大 mPL
    prior = {"A": 1/3, "B": 1/3, "C": 1/3}
    post = {"A": 0.6, "B": 0.3, "C": 0.1}
    dist = {("A", "B"): 1.0, ("A", "C"): 10.0, ("B", "C"): 1.0}
    def d(i, j):
        return dist.get((i, j)) or dist.get((j, i))
    res = mpl_sup(prior, post, d)
    # 手算各对 |LLR_i-LLR_j|/d，取最大
    llr = {k: math.log(post[k] / prior[k]) for k in prior}
    best = max((abs(llr[i] - llr[j]) / d(i, j), (i, j))
               for i in prior for j in prior if i < j)
    assert res["mpl"] == pytest.approx(best[0])
    assert set(res["pair"]) == set(best[1])


def test_mpl_sup_skips_undefined_distances():
    # 距离函数对某对返回 None（无法地理编码）→ 跳过该对，不报错
    prior = {"A": 0.5, "B": 0.5}
    post = {"A": 0.9, "B": 0.1}
    def d(i, j):
        return None
    res = mpl_sup(prior, post, d)
    assert res["mpl"] is None
    assert res["n_pairs"] == 0
