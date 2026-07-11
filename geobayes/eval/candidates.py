"""闭集候选构造：真值 + K 个硬负样本（地理最近的其它 GT 点）。

硬负样本 = 池中与查询 GT 地理最近的 k 个不同地点 → 视觉/地理相似，难区分，
使闭集打分成为有意义的排序任务（而非随机全球干扰项那种一眼排除的退化设置）。
候选顺序按 seed 随机化以抵消 LLM 的位置偏好，但可复现。
"""
import random

from .metrics import haversine_km


def cluster_representatives(coords: dict, min_dist_km: float) -> dict:
    """把 min_dist_km 内的近重复地点合并为一簇，返回 {label: 代表label}。

    贪心：按 label 排序遍历，每个 label 归入首个距其 < min_dist_km 的已建代表簇；
    否则自立为新代表。确定性（排序保证可复现）。
    """
    reps = {}
    centers = []   # [(rep_label, lat, lon)]
    for lbl in sorted(coords):
        lat, lon = coords[lbl]
        assigned = None
        for rep, rlat, rlon in centers:
            if haversine_km(lat, lon, rlat, rlon) < min_dist_km:
                assigned = rep
                break
        if assigned is None:
            centers.append((lbl, lat, lon))
            assigned = lbl
        reps[lbl] = assigned
    return reps


def merge_distribution(dist: dict, label_to_rep: dict) -> dict:
    """按代表映射把概率相加，得到合并后（干净全集）的分布。"""
    merged = {}
    for lbl, p in dist.items():
        rep = label_to_rep.get(lbl, lbl)
        merged[rep] = merged.get(rep, 0.0) + p
    return merged


def build_hard_negative_set(true_label, true_lat, true_lon, pool, k, seed):
    # 去重（按 label）、排除与真值同名者
    seen, uniq = set(), []
    for it in pool:
        lbl = it["label"]
        if lbl == true_label or lbl in seen:
            continue
        seen.add(lbl)
        uniq.append(it)
    uniq.sort(key=lambda it: haversine_km(true_lat, true_lon, it["lat"], it["lon"]))
    negatives = [it["label"] for it in uniq[:k]]

    candidates = [true_label] + negatives
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return {"candidates": candidates, "true_label": true_label}
