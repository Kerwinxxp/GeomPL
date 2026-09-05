"""mPL 的几何与度量基元(纯 Python,零依赖):

- haversine_km / EARTH_RADIUS_KM:大圆距离;
- cluster_representatives / merge_distribution:把 min_dist_km 内的近重复地点合并
  为一簇,并把信念分布按代表相加(2 km 别名去重即用此对);
- build_geometry / mpl:gallery → (代表映射, 簇列表, 距离函数),以及
  mPL = 逐候选对 |Δ log-odds| / 距离 的均值(nats/1000 km)。

原先分散在 geobayes.eval.{metrics,candidates} 与 run_georanker_check.py 中,
现集中于此:所有 GeoRanker 口径的脚本共用同一份几何,保证结果可比。
"""
import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def cluster_representatives(coords: dict, min_dist_km: float) -> dict:
    """把 min_dist_km 内的近重复地点合并为一簇,返回 {label: 代表label}。

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


def build_geometry(gv, merge_km=25.0):
    """gallery(含 gps 的记录列表)→ (label→代表, 代表列表, 距离函数(i,j)→km)。"""
    coords = {g["label"]: g["gps"] for g in gv if g["gps"]}
    rep = cluster_representatives(coords, merge_km)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mpl(prior, post, rep, clusters, dist):
    """metric-normalized posterior leakage:逐候选对 |Δlog-odds|/距离 的均值(×1000 km)。"""
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)
    keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    llr = {k: math.log(po[k] / pr[k]) for k in keys}
    ks = list(llr)
    vals = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = dist(ks[i], ks[j])
            if d and d > 0:
                vals.append(abs(llr[ks[i]] - llr[ks[j]]) / d * 1000)
    return sum(vals) / len(vals) if vals else 0.0
