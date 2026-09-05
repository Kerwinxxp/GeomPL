"""Geometry and metric primitives for mPL (pure Python, no dependencies).

- haversine_km / EARTH_RADIUS_KM: great-circle distance;
- cluster_representatives / merge_distribution: collapse near-duplicate places within
  min_dist_km into one cluster and add up the belief mass per representative (the 2 km
  alias dedup uses this pair);
- build_geometry / mpl: gallery -> (representative map, cluster list, distance function),
  and mPL = mean over candidate pairs of |delta log-odds| / distance (nats/1000 km).

Previously scattered across geobayes.eval.{metrics,candidates} and
run_georanker_check.py; centralised here so every GeoRanker-line script shares one
geometry and the numbers stay comparable.

Library module — no CLI.
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
    """Merge places closer than min_dist_km into one cluster -> {label: representative}.

    Greedy over sorted labels: each label joins the first existing cluster whose centre is
    < min_dist_km away, otherwise it starts one. Deterministic (the sort makes it reproducible).
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
    """Add probabilities up per representative -> the merged (de-aliased) distribution."""
    merged = {}
    for lbl, p in dist.items():
        rep = label_to_rep.get(lbl, lbl)
        merged[rep] = merged.get(rep, 0.0) + p
    return merged


def build_geometry(gv, merge_km=25.0):
    """Gallery (records carrying gps) -> (label->representative, clusters, dist(i, j) in km)."""
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
    """Metric-normalized posterior leakage: mean over candidate pairs of |dlog-odds|/km * 1000."""
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
