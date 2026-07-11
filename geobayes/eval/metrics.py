"""haversine 距离与论文五档阈值准确率（1/25/200/750/2500 km）。"""
import math

EARTH_RADIUS_KM = 6371.0
PAPER_THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def top_k_recall(records, ks=(1, 3, 5)) -> dict:
    """Table 3 口径：records = [(有序候选国家列表, GT 国家)]，规范化后比较。"""
    from .geocode import canonicalize_country
    records = list(records)
    if not records:
        raise ValueError("no records given")
    hits = {k: 0 for k in ks}
    for candidates, gt in records:
        canon = [canonicalize_country(c) for c in candidates]
        gt_c = canonicalize_country(gt)
        for k in ks:
            if gt_c in canon[:k]:
                hits[k] += 1
    return {k: hits[k] / len(records) for k in ks}


def top_k_recall_from_prior(records, ks=(1, 3, 5)) -> dict:
    """records = [({location: prob}, GT 国家)]，按概率降序后计 recall。"""
    ordered = [
        (sorted(prior, key=prior.get, reverse=True), gt)
        for prior, gt in records
    ]
    return top_k_recall(ordered, ks=ks)


def localization_distance_km(result, gt_lat, gt_lon, geocoder):
    """论文 Table 1 口径：结果 → 层级地名 → geocoder 坐标 → 与 GT 的 haversine 距离。

    geocoder: Callable[[name], [lat,lon]|None]。地名无法编码 → 返回 None（该图不计入）。
    """
    from .geocode import hierarchical_name
    name = hierarchical_name(result)
    if not name:
        return None
    coords = geocoder(name)
    if not coords:
        return None
    return haversine_km(gt_lat, gt_lon, coords[0], coords[1])


def threshold_accuracy(distances_km, thresholds=PAPER_THRESHOLDS_KM) -> dict:
    """None = 无法地理编码 = 未命中，但计入分母（公平口径）。"""
    distances_km = list(distances_km)
    if not distances_km:
        raise ValueError("no distances given")
    n = len(distances_km)
    return {t: sum(1 for d in distances_km if d is not None and d <= t) / n
            for t in thresholds}
