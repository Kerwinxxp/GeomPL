"""地名规范化（离线）与地理编码（在线，Nominatim，Phase 3 接入）。

规范化用于：Top-K recall 对拍 Table 3、假设标签跨步匹配。
论文自身混用 Scotland/UK/United Kingdom（Fig.4、正文），故 UK 构成国并入 United Kingdom
计 recall [inferred, plan §1.5#16]。
"""

_ALIASES = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "scotland": "United Kingdom",
    "england": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "america": "United States",
    "prc": "China",
    "people's republic of china": "China",
    "mainland china": "China",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "uae": "United Arab Emirates",
    "russia": "Russia",
    "russian federation": "Russia",
    "czechia": "Czech Republic",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "burma": "Myanmar",
    "côte d'ivoire": "Ivory Coast",
}


def canonicalize_country(name: str) -> str:
    cleaned = " ".join(str(name).strip().split())
    return _ALIASES.get(cleaned.lower(), cleaned)


_LEVEL_ORDER = ["street", "city", "country"]   # 最细在前


def assemble_name(parts) -> str:
    """把 [street, city, country] 组装成 '最细, ..., 最粗'，跳过空/None/空白。"""
    kept = [str(p).strip() for p in parts if p and str(p).strip()]
    return ", ".join(kept)


def hierarchical_name(result: dict) -> str:
    """从结果重建层级地名（最细在前）。支持三种形态：
    - zero-shot：{"zero_shot": {"country","city","street"}}
    - GeoBayes：{"levels": [...每层 posterior...]}（每层取 argmax）
    - 兜底：{"final_posterior": {"hypotheses": {...}}}（取 argmax）
    """
    zs = result.get("zero_shot")
    if zs:
        return assemble_name([zs.get("street"), zs.get("city"), zs.get("country")])
    levels = result.get("levels")
    if levels:
        by_level = {L["level"]: max(L["posterior"], key=L["posterior"].get) for L in levels}
        return assemble_name([by_level.get(lv) for lv in _LEVEL_ORDER])
    hyp = result.get("final_posterior", {}).get("hypotheses", {})
    return max(hyp, key=hyp.get) if hyp else ""


def forward_geocode(name: str, transport=None, cache=None):
    """地名 → [lat, lon]，失败返回 None。

    transport 可注入（Callable[[name], [lat,lon]|None]），离线可测；默认走 Nominatim。
    cache（dict）命中免请求；失败也缓存（None）以避免重复请求。
    """
    if cache is not None and name in cache:
        return cache[name]
    tr = transport or _nominatim_forward
    try:
        result = tr(name)
    except Exception:
        result = None
    if cache is not None:
        cache[name] = result
    return result


def _nominatim_forward(name: str):
    import json
    import urllib.parse
    import urllib.request
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": name, "format": "jsonv2", "limit": 1})
    req = urllib.request.Request(
        url, headers={"User-Agent": "GeoBayes-reproduction/0.1 (academic)"})
    d = json.loads(urllib.request.urlopen(req, timeout=20).read())
    return [float(d[0]["lat"]), float(d[0]["lon"])] if d else None


def hierarchical_name(result: dict) -> str:
    """从推理结果重建完整层级地名 'street, city, country'（论文口径 geocode 用）。

    优先用 result['levels'] 每层 posterior 的 argmax（street→city→country）；
    zero-shot 结果用 result['zero_shot'] 三段；否则退回 final_posterior argmax。
    """
    zs = result.get("zero_shot")
    if zs:
        parts = [zs.get("street"), zs.get("city"), zs.get("country")]
        return ", ".join(p for p in parts if p)

    levels = result.get("levels")
    if levels:
        by_level = {L["level"]: max(L["posterior"], key=L["posterior"].get) for L in levels}
        parts = [by_level.get(lv) for lv in ("street", "city", "country")]
        return ", ".join(p for p in parts if p)

    fp = result.get("final_posterior", {}).get("hypotheses", {})
    return max(fp, key=fp.get) if fp else ""


_last_forward = [0.0]


def forward_geocode(name: str, cache: dict | None = None, transport=None):
    """地名 → [lat, lon]（Nominatim search）。可注入 transport 离线测试；失败/空 → None。"""
    if cache is not None and name in cache:
        return cache[name]
    transport = transport or _default_forward_transport
    try:
        result = transport(name)
    except Exception:
        result = None
    if cache is not None:
        cache[name] = result
    return result


def _default_forward_transport(name: str):
    import json
    import time
    import urllib.parse
    import urllib.request
    wait = 1.1 - (time.time() - _last_forward[0])
    if wait > 0:
        time.sleep(wait)
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": name, "format": "jsonv2", "limit": 1})
    req = urllib.request.Request(url, headers={
        "User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"})
    _last_forward[0] = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=20).read())
    return [float(d[0]["lat"]), float(d[0]["lon"])] if d else None
