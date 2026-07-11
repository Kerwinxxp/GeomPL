"""WebSearch 模块（v2）：Tavily 文本检索 + SerpApi Google Lens 图像检索。

- transport 可注入（Callable[[query], list[dict]]），离线可测；
- 磁盘缓存，key = sha256(kind + query)；
- 检索是尽力而为：transport 抛错或无结果 → 返回 []，绝不炸掉推理
  （论文 Enhance：ImageSearch 失败则回退 TextSearch，检索层容错）。
- 查询按推理层级适配（论文 Enhance 段的 level-adapted query）。

level-adapted 查询模板 [assumption, map §2.2]，论文只给了国家/美国城市两个示例。
"""
import hashlib
import json
import os


def build_level_query(obj: str, level: str, parent: str | None) -> str:
    obj = str(obj).strip()
    if level == "country":
        return f"{obj} in which country?"
    if level == "city":
        where = f"in which city of {parent}" if parent else "in which city"
        return f"{obj} {where}?"
    if level == "street":
        where = f"on which street in {parent}" if parent else "on which street"
        return f"{obj} {where}?"
    return f"{obj} where?"


class WebSearchClient:
    def __init__(self, tavily_key: str | None = None, serpapi_key: str | None = None,
                 cache_dir: str | None = None, text_transport=None, image_transport=None,
                 max_results: int = 3):
        self.tavily_key = tavily_key
        self.serpapi_key = serpapi_key
        self.max_results = max_results
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._text = text_transport or self._default_text_transport()
        self._image = image_transport or self._default_image_transport()

    def text_search(self, query: str, max_results: int | None = None) -> list:
        return self._search("text", self._text, query, max_results)

    def image_search(self, image_url: str | None, query_hint: str = "") -> list:
        if not image_url:
            return []
        return self._search("image", self._image, f"{image_url}||{query_hint}", None)

    # ---------- 内部 ----------

    def _search(self, kind: str, transport, query: str, max_results) -> list:
        cached = self._cache_get(kind, query)
        if cached is not None:
            return cached
        try:
            results = transport(query) or []
        except Exception:
            results = []   # 检索失败静默降级
        results = results[: (max_results or self.max_results)]
        self._cache_put(kind, query, results)
        return results

    def _cache_key(self, kind: str, query: str) -> str:
        return hashlib.sha256(f"{kind}|{query}".encode()).hexdigest()

    def _cache_get(self, kind, query):
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, self._cache_key(kind, query) + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, kind, query, results):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, self._cache_key(kind, query) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    def _default_text_transport(self):
        def transport(query):
            key = self.tavily_key or os.environ.get("TAVILY_API_KEY")
            if not key:
                raise RuntimeError("no TAVILY_API_KEY; inject text_transport for offline use")
            import urllib.request
            body = json.dumps({"api_key": key, "query": query,
                               "max_results": self.max_results}).encode()
            req = urllib.request.Request(
                "https://api.tavily.com/search", data=body,
                headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=25).read())
            return [{"title": r.get("title", ""), "content": r.get("content", ""),
                     "url": r.get("url", "")} for r in d.get("results", [])]
        return transport

    def _default_image_transport(self):
        def transport(query):
            key = self.serpapi_key or os.environ.get("SERPAPI_API_KEY")
            if not key:
                raise RuntimeError("no SERPAPI_API_KEY; inject image_transport for offline use")
            import urllib.parse
            import urllib.request
            image_url, _, hint = query.partition("||")
            params = urllib.parse.urlencode(
                {"engine": "google_lens", "url": image_url, "api_key": key})
            d = json.loads(urllib.request.urlopen(
                f"https://serpapi.com/search?{params}", timeout=25).read())
            out = []
            for m in d.get("visual_matches", [])[: self.max_results]:
                out.append({"title": m.get("title", ""),
                            "content": m.get("source", ""), "url": m.get("link", "")})
            return out
        return transport


class MockSearchClient:
    """脚本化搜索客户端，用于 controller v2 控制流测试。

    script: {"text": [result_list, ...], "image": [result_list, ...]}
    依次弹出；耗尽后重复最后一个（None 表示空结果）。
    """
    def __init__(self, script: dict | None = None):
        script = script or {}
        self._text = list(script.get("text", [[]]))
        self._image = list(script.get("image", [[]]))
        self.text_calls, self.image_calls = [], []

    def text_search(self, query: str, max_results: int | None = None) -> list:
        self.text_calls.append(query)
        item = self._text.pop(0) if len(self._text) > 1 else (self._text[0] if self._text else [])
        return list(item or [])

    def image_search(self, image_url: str | None, query_hint: str = "") -> list:
        self.image_calls.append((image_url, query_hint))
        if not image_url:
            return []
        item = self._image.pop(0) if len(self._image) > 1 else (self._image[0] if self._image else [])
        return list(item or [])
