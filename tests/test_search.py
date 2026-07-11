"""WebSearchClient 离线单元测试：可注入 transport / 磁盘缓存 / 空结果容错。

真实后端：Tavily（text）+ SerpApi Google Lens（image，需公网 URL）。
transport 注入使测试不发网络请求。
"""
import pytest

from geobayes.search.client import WebSearchClient, build_level_query


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def make_client(tmp_path, text=None, image=None):
    return WebSearchClient(cache_dir=str(tmp_path / "search_cache"),
                           text_transport=text, image_transport=image)


# ---------- level-adapted 查询模板（map §2.2 / 论文 Enhance 段） ----------

def test_query_country_level():
    q = build_level_query("red double-decker bus", level="country", parent=None)
    assert "red double-decker bus" in q
    assert "country" in q.lower()


def test_query_city_level_names_parent():
    q = build_level_query("cable car", level="city", parent="United States")
    assert "cable car" in q
    assert "United States" in q
    assert "cit" in q.lower()


def test_query_street_level():
    q = build_level_query("tram wires", level="street", parent="San Francisco")
    assert "San Francisco" in q
    assert "street" in q.lower()


# ---------- text_search ----------

def test_text_search_returns_snippets(tmp_path):
    t = FakeTransport([[{"title": "Route master", "content": "London red bus", "url": "u"}]])
    client = make_client(tmp_path, text=t)
    r = client.text_search("red bus in which country?")
    assert r[0]["content"] == "London red bus"
    assert t.calls == ["red bus in which country?"]


def test_text_search_cached(tmp_path):
    t = FakeTransport([[{"title": "x", "content": "y", "url": "u"}]])
    client = make_client(tmp_path, text=t)
    client.text_search("q1")
    client.text_search("q1")
    assert len(t.calls) == 1  # 第二次命中缓存

    # 新实例仍命中磁盘缓存
    t2 = FakeTransport([[{"title": "other", "content": "z", "url": "u"}]])
    client2 = make_client(tmp_path, text=t2)
    r = client2.text_search("q1")
    assert r[0]["content"] == "y"
    assert len(t2.calls) == 0


def test_text_search_empty_results_ok(tmp_path):
    t = FakeTransport([[]])
    client = make_client(tmp_path, text=t)
    assert client.text_search("nonsense") == []


def test_text_search_transport_error_returns_empty(tmp_path):
    def boom(query):
        raise RuntimeError("network down")
    client = make_client(tmp_path, text=boom)
    # 搜索失败不得炸掉推理（论文：ImageSearch 失败回退，检索是尽力而为）
    assert client.text_search("q") == []


# ---------- image_search ----------

def test_image_search_returns_snippets(tmp_path):
    t = FakeTransport([[{"title": "Golden Gate", "content": "SF landmark", "url": "u"}]])
    client = make_client(tmp_path, image=t)
    r = client.image_search("https://example.com/img.jpg", query_hint="in which city?")
    assert r[0]["title"] == "Golden Gate"


def test_image_search_none_url_returns_empty(tmp_path):
    client = make_client(tmp_path)
    # 无公网 URL（如裁剪对象无法托管）→ 空，触发 TextSearch 回退（由 controller 处理）
    assert client.image_search(None) == []
