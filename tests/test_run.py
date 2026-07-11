"""run.py 配置装载：显式路径不存在必须报错（防止拿默认值静默跑付费 API）。"""
import pytest

from run import load_config


def test_load_config_none_returns_empty():
    assert load_config(None) == {}


def test_load_config_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "no_such_config.yaml"))


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("k: 3\ntau_stop: 0.7\n", encoding="utf-8")
    assert load_config(str(p)) == {"k": 3, "tau_stop": 0.7}


def test_build_search_client_none_when_v2_disabled():
    from run import build_search_client
    assert build_search_client({"enable_hierarchy": False, "enable_enhance": False}) is None


def test_build_search_client_present_when_v2_enabled():
    from run import build_search_client
    sc = build_search_client({"enable_hierarchy": True, "enable_enhance": False,
                              "search_cache_dir": None})
    assert sc is not None
    assert hasattr(sc, "text_search")
