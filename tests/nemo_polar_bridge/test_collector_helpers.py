from __future__ import annotations

from nemo_polar_bridge.collector import normalize_openai_base_url, reward_std, stable_bucket


def test_normalize_openai_base_url_strips_v1_suffix() -> None:
    assert normalize_openai_base_url("http://10.0.0.2:31000/v1") == "http://10.0.0.2:31000"
    assert normalize_openai_base_url("http://10.0.0.2:31000/v1/") == "http://10.0.0.2:31000"
    assert normalize_openai_base_url("http://10.0.0.2:31000") == "http://10.0.0.2:31000"


def test_reward_std_detects_non_degenerate_group() -> None:
    assert reward_std([1.0, 0.0]) == 0.5
    assert reward_std([1.0, 1.0]) == 0.0


def test_stable_bucket_is_deterministic() -> None:
    assert stable_bucket("abc", 7) == stable_bucket("abc", 7)
    assert 0 <= stable_bucket("abc", 7) < 7
