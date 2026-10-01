"""Saved app settings: defaults, updates, and a broken file never stops the app."""
import json

from nexus import settings


def test_defaults_when_missing(tmp_path):
    s = settings.load(tmp_path / "none.json")
    assert s["hands"] is True and s["voice"] is True and s["wake_threshold"] == 0.35


def test_update_persists_and_ignores_unknown_keys(tmp_path):
    p = tmp_path / "s.json"
    settings.update(p, hands=False, bogus=1)
    s = settings.load(p)
    assert s["hands"] is False and s["voice"] is True and "bogus" not in s


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert settings.load(p) == settings.DEFAULTS


def test_old_default_threshold_is_upgraded(tmp_path):
    # Earlier versions saved 0.5, which was just too strict (near misses at 0.44-0.48)
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"wake_threshold": 0.5, "hands": False}))
    s = settings.load(p)
    assert s["wake_threshold"] == 0.35 and s["hands"] is False
    p.write_text(json.dumps({"wake_threshold": 0.6}))  # a value you chose on purpose is kept
    assert settings.load(p)["wake_threshold"] == 0.6
