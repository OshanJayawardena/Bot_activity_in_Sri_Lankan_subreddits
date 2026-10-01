from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_declares_collection_analysis_and_topics():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    assert cfg["subreddits"] == ["ask_srilanka", "srilanka", "Colombo"]
    for key in ("after", "limit_per_request", "sleep_seconds", "max_pages_per_subreddit"):
        assert key in cfg["collection"]
    for key in (
        "new_account_days",
        "very_new_account_days",
        "min_text_chars",
        "near_duplicate_threshold",
        "semantic_similarity_threshold",
        "temporal_window_minutes",
        "min_cluster_size",
        "max_embedding_rows",
        "device",
    ):
        assert key in cfg["analysis"]
    assert set(cfg["topics"]) == {
        "sinhala",
        "tamil",
        "muslim",
        "buddhist",
        "hindu",
        "christian",
        "ethnic",
        "communal",
    }
    assert all(isinstance(words, list) and words for words in cfg["topics"].values())
