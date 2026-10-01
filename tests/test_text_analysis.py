import math
import sys

import numpy as np
import pandas as pd
import pytest

import text_analysis


DUP = "the committee met on monday and repeated the same sentence about the harbour."
OTHER = "buses were late because the rain flooded the baseline road near the market."


def rows(records):
    frame = pd.DataFrame(records)
    frame["created_dt"] = pd.to_datetime(frame["created_utc"], unit="s", utc=True)
    return frame


def test_import_does_not_load_sentence_transformers():
    assert "sentence_transformers" not in sys.modules
    assert not hasattr(text_analysis, "SentenceTransformer")


def test_norm_text_lowercases_urls_mentions_and_whitespace():
    raw = "Hello  HTTPS://Example.com/a?x=1  u/Some_User \n bye"
    assert text_analysis.norm_text(raw) == "hello URL USER bye"
    assert text_analysis.norm_text(None) == "none"


def test_eligible_activity_drops_deleted_and_short_text():
    act = rows([
        {"id": "1", "author": "[deleted]", "created_utc": 1, "text": DUP, "subreddit": "srilanka"},
        {"id": "2", "author": "alice", "created_utc": 2, "text": "too short", "subreddit": "srilanka"},
        {"id": "3", "author": "alice", "created_utc": 3, "text": "See HTTPS://x.test/a and u/Bob", "subreddit": "Colombo"},
    ])

    eligible = text_analysis.eligible_activity(act, min_text_chars=30)

    assert list(eligible["id"]) == ["3"]
    assert eligible.iloc[0]["norm_text"] == "see URL and USER"


def test_near_duplicates_are_cross_account_only():
    text = "please read https://example.com/one for the full notice today"
    other_url = "please read https://other.test/two for the full notice today"
    act = rows([
        {"id": "a1", "author": "alice", "created_utc": 1, "text": text, "subreddit": "srilanka"},
        {"id": "a2", "author": "alice", "created_utc": 2, "text": text, "subreddit": "srilanka"},
        {"id": "b1", "author": "bob", "created_utc": 3, "text": other_url, "subreddit": "Colombo"},
        {"id": "c1", "author": "cara", "created_utc": 4, "text": OTHER, "subreddit": "AskSriLanka"},
    ])
    act = text_analysis.eligible_activity(act, min_text_chars=30)

    pairs = text_analysis.find_near_duplicate_pairs(act, threshold=0.88, n_jobs=1)
    undirected = {frozenset((p["id_a"], p["id_b"])) for p in pairs}

    assert undirected == {frozenset(("a1", "b1")), frozenset(("a2", "b1"))}
    assert all(p["similarity"] == pytest.approx(1.0) for p in pairs)
    assert all(p["author_a"] != p["author_b"] for p in pairs)
    assert all("c1" not in (p["id_a"], p["id_b"]) for p in pairs)


def test_near_duplicates_of_identical_text_ignore_unrelated_rows():
    act = rows([
        {"id": "a1", "author": "alice", "created_utc": 1, "text": DUP, "subreddit": "srilanka"},
        {"id": "b1", "author": "bob", "created_utc": 2, "text": DUP, "subreddit": "srilanka"},
        {"id": "c1", "author": "cara", "created_utc": 3, "text": OTHER, "subreddit": "Colombo"},
    ])
    act = text_analysis.eligible_activity(act, min_text_chars=30)

    pairs = text_analysis.find_near_duplicate_pairs(act, threshold=0.88, n_jobs=1)

    assert {frozenset((p["id_a"], p["id_b"])) for p in pairs} == {frozenset(("a1", "b1"))}
    assert all(p["similarity"] > 0.99 for p in pairs)
    assert all("c1" not in (p["id_a"], p["id_b"]) for p in pairs)


def test_single_row_has_no_pairs():
    act = rows([
        {"id": "a1", "author": "alice", "created_utc": 1, "text": DUP, "subreddit": "srilanka"},
    ])
    act = text_analysis.eligible_activity(act, min_text_chars=30)

    assert text_analysis.find_near_duplicate_pairs(act, threshold=0.88, n_jobs=1) == []


def test_embedding_frame_keeps_the_latest_rows():
    act = rows([
        {"id": "old", "author": "a", "created_utc": 10, "text": DUP, "subreddit": "srilanka"},
        {"id": "mid", "author": "b", "created_utc": 30, "text": DUP, "subreddit": "srilanka"},
        {"id": "new", "author": "c", "created_utc": 20, "text": DUP, "subreddit": "srilanka"},
    ])

    sample = text_analysis.embedding_frame(act, nmax=2)

    assert list(sample["id"]) == ["new", "mid"]
    assert list(sample.index) == [0, 1]

    assert list(text_analysis.embedding_frame(act, nmax=10)["id"]) == ["old", "new", "mid"]


def test_summarize_semantic_clusters_drops_noise_and_empty():
    created = pd.to_datetime([1, 2, 3, 4], unit="s", utc=True)
    emb = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "author": ["alice", "bob", "cara", "alice"],
        "created_dt": created,
        "subreddit": ["Colombo", "srilanka", "AskSriLanka", "srilanka"],
        "semantic_cluster": [1, 1, 1, -1],
    })

    summary = text_analysis.summarize_semantic_clusters(emb)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["semantic_cluster"] == 1
    assert row["cluster_size"] == 3
    assert row["unique_accounts"] == 3
    assert row["first_time"] == created[0]
    assert row["last_time"] == created[2]
    assert row["subreddits"] == "AskSriLanka,Colombo,srilanka"

    empty = text_analysis.summarize_semantic_clusters(emb.assign(semantic_cluster=-1))
    assert list(empty.columns) == ["semantic_cluster", "cluster_size", "unique_accounts"]
    assert empty.empty


def test_similarity_at_or_above_threshold_is_kept():
    act = rows([
        {"id": "a1", "author": "alice", "created_utc": 1, "text": DUP, "subreddit": "srilanka"},
        {"id": "b1", "author": "bob", "created_utc": 2, "text": DUP, "subreddit": "srilanka"},
    ])
    act = text_analysis.eligible_activity(act, min_text_chars=30)
    pairs = text_analysis.find_near_duplicate_pairs(act, threshold=0.88, n_jobs=1)

    assert pairs
    similarity = pairs[0]["similarity"]
    assert math.isfinite(similarity)
    assert similarity >= 0.88
    assert text_analysis.find_near_duplicate_pairs(act, threshold=similarity, n_jobs=1)
    assert text_analysis.find_near_duplicate_pairs(act, threshold=similarity + 0.001, n_jobs=1) == []


def test_resolve_device_uses_cuda_only_when_it_is_available():
    assert text_analysis.resolve_device("auto", cuda_is_available=True) == "cuda"
    assert text_analysis.resolve_device("auto", cuda_is_available=False) == "cpu"
    assert text_analysis.resolve_device("cuda", cuda_is_available=False) == "cpu"
    assert text_analysis.resolve_device("cpu", cuda_is_available=True) == "cpu"
    assert text_analysis.encode_batch_size(None, "cuda") == 256
    assert text_analysis.encode_batch_size(None, "cpu") == 32
    assert text_analysis.encode_batch_size(64, "cpu") == 64


def test_labels_from_neighborhoods_marks_border_points_and_noise():
    neighborhoods = [
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 3]),
        np.array([4]),
    ]

    labels = text_analysis.labels_from_neighborhoods(neighborhoods, min_samples=3)

    assert list(labels) == [0, 0, 0, 0, -1]


def test_cosine_neighborhoods_match_a_direct_similarity_cutoff():
    pytest.importorskip("torch")
    first = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32)
    first /= np.linalg.norm(first, axis=1, keepdims=True)

    neighborhoods = text_analysis.cosine_neighborhoods(first, min_similarity=0.9, device="cpu", batch_size=2)

    assert list(neighborhoods[0]) == [0, 1]
    assert list(neighborhoods[1]) == [0, 1]
    assert list(neighborhoods[2]) == [2]
    assert text_analysis.cluster_embeddings(first, eps=0.1, min_samples=2, device="cpu", batch_size=2).shape == (3,)
