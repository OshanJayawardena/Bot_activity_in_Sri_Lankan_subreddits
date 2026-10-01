import pandas as pd
import pytest

import features


CFG = {
    "analysis": {
        "new_account_days": 30,
        "very_new_account_days": 7,
    },
    "topics": {
        "tamil": ["tamil", "tamils"],
        "islam_only": ["islam"],
        "dotted": ["a.b"],
    },
}


def stamp(seconds):
    return pd.to_datetime(seconds, unit="s", utc=True)


def activity_row(**overrides):
    row = {
        "id": "1",
        "author": "alice",
        "created_dt": stamp(1_700_000_000),
        "kind": "comment",
        "subreddit": "srilanka",
        "link_id": "t3_thread",
        "parent_id": None,
        "score": 1,
        "text": "hello",
    }
    row.update(overrides)
    return row


def test_safe_author():
    assert features.safe_author(None) == "[deleted]"
    assert features.safe_author(float("nan")) == "[deleted]"
    assert features.safe_author("") == "[deleted]"
    assert features.safe_author(0) == "[deleted]"
    assert features.safe_author("alice") == "alice"
    assert features.safe_author(5) == "5"


def test_normalize_activity_builds_text_by_kind():
    df = pd.DataFrame([
        {
            "id": "p1",
            "kind": "post",
            "author": None,
            "created_utc": 1_700_000_000,
            "title": "Hello   world",
            "selftext": "more\ntext",
            "body": "should-not-be-used",
        },
        {
            "id": "c1",
            "kind": "comment",
            "author": "",
            "created_utc": "not-a-time",
            "title": "ignored title",
            "selftext": "ignored selftext",
            "body": "  comment   body  ",
        },
    ])

    out = features.normalize_activity(df)

    assert out.loc[0, "author"] == "[deleted]"
    assert out.loc[0, "text"] == "Hello world more text"
    assert out.loc[0, "created_dt"] == stamp(1_700_000_000)
    assert out.loc[1, "author"] == "[deleted]"
    assert out.loc[1, "text"] == "comment body"
    assert pd.isna(out.loc[1, "created_dt"])


def test_load_raw_reads_posts_and_comments(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    posts = pd.DataFrame([
        {
            "id": "p1",
            "author": "alice",
            "created_utc": 1_700_000_000,
            "subreddit": "srilanka",
            "title": "A title",
            "selftext": "body of post",
            "score": 3,
        }
    ])
    comments = pd.DataFrame([
        {
            "id": "c1",
            "author": "bob",
            "created_utc": 1_700_000_100,
            "subreddit": "Colombo",
            "body": "A comment",
            "score": 1,
            "link_id": "t3_p1",
            "parent_id": "t3_p1",
        }
    ])
    posts.to_parquet(raw / "srilanka_posts.parquet", index=False)
    comments.to_parquet(raw / "Colombo_comments.parquet", index=False)

    df = features.load_raw(raw)

    kinds = dict(zip(df["id"], df["kind"]))
    assert kinds == {"p1": "post", "c1": "comment"}
    texts = dict(zip(df["id"], df["text"]))
    assert texts["p1"] == "A title body of post"
    assert texts["c1"] == "A comment"


def test_load_raw_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="No raw parquet files"):
        features.load_raw(tmp_path)


def test_account_features_counts_age_burst_and_topics():
    base = 1_700_000_000
    rows = pd.DataFrame([
        activity_row(
            id="p1", author="alice", kind="post", created_dt=stamp(base),
            subreddit="srilanka", link_id=None, score=2, text="hello tamil friends",
        ),
        activity_row(
            id="c1", author="alice", kind="comment", created_dt=stamp(base + 120 * 60),
            subreddit="srilanka", link_id="t3_p1", score=4, text="plain text",
        ),
        activity_row(
            id="c2", author="alice", kind="comment", created_dt=stamp(base + 60 * 60),
            subreddit="Colombo", link_id="t3_p1", score=0, text="more tamil notes",
        ),
        activity_row(
            id="b1", author="bob", kind="comment", created_dt=stamp(base + 10 * 86400),
            subreddit="srilanka", link_id="t3_other", score=5, text="nothing here",
        ),
    ])

    feat = features.build_account_features(rows, CFG)

    alice = feat.loc["alice"]
    assert alice["activity_count"] == 3
    assert alice["post_count"] == 1
    assert alice["comment_count"] == 2
    assert alice["subreddit_count"] == 2
    assert alice["thread_count"] == 1
    assert alice["mean_score"] == pytest.approx(2)
    assert alice["observed_span_days"] == pytest.approx(2 / 24)
    assert alice["activity_per_observed_day"] == pytest.approx(3 / (2 / 24))
    assert alice["median_interarrival_min"] == pytest.approx(60)
    assert alice["max_subreddit_share"] == pytest.approx(2 / 3)
    assert bool(alice["new_account_7d"]) is False
    assert bool(alice["new_account_30d"]) is True
    assert alice["topic_tamil_rate"] == pytest.approx(2 / 3)

    bob = feat.loc["bob"]
    assert bob["activity_count"] == 1
    assert bob["thread_count"] == 1
    assert bob["activity_per_observed_day"] == pytest.approx(24)
    assert pd.isna(bob["median_interarrival_min"])
    assert bob["max_subreddit_share"] == pytest.approx(1)
    assert bool(bob["new_account_7d"]) is True
    assert bool(bob["new_account_30d"]) is True
    assert bob["topic_tamil_rate"] == 0


def test_new_account_windows_are_inclusive():
    latest = pd.Timestamp("2026-06-01T00:00:00Z")
    rows = pd.DataFrame([
        activity_row(id="anchor", author="anchor", created_dt=latest),
        activity_row(id="exact7", author="exact7", created_dt=latest - pd.Timedelta(days=7)),
        activity_row(id="over7", author="over7", created_dt=latest - pd.Timedelta(days=7, seconds=1)),
        activity_row(id="exact30", author="exact30", created_dt=latest - pd.Timedelta(days=30)),
        activity_row(id="over30", author="over30", created_dt=latest - pd.Timedelta(days=30, seconds=1)),
    ])

    feat = features.build_account_features(rows, CFG)

    assert bool(feat.loc["exact7", "new_account_7d"]) is True
    assert bool(feat.loc["over7", "new_account_7d"]) is False
    assert bool(feat.loc["exact30", "new_account_30d"]) is True
    assert bool(feat.loc["over30", "new_account_30d"]) is False
    assert bool(feat.loc["over7", "new_account_30d"]) is True


def test_topic_patterns_use_word_boundaries_and_escape_punctuation():
    rows = pd.DataFrame([
        activity_row(id="1", author="a", text="The islamic calendar"),
        activity_row(id="2", author="b", text="Talk about islam today"),
        activity_row(id="3", author="c", text="see a.b now"),
        activity_row(id="4", author="d", text="see axb now"),
        activity_row(id="5", author="e", text="tamils and tamil"),
    ])

    feat = features.build_account_features(rows, CFG)

    assert feat.loc["a", "topic_islam_only_rate"] == 0
    assert feat.loc["b", "topic_islam_only_rate"] == 1
    assert feat.loc["c", "topic_dotted_rate"] == 1
    assert feat.loc["d", "topic_dotted_rate"] == 0
    assert feat.loc["e", "topic_tamil_rate"] == 1


def test_posts_without_link_id_have_zero_threads():
    rows = pd.DataFrame([
        activity_row(id="p1", author="alice", kind="post", link_id=None),
        activity_row(id="p2", author="alice", kind="post", link_id=float("nan")),
    ])

    feat = features.build_account_features(rows, CFG)

    assert feat.loc["alice", "thread_count"] == 0


def test_interaction_edges_follow_parent_comments_only():
    rows = pd.DataFrame([
        activity_row(id="c_parent", author="bob", kind="comment", parent_id=None),
        activity_row(id="c1", author="alice", kind="comment", parent_id="t1_c_parent"),
        activity_row(id="c2", author="alice", kind="comment", parent_id="t1_c_parent"),
        activity_row(id="c_self", author="bob", kind="comment", parent_id="t1_c_parent"),
        activity_row(id="c_post", author="alice", kind="comment", parent_id="t3_post"),
        activity_row(id="c_missing", author="alice", kind="comment", parent_id="t1_missing"),
        activity_row(id="p1", author="carol", kind="post", parent_id="t1_c_parent"),
    ])

    edges = features.build_interaction_edges(rows)

    assert edges.to_dict("records") == [
        {"author": "alice", "parent_author": "bob", "weight": 2},
    ]


def test_interaction_edges_empty_when_there_are_no_comments():
    rows = pd.DataFrame([
        activity_row(id="p1", author="carol", kind="post", parent_id=None),
    ])

    edges = features.build_interaction_edges(rows)

    assert list(edges.columns) == ["author", "parent_author", "weight"]
    assert edges.empty


def test_shared_thread_edges_weight_direction_and_size_cap():
    base = stamp(1_700_000_000)
    small = [
        activity_row(id="1", author="zara", link_id="t3_one", created_dt=base),
        activity_row(id="2", author="amy", link_id="t3_one", created_dt=base),
        activity_row(id="3", author="amy", link_id="t3_two", created_dt=base),
        activity_row(id="4", author="zara", link_id="t3_two", created_dt=base),
        activity_row(id="5", author="amy", link_id="t3_solo", created_dt=base),
        activity_row(id="6", author="[deleted]", link_id="t3_solo", created_dt=base),
        activity_row(id="7", author="amy", link_id=None, created_dt=base),
    ]
    huge = [
        activity_row(id=f"h{i}", author=f"user{i:02d}", link_id="t3_huge", created_dt=base)
        for i in range(81)
    ]
    capped = [
        activity_row(id=f"k{i}", author=f"keep{i:02d}", link_id="t3_cap", created_dt=base)
        for i in range(80)
    ]

    edges = features.build_shared_thread_edges(pd.DataFrame(small + huge + capped))
    pair = edges.set_index(["source", "target"])

    assert pair.loc[("amy", "zara"), "weight"] == 2
    assert ("zara", "amy") not in pair.index
    assert not pair.index.to_series().map(lambda ix: "user00" in ix).any()
    cap_edges = pair.index.to_series().map(lambda ix: ix[0].startswith("keep") and ix[1].startswith("keep"))
    assert int(cap_edges.sum()) == 80 * 79 // 2


def test_shared_thread_edges_empty_frame():
    rows = pd.DataFrame([
        activity_row(id="1", author="amy", link_id="t3_solo"),
    ])

    edges = features.build_shared_thread_edges(rows)

    assert list(edges.columns) == ["source", "target", "weight"]
    assert edges.empty
