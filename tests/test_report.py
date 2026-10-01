import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pytest

import report


def test_savefig_writes_and_closes(tmp_path):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    path = tmp_path / "plot.png"

    report.savefig(path)

    assert path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_daily_activity_counts_fill_empty_days():
    created = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-03"], utc=True)
    act = pd.DataFrame({"author": ["a", "b", "c"], "created_dt": created})

    counts = report.daily_activity_counts(act)

    assert list(counts.values) == [1, 0, 2]
    assert len(counts) == 3


def test_new_account_share_uses_first_seen_in_frame():
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    act = pd.DataFrame({
        "author": ["a", "a", "b"],
        "created_dt": [t0, t0 + pd.Timedelta(days=10), t0 + pd.Timedelta(days=10)],
    })

    daily = report.new_account_daily_shares(act, very_new_days=7, new_days=30)

    first = daily.iloc[0]
    assert first["total"] == 1
    assert first["share_7d"] == pytest.approx(1)
    assert first["share_30d"] == pytest.approx(1)

    gap = daily.iloc[1]
    assert gap["total"] == 0
    assert pd.isna(gap["share_7d"])
    assert pd.isna(gap["share_30d"])

    last = daily.iloc[-1]
    assert last["total"] == 2
    assert last["new_7d"] == 1
    assert last["new_30d"] == 2
    assert last["share_7d"] == pytest.approx(0.5)
    assert last["share_30d"] == pytest.approx(1)


def test_interaction_graph_drops_weight_below_two():
    edges = pd.DataFrame([
        {"author": "alice", "parent_author": "bob", "weight": 1},
        {"author": "alice", "parent_author": "cara", "weight": 2},
        {"author": "bob", "parent_author": "cara", "weight": 3},
    ])

    graph = report.interaction_graph(edges)

    assert list(graph.edges(data="weight")) == [
        ("alice", "cara", 2.0),
        ("bob", "cara", 3.0),
    ]
    assert report.interaction_graph(edges.iloc[0:0]).number_of_nodes() == 0
    assert report.interaction_graph(None).number_of_nodes() == 0


def test_top_degree_subgraph_keeps_highest_degree_nodes():
    graph = nx.DiGraph()
    graph.add_edge("hub", "a")
    graph.add_edge("hub", "b")
    graph.add_edge("hub", "c")
    graph.add_edge("a", "b")

    top = report.top_degree_subgraph(graph, n=1)

    assert list(top.nodes) == ["hub"]


def test_coordination_score_uses_documented_weights():
    feat = pd.DataFrame({
        "new_account_7d": [True, False],
        "new_account_30d": [True, True],
        "median_interarrival_min": [0.0, 10.0],
        "max_subreddit_share": [1.0, 0.0],
        "activity_count": [10, 1],
        "post_count": [1, 0],
        "comment_count": [9, 1],
    }, index=["new_user", "old_user"])

    scored = report.coordination_frame(feat)
    volume_old = math.log1p(1) / math.log1p(10)
    expected_old = 0.20 * 1 + 0.20 * 0.5 + 0.15 * 0 + 0.15 * volume_old

    assert scored.loc["new_user", "coordination_score"] == pytest.approx(1.0)
    assert scored.loc["old_user", "coordination_score"] == pytest.approx(expected_old)
    assert list(report.rank_candidates(feat, limit=100).index) == ["new_user", "old_user"]


def test_missing_interarrival_uses_filled_burst_and_volume_floor():
    feat = pd.DataFrame({
        "new_account_7d": [False],
        "new_account_30d": [False],
        "median_interarrival_min": [float("nan")],
        "max_subreddit_share": [float("nan")],
        "activity_count": [1],
        "post_count": [1],
        "comment_count": [0],
    }, index=["solo"])

    scored = report.coordination_frame(feat)
    burst = 1 / (1 + 999 / 10)
    volume = math.log1p(1) / math.log1p(2)
    expected = 0.20 * burst + 0.15 * volume

    assert scored.loc["solo", "burst_signal"] == pytest.approx(burst)
    assert scored.loc["solo", "concentration_signal"] == 0
    assert scored.loc["solo", "coordination_score"] == pytest.approx(expected)


def test_rank_candidates_limits_to_one_hundred():
    feat = pd.DataFrame({
        "new_account_7d": [False] * 105,
        "new_account_30d": [False] * 105,
        "median_interarrival_min": [float("nan")] * 105,
        "max_subreddit_share": [0.0] * 105,
        "activity_count": list(range(1, 106)),
        "post_count": [0] * 105,
        "comment_count": list(range(1, 106)),
    }, index=[f"user{i}" for i in range(1, 106)])

    ranked = report.rank_candidates(feat, limit=100)

    assert len(ranked) == 100
    assert ranked.index[0] == "user105"
    assert ranked["activity_count"].min() == 6


def test_candidate_rows_escape_usernames():
    candidates = report.coordination_frame(pd.DataFrame({
        "new_account_7d": [True],
        "new_account_30d": [False],
        "median_interarrival_min": [1.5],
        "max_subreddit_share": [0.5],
        "activity_count": [4],
        "post_count": [1],
        "comment_count": [3],
    }, index=["<bob&>"]))

    rows = report.candidate_rows_html(candidates)
    score = candidates.loc["<bob&>", "coordination_score"]

    assert len(rows) == 1
    assert "&lt;bob&amp;&gt;" in rows[0]
    assert "<bob" not in rows[0]
    assert "<td>4</td><td>1</td><td>3</td>" in rows[0]
    assert "<td>1.50</td>" in rows[0]
    assert "<td>0.50</td>" in rows[0]
    assert f"<td>{score:.3f}</td>" in rows[0]


def test_figures_html_skips_missing_files_and_escapes_titles(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "daily_activity.png").write_bytes(b"png")

    html = report.figures_html(tmp_path, [
        ("Day <1>", "figures/daily_activity.png"),
        ("Missing", "figures/nope.png"),
    ])

    assert "Day &lt;1&gt;" in html
    assert "daily_activity.png" in html
    assert "nope.png" not in html


def test_render_report_includes_counts_and_interpretation():
    html = report.render_report(
        {"subreddits": ["AskSriLanka", "srilanka", "Colombo"]},
        n_activity=1000,
        n_accounts=25,
        n_dup=3,
        n_clusters=4000,
        rows=["<tr><td>alice</td></tr>"],
        img_html="<h2>Daily activity</h2>",
    )

    assert "r/AskSriLanka, r/srilanka, r/Colombo" in html
    assert "Activity rows: 1,000" in html
    assert "Accounts: 25" in html
    assert "Near-duplicate cross-account pairs: 3" in html
    assert "Semantic-clustered rows: 4,000" in html
    assert "not a determination that an account is" in html
    assert "<td>alice</td>" in html
    assert "<h2>Daily activity</h2>" in html
