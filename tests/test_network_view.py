import pandas as pd

import network_view


def edges():
    return pd.DataFrame({
        "author": ["alice", "alice", "bob", "dave"],
        "parent_author": ["bob", "cara", "cara", "erin"],
        "weight": [4, 3, 5, 1],
    })


def test_select_graph_drops_light_edges_and_caps_accounts():
    selected = network_view.select_graph(edges(), min_weight=2, max_nodes=2)
    nodes = set(selected["source"]).union(selected["target"])

    assert "dave" not in nodes
    assert "erin" not in nodes
    assert len(nodes) <= 2
    assert selected["weight"].min() >= 2


def test_network_html_colors_new_accounts_and_escapes_names():
    frame = pd.DataFrame({
        "source": ["<alice>", "bob"],
        "target": ["bob", "cara"],
        "weight": [3, 2],
    })
    features = pd.DataFrame({
        "new_account_7d": [True, False, False],
        "new_account_30d": [True, False, False],
        "median_interarrival_min": [1.0, 10.0, 10.0],
        "max_subreddit_share": [1.0, 0.2, 0.2],
        "activity_count": [8, 2, 2],
        "coordination_score": [0.8, 0.1, 0.1],
    }, index=["<alice>", "bob", "cara"])

    html, selected = network_view.build_network(frame, features, min_weight=2, max_nodes=10)
    blob = html.split("const data = ", 1)[1].split(";", 1)[0]

    assert len(selected) == 2
    assert "<alice>" not in blob
    assert "\\u003calice>" in blob
    assert "#d85a30" in html
    assert "signal score" in html
    assert "accounts, " in html
    assert "only that account" in html
    assert "Replied to" in html
