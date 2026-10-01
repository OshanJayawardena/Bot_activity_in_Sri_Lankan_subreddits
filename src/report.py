import argparse
from pathlib import Path
import html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import yaml

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def daily_activity_counts(act):
    return act.set_index("created_dt").resample("D").size()

def new_account_daily_shares(act, very_new_days, new_days):
    """Share of each day's rows whose author was first seen within the window.

    Age is measured from the author's first row in this frame, not from
    Reddit account creation.
    """
    acct_first = act.groupby("author")["created_dt"].min()
    first_map = act["author"].map(acct_first)
    age_days = (act["created_dt"] - first_map).dt.total_seconds() / 86400
    daily = act.assign(age_days=age_days).set_index("created_dt").resample("D")["age_days"].agg(
        total="size",
        new_7d=lambda x: (x <= very_new_days).sum(),
        new_30d=lambda x: (x <= new_days).sum(),
    )
    daily["share_7d"] = daily["new_7d"] / daily["total"].replace(0, np.nan)
    daily["share_30d"] = daily["new_30d"] / daily["total"].replace(0, np.nan)
    return daily

def interaction_graph(edges):
    """Directed reply graph. Edges with weight below 2 are dropped."""
    G = nx.DiGraph()
    if edges is None or len(edges) == 0:
        return G
    e = edges[edges["weight"] >= 2]
    for _, r in e.iterrows():
        G.add_edge(r["author"], r["parent_author"], weight=float(r["weight"]))
    return G

def top_degree_subgraph(G, n=50):
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n]
    return G.subgraph([node for node, _ in top_nodes]).copy()

def coordination_frame(feat):
    """Transparent signal score. This is not a bot classification."""
    f = feat.copy()
    f["new7"] = f["new_account_7d"].astype(bool).astype(int)
    f["new30"] = f["new_account_30d"].astype(bool).astype(int)
    f["burst_signal"] = 1 / (1 + f["median_interarrival_min"].fillna(999) / 10)
    f["concentration_signal"] = f["max_subreddit_share"].fillna(0)
    f["volume_signal"] = np.log1p(f["activity_count"]) / np.log1p(max(f["activity_count"].max(), 2))
    f["coordination_score"] = (
        0.30 * f["new7"] +
        0.20 * f["new30"] +
        0.20 * f["burst_signal"] +
        0.15 * f["concentration_signal"] +
        0.15 * f["volume_signal"]
    )
    return f

def rank_candidates(feat, limit=100):
    return coordination_frame(feat).sort_values("coordination_score", ascending=False).head(limit)

def candidate_rows_html(candidates, limit=40):
    rows = []
    for username, r in candidates.head(limit).iterrows():
        rows.append(
            "<tr>" +
            f"<td>{html.escape(str(username))}</td>" +
            f"<td>{r['activity_count']:.0f}</td>" +
            f"<td>{r['post_count']:.0f}</td>" +
            f"<td>{r['comment_count']:.0f}</td>" +
            f"<td>{r['new_account_7d']}</td>" +
            f"<td>{r['new_account_30d']}</td>" +
            f"<td>{r['median_interarrival_min']:.2f}</td>" +
            f"<td>{r['max_subreddit_share']:.2f}</td>" +
            f"<td>{r['coordination_score']:.3f}</td></tr>"
        )
    return rows

def figures_html(out, imgs):
    return "".join(
        f'<h2>{html.escape(title)}</h2><img src="{path}" style="max-width:100%;"><br>'
        for title, path in imgs if (Path(out) / path).exists()
    )

def render_report(cfg, n_activity, n_accounts, n_dup, n_clusters, rows, img_html):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sri Lanka Reddit coordination analysis</title>
<style>
body {{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.5}}
table {{border-collapse:collapse;width:100%;font-size:13px}}
th,td {{border:1px solid #ddd;padding:6px;text-align:left}}
th {{background:#eee}}
.note {{background:#f5f5f5;padding:15px;border-left:4px solid #777}}
img {{border:1px solid #ddd}}
</style></head>
<body>
<h1>Sri Lanka Reddit Coordination Analysis</h1>
<p>Subreddits: {", ".join("r/"+x for x in cfg["subreddits"])}</p>
<div class="note"><b>Interpretation:</b> Candidate scores indicate observable
coordination-related signals. They are not a determination that an account is
a bot, sockpuppet, or coordinated actor.</div>

<h2>Dataset</h2>
<ul>
<li>Activity rows: {n_activity:,}</li>
<li>Accounts: {n_accounts:,}</li>
<li>Near-duplicate cross-account pairs: {n_dup:,}</li>
<li>Semantic-clustered rows: {n_clusters:,}</li>
</ul>

{img_html}

<h2>Candidate accounts for manual inspection</h2>
<table><thead><tr>
<th>Account</th><th>Activity</th><th>Posts</th><th>Comments</th>
<th>New ≤7d</th><th>New ≤30d</th><th>Median gap (min)</th>
<th>Max subreddit share</th><th>Signal score</th>
</tr></thead><tbody>
{"".join(rows)}
</tbody></table>

<h2>How to investigate a candidate</h2>
<ol>
<li>Check whether the account is genuinely new on Reddit, rather than merely
new to the three-subreddit dataset.</li>
<li>Open the candidate's comments and compare their timing with other accounts.</li>
<li>Inspect the near-duplicate pairs in
<code>data/processed/near_duplicate_pairs.parquet</code>.</li>
<li>Inspect semantic clusters in
<code>data/processed/semantic_clusters.parquet</code>.</li>
<li>Look for repeated interaction between the same accounts.</li>
<li>Consider breaking-news events: genuine users can naturally converge on the
same wording and timing.</li>
</ol>
</body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out = Path("reports")
    out.mkdir(exist_ok=True)
    img = out / "figures"
    img.mkdir(exist_ok=True)

    act = pd.read_parquet("data/processed/activity.parquet")
    feat = pd.read_csv("data/processed/account_features.csv", index_col=0)
    dup = pd.read_parquet("data/processed/near_duplicate_pairs.parquet")
    clusters = pd.read_parquet("data/processed/semantic_clusters.parquet")

    ts = daily_activity_counts(act)
    plt.figure(figsize=(11, 4))
    plt.plot(ts.index, ts.values)
    plt.title("Daily Reddit activity in the three target subreddits")
    plt.xlabel("Date")
    plt.ylabel("Posts + comments")
    savefig(img / "daily_activity.png")

    daily = new_account_daily_shares(
        act,
        cfg["analysis"]["very_new_account_days"],
        cfg["analysis"]["new_account_days"],
    )
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index, daily["share_7d"], label="Account first seen <= 7 days earlier")
    plt.plot(daily.index, daily["share_30d"], label="Account first seen <= 30 days earlier")
    plt.ylim(0, 1)
    plt.title("Share of activity from recently observed accounts")
    plt.xlabel("Date")
    plt.ylabel("Share")
    plt.legend()
    savefig(img / "new_account_share.png")

    topic_cols = [c for c in feat.columns if c.startswith("topic_") and c.endswith("_rate")]
    if topic_cols:
        vals = feat[topic_cols].mean().sort_values()
        plt.figure(figsize=(9, 5))
        plt.barh([x.replace("topic_", "").replace("_rate", "") for x in vals.index], vals.values)
        plt.title("Mean topic-keyword hit rate across accounts")
        plt.xlabel("Fraction of account activity")
        savefig(img / "topic_rates.png")

    edges_path = Path("data/processed/interaction_edges.csv")
    G = nx.DiGraph()
    if edges_path.exists():
        G = interaction_graph(pd.read_csv(edges_path))
    if len(G):
        H = top_degree_subgraph(G, 50)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(H, seed=42, k=1.2)
        nx.draw_networkx_nodes(H, pos, node_size=100)
        nx.draw_networkx_edges(H, pos, alpha=0.35, arrows=True)
        nx.draw_networkx_labels(H, pos, font_size=6)
        plt.title("Interaction network: top 50 accounts by degree")
        plt.axis("off")
        savefig(img / "interaction_network.png")

    candidates = rank_candidates(feat, 100)
    candidates.to_csv("reports/candidate_accounts.csv")
    rows = candidate_rows_html(candidates, 40)

    imgs = [
        ("Daily activity", "figures/daily_activity.png"),
        ("New-account share", "figures/new_account_share.png"),
        ("Topic rates", "figures/topic_rates.png"),
        ("Interaction network", "figures/interaction_network.png"),
    ]
    img_html = figures_html(out, imgs)
    report = render_report(
        cfg,
        len(act),
        act["author"].nunique(),
        len(dup),
        len(clusters),
        rows,
        img_html,
    )

    (out / "report.html").write_text(report, encoding="utf-8")
    print("Report:", out / "report.html")
    print("Candidates:", out / "candidate_accounts.csv")

if __name__ == "__main__":
    main()
