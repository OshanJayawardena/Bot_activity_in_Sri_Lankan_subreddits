import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def safe_author(x):
    if pd.isna(x) or not x:
        return "[deleted]"
    return str(x)

def normalize_activity(df):
    """Attach author, timestamp, and text columns to a raw activity frame."""
    df = df.copy()
    df["author"] = df["author"].map(safe_author)
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df["text"] = ""
    if "title" in df:
        df.loc[df["kind"].eq("post"), "text"] = df.loc[df["kind"].eq("post"), "title"].fillna("")
    if "selftext" in df:
        m = df["kind"].eq("post")
        df.loc[m, "text"] = df.loc[m, "text"] + "\n" + df.loc[m, "selftext"].fillna("")
    if "body" in df:
        m = df["kind"].eq("comment")
        df.loc[m, "text"] = df.loc[m, "body"].fillna("")
    df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def load_raw(raw_dir="data/raw"):
    raw_dir = Path(raw_dir)
    paths = list(raw_dir.glob("*_posts.parquet")) + list(raw_dir.glob("*_comments.parquet"))
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["kind"] = "post" if "_posts" in p.name else "comment"
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw parquet files. Run collect_arctic.py first.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_activity(df)

def build_account_features(valid, cfg):
    """Account-level signals from rows whose author is not ``[deleted]``."""
    valid = valid.sort_values(["author", "created_dt"])
    first_seen = valid.groupby("author")["created_dt"].min().rename("author_first_seen")
    last_seen = valid.groupby("author")["created_dt"].max().rename("author_last_seen")
    counts = valid.groupby("author").agg(
        activity_count=("id", "count"),
        post_count=("kind", lambda s: int((s == "post").sum())),
        comment_count=("kind", lambda s: int((s == "comment").sum())),
        subreddit_count=("subreddit", "nunique"),
        thread_count=("link_id", lambda s: s.nunique() if s.notna().any() else 0),
        mean_score=("score", "mean"),
    )
    features = counts.join(first_seen).join(last_seen)
    now_ref = valid["created_dt"].max()
    features["observed_span_days"] = (features["author_last_seen"] - features["author_first_seen"]).dt.total_seconds() / 86400
    features["activity_per_observed_day"] = features["activity_count"] / features["observed_span_days"].clip(lower=1/24)
    features["new_account_7d"] = ((now_ref - features["author_first_seen"]).dt.total_seconds() / 86400 <= cfg["analysis"]["very_new_account_days"])
    features["new_account_30d"] = ((now_ref - features["author_first_seen"]).dt.total_seconds() / 86400 <= cfg["analysis"]["new_account_days"])

    pivot = pd.crosstab(valid["author"], valid["subreddit"])
    shares = pivot.div(pivot.sum(axis=1), axis=0).max(axis=1).rename("max_subreddit_share")
    features = features.join(shares)

    delta = valid.groupby("author")["created_dt"].diff().dt.total_seconds() / 60
    burst = delta.groupby(valid["author"]).median().rename("median_interarrival_min")
    features = features.join(burst)

    topics = cfg.get("topics", {})
    lowtext = valid["text"].str.lower()
    for topic, words in topics.items():
        pat = "|".join(pd.Series(words).astype(str).str.replace(r"([.^$*+?{}\[\]\\|()])", r"\\\1", regex=True))
        hit = lowtext.str.contains(r"\b(?:" + pat + r")\b", regex=True, na=False)
        rates = hit.groupby(valid["author"]).mean().rename(f"topic_{topic}_rate")
        features = features.join(rates)

    features = features.fillna({
        "median_interarrival_min": np.nan,
        "max_subreddit_share": 0,
    })
    return features

def build_interaction_edges(valid):
    """Reply edges from a comment author to the parent comment's author."""
    comments = valid[valid["kind"].eq("comment")].copy()
    author_by_id = valid.set_index("id")["author"].to_dict()
    comments["parent_id_clean"] = comments["parent_id"].astype(str).str.replace(r"^t1_", "", regex=True)
    comments["parent_author"] = comments["parent_id_clean"].map(author_by_id)
    edges = comments.dropna(subset=["parent_author"])
    edges = edges[edges["parent_author"] != edges["author"]]
    return edges.groupby(["author", "parent_author"]).size().reset_index(name="weight")

def build_shared_thread_edges(valid):
    """Undirected co-participation edges. Threads larger than 80 authors are skipped."""
    valid = valid.sort_values(["author", "created_dt"])
    thread_edges = []
    for link_id, g in valid.groupby("link_id", dropna=True):
        authors = [a for a in g["author"].unique() if a != "[deleted]"]
        if len(authors) < 2 or len(authors) > 80:
            continue
        for i, a in enumerate(authors):
            for b in authors[i+1:]:
                thread_edges.append((a, b, 1))
    if thread_edges:
        te = pd.DataFrame(thread_edges, columns=["source", "target", "weight"])
        return te.groupby(["source", "target"], as_index=False)["weight"].sum()
    return pd.DataFrame(columns=["source", "target", "weight"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    df = load_raw()
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "activity.parquet", index=False)

    valid = df[~df["author"].eq("[deleted]")].copy()
    features = build_account_features(valid, cfg)
    features.to_csv(out / "account_features.csv")
    print(f"Saved {len(features):,} account feature rows.")
    print(features.sort_values(["new_account_7d", "activity_count"], ascending=False).head(25).to_string())

    valid = valid.sort_values(["author", "created_dt"])
    edge_counts = build_interaction_edges(valid)
    edge_counts.to_csv(out / "interaction_edges.csv", index=False)

    te = build_shared_thread_edges(valid)
    te.to_csv(out / "shared_thread_edges.csv", index=False)

if __name__ == "__main__":
    main()
