import argparse
from pathlib import Path
import re
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def norm_text(s):
    s = str(s).lower()
    s = re.sub(r"https?://\S+", " URL ", s)
    s = re.sub(r"\bu/\w+", " USER ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def eligible_activity(act, min_text_chars):
    act = act[(act["author"] != "[deleted]") & (act["text"].str.len() >= min_text_chars)].copy()
    act["norm_text"] = act["text"].map(norm_text)
    return act

def find_near_duplicate_pairs(act, threshold, n_jobs=-1):
    """Cross-account pairs whose character n-gram cosine similarity meets ``threshold``."""
    if len(act) < 2:
        return []
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000)
    X = vectorizer.fit_transform(act["norm_text"])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(10, len(act)), metric="cosine", n_jobs=n_jobs)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    pairs = []
    for i in range(len(act)):
        for j, d in zip(indices[i, 1:], distances[i, 1:]):
            sim = 1 - d
            if sim >= threshold and act.iloc[i]["author"] != act.iloc[j]["author"]:
                pairs.append({
                    "id_a": act.iloc[i]["id"],
                    "id_b": act.iloc[j]["id"],
                    "author_a": act.iloc[i]["author"],
                    "author_b": act.iloc[j]["author"],
                    "similarity": float(sim),
                    "created_a": act.iloc[i]["created_dt"],
                    "created_b": act.iloc[j]["created_dt"],
                    "subreddit_a": act.iloc[i]["subreddit"],
                    "subreddit_b": act.iloc[j]["subreddit"],
                    "text_a": act.iloc[i]["text"],
                    "text_b": act.iloc[j]["text"],
                })
    return pairs

def embedding_frame(act, nmax):
    """Latest ``nmax`` rows, which is the slice embedded in the pipeline."""
    return act.sort_values("created_dt").tail(nmax).reset_index(drop=True)

def summarize_semantic_clusters(emb_df):
    clustered = emb_df[emb_df["semantic_cluster"] >= 0]
    if len(clustered):
        return (
            clustered.groupby("semantic_cluster")
            .agg(
                cluster_size=("id", "count"),
                unique_accounts=("author", "nunique"),
                first_time=("created_dt", "min"),
                last_time=("created_dt", "max"),
                subreddits=("subreddit", lambda x: ",".join(sorted(set(x)))),
            )
            .reset_index()
        )
    return pd.DataFrame(columns=["semantic_cluster", "cluster_size", "unique_accounts"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    act = pd.read_parquet("data/processed/activity.parquet")
    act = eligible_activity(act, cfg["analysis"]["min_text_chars"])

    threshold = cfg["analysis"]["near_duplicate_threshold"]
    pairs = find_near_duplicate_pairs(act, threshold)
    pd.DataFrame(pairs).to_parquet("data/processed/near_duplicate_pairs.parquet", index=False)

    # Semantic embeddings on a bounded sample for practical CPU/GPU use.
    # Imported here so unit tests never load or download the embedding model.
    from sentence_transformers import SentenceTransformer
    nmax = cfg["analysis"]["max_embedding_rows"]
    emb_df = embedding_frame(act, nmax)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    E = model.encode(emb_df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    # DBSCAN on cosine distance. eps = 1 - similarity threshold.
    eps = 1 - cfg["analysis"]["semantic_similarity_threshold"]
    clustering = DBSCAN(eps=eps, min_samples=cfg["analysis"]["min_cluster_size"], metric="cosine", n_jobs=-1)
    labels = clustering.fit_predict(E)
    emb_df["semantic_cluster"] = labels
    emb_df.to_parquet("data/processed/semantic_clusters.parquet", index=False)

    cluster_accounts = summarize_semantic_clusters(emb_df)
    cluster_accounts.to_csv("data/processed/semantic_cluster_summary.csv", index=False)

    print(f"Near-duplicate cross-account pairs: {len(pairs):,}")
    print(f"Semantic rows analysed: {len(emb_df):,}")

if __name__ == "__main__":
    main()
