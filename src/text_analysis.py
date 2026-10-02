import argparse
from pathlib import Path
import re
import numpy as np
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

def sparse_topk_cosine(matrix, k, device, batch_size):
    """Top-k cosine neighbors for an L2-normalized sparse TF-IDF matrix.

    On CUDA the corpus stays sparse and each query batch is multiplied on the GPU.
    """
    import torch
    matrix = matrix.tocsr().astype(np.float32)
    n = matrix.shape[0]
    k = min(k, n)
    sparse = torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64, copy=False)),
        torch.from_numpy(matrix.indices.astype(np.int64, copy=False)),
        torch.from_numpy(np.ascontiguousarray(matrix.data)),
        size=matrix.shape,
        device=device,
    )
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        queries = torch.from_numpy(np.ascontiguousarray(matrix[start:stop].toarray())).to(device)
        similarity = torch.sparse.mm(sparse, queries.T).T
        values, neighbors = torch.topk(similarity, k=k, dim=1)
        indices[start:stop] = neighbors.detach().cpu().numpy()
        distances[start:stop] = (1 - values).detach().cpu().numpy()
        if start == 0 or stop == n or (start // batch_size) % 20 == 0:
            print(f"  compared {stop:,}/{n:,} texts on {device}", flush=True)
    return distances, indices


def cpu_neighbors(matrix, k, n_jobs):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=n_jobs)
    nn.fit(matrix)
    return nn.kneighbors(matrix)


def pairs_from_neighbor_arrays(act, indices, distances, threshold):
    """Cross-account pairs. A row is not paired with itself."""
    authors = act["author"].to_numpy()
    ids = act["id"].to_numpy()
    created = act["created_dt"].to_numpy()
    subreddits = act["subreddit"].to_numpy()
    texts = act["text"].to_numpy()
    similarity = 1 - distances
    row = np.arange(indices.shape[0])[:, None]
    mask = (similarity >= threshold) & (indices != row) & (authors[indices] != authors[:, None])
    left, column = np.nonzero(mask)
    right = indices[left, column]
    scores = similarity[left, column]
    pairs = []
    for a, b, score in zip(left, right, scores):
        pairs.append({
            "id_a": ids[a],
            "id_b": ids[b],
            "author_a": authors[a],
            "author_b": authors[b],
            "similarity": float(score),
            "created_a": created[a],
            "created_b": created[b],
            "subreddit_a": subreddits[a],
            "subreddit_b": subreddits[b],
            "text_a": texts[a],
            "text_b": texts[b],
        })
    return pairs


def find_near_duplicate_pairs(act, threshold, n_jobs=-1, device="cpu", batch_size=256):
    """Cross-account pairs whose character n-gram cosine similarity meets ``threshold``."""
    if len(act) < 2:
        return []
    print(f"Near-duplicate search on {device}: {len(act):,} texts", flush=True)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000)
    matrix = vectorizer.fit_transform(act["norm_text"])
    k = min(10, len(act))
    if device == "cuda":
        try:
            distances, indices = sparse_topk_cosine(matrix, k, device, batch_size)
        except Exception as exc:
            print(f"GPU near-duplicate search failed ({exc}). Using CPU.", flush=True)
            distances, indices = cpu_neighbors(matrix, k, n_jobs)
    else:
        distances, indices = cpu_neighbors(matrix, k, n_jobs)
    return pairs_from_neighbor_arrays(act, indices, distances, threshold)

def embedding_frame(act, nmax):
    """Latest ``nmax`` rows, which is the slice embedded in the pipeline."""
    return act.sort_values("created_dt").tail(nmax).reset_index(drop=True)

def cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())

def resolve_device(requested, cuda_is_available):
    """``auto`` uses CUDA when a GPU is visible, otherwise CPU."""
    requested = (requested or "auto").strip().lower()
    if requested in ("", "auto"):
        return "cuda" if cuda_is_available else "cpu"
    if requested == "cuda" and not cuda_is_available:
        return "cpu"
    return requested

def select_device(requested):
    device = resolve_device(requested, cuda_available())
    if str(requested or "auto").strip().lower() == "cuda" and device != "cuda":
        print("CUDA was requested but is not available. Using CPU.")
    return device

def encode_batch_size(configured, device):
    if configured:
        return int(configured)
    return 256 if device == "cuda" else 32

def encode_texts(texts, model_name, device, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    if len(texts) == 0:
        width = model.get_sentence_embedding_dimension()
        return np.empty((0, width), dtype=np.float32)
    return np.asarray(model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    ), dtype=np.float32)

def cosine_neighborhoods(embeddings, min_similarity, device, batch_size):
    """Indices within cosine similarity ``min_similarity``, including each point itself.

    Embeddings are normalized and compared with a matrix product on ``device``.
    """
    import torch
    matrix = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    n = matrix.shape[0]
    neighborhoods = [None] * n
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        similarity = matrix[start:stop] @ matrix.T
        local = torch.arange(stop - start, device=device)
        similarity[local, torch.arange(start, stop, device=device)] = 1
        hits = torch.nonzero(similarity >= min_similarity, as_tuple=False).detach().cpu().numpy()
        counts = np.bincount(hits[:, 0], minlength=stop - start) if len(hits) else np.zeros(stop - start, dtype=int)
        offset = 0
        for row, count in enumerate(counts):
            neighborhoods[start + row] = hits[offset:offset + count, 1]
            offset += count
    return neighborhoods

def labels_from_neighborhoods(neighborhoods, min_samples):
    """DBSCAN labels from precomputed neighborhoods. Noise is -1."""
    n = len(neighborhoods)
    labels = np.full(n, -1, dtype=np.int32)
    is_core = np.fromiter((len(neighbors) >= min_samples for neighbors in neighborhoods), dtype=bool, count=n)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            point = stack.pop()
            if not is_core[point]:
                continue
            for neighbor in neighborhoods[point]:
                neighbor = int(neighbor)
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster
                    stack.append(neighbor)
        cluster += 1
    return labels

def cluster_embeddings(embeddings, eps, min_samples, device, batch_size):
    """Cosine DBSCAN. CUDA computes neighborhoods on the GPU; CPU uses scikit-learn."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) == 0:
        return np.array([], dtype=np.int32)
    if device == "cuda":
        neighborhoods = cosine_neighborhoods(
            embeddings,
            min_similarity=1.0 - eps,
            device=device,
            batch_size=max(int(batch_size), 2048),
        )
        return labels_from_neighborhoods(neighborhoods, min_samples)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return clustering.fit_predict(embeddings)

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

    device = select_device(cfg["analysis"].get("device"))
    batch_size = encode_batch_size(cfg["analysis"].get("embedding_batch_size"), device)
    if device == "cuda":
        import torch
        print(f"Embedding device: cuda ({torch.cuda.get_device_name(0)}), batch size {batch_size}", flush=True)
    else:
        print(f"Embedding device: cpu, batch size {batch_size}", flush=True)

    act = pd.read_parquet("data/processed/activity.parquet")
    act = eligible_activity(act, cfg["analysis"]["min_text_chars"])

    threshold = cfg["analysis"]["near_duplicate_threshold"]
    pairs = find_near_duplicate_pairs(act, threshold, device=device, batch_size=max(batch_size, 128))
    pd.DataFrame(pairs).to_parquet("data/processed/near_duplicate_pairs.parquet", index=False)

    # Semantic embeddings. CUDA is used when a GPU is available.
    # The model import stays inside this function so unit tests never download it.
    nmax = cfg["analysis"]["max_embedding_rows"]
    emb_df = embedding_frame(act, nmax)
    print(f"Encoding {len(emb_df):,} texts on {device}", flush=True)
    embeddings = encode_texts(emb_df["text"].tolist(), "all-MiniLM-L6-v2", device, batch_size)
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    # DBSCAN on cosine distance. eps = 1 - similarity threshold.
    eps = 1 - cfg["analysis"]["semantic_similarity_threshold"]
    labels = cluster_embeddings(
        embeddings,
        eps=eps,
        min_samples=cfg["analysis"]["min_cluster_size"],
        device=device,
        batch_size=batch_size,
    )
    emb_df["semantic_cluster"] = labels
    emb_df.to_parquet("data/processed/semantic_clusters.parquet", index=False)

    cluster_accounts = summarize_semantic_clusters(emb_df)
    cluster_accounts.to_csv("data/processed/semantic_cluster_summary.csv", index=False)

    print(f"Near-duplicate cross-account pairs: {len(pairs):,}")
    print(f"Semantic rows analysed: {len(emb_df):,}")

if __name__ == "__main__":
    main()
