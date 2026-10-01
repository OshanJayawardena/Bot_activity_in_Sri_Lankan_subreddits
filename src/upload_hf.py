"""Upload scraped subreddit parquet files to a Hugging Face dataset.

The dataset is private unless --public is set. Authenticate with HF_TOKEN
(a write token from https://huggingface.co/settings/tokens). The token is read
from the environment and is not accepted as a command-line argument.

Example:

    export HF_TOKEN=hf_...
    python src/upload_hf.py --repo your-username/sri-lanka-reddit
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

KINDS = ("posts", "comments")

POST_COLUMNS = [
    "id", "author", "author_fullname", "created_utc", "retrieved_on", "subreddit",
    "subreddit_id", "score", "title", "selftext", "url", "link_flair_text",
    "num_comments", "crosspost_parent",
]
COMMENT_COLUMNS = [
    "id", "author", "author_fullname", "created_utc", "retrieved_on", "subreddit",
    "subreddit_id", "score", "body", "link_id", "parent_id", "distinguished",
]
PREFERRED_COLUMNS = {"posts": POST_COLUMNS, "comments": COMMENT_COLUMNS}


def subreddit_from_path(path, kind):
    suffix = f"_{kind}.parquet"
    name = Path(path).name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(path).stem


def raw_paths(raw_dir, kind):
    return sorted(Path(raw_dir).glob(f"*_{kind}.parquet"))


def load_kind(raw_dir, kind):
    """Concatenate one kind of raw parquet file. Empty files are skipped."""
    frames = []
    skipped = []
    for path in raw_paths(raw_dir, kind):
        frame = pd.read_parquet(path)
        if frame.empty:
            skipped.append(path.name)
            continue
        if "subreddit" not in frame.columns:
            frame = frame.copy()
            frame["subreddit"] = subreddit_from_path(path, kind)
        else:
            missing = frame["subreddit"].isna() | frame["subreddit"].astype(str).str.strip().eq("")
            if missing.any():
                frame = frame.copy()
                frame.loc[missing, "subreddit"] = subreddit_from_path(path, kind)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), skipped
    combined = pd.concat(frames, ignore_index=True)
    preferred = [column for column in PREFERRED_COLUMNS[kind] if column in combined.columns]
    extra = [column for column in combined.columns if column not in preferred]
    return combined[preferred + extra], skipped


def size_category(n_rows):
    bounds = [
        (1_000, "n<1K"),
        (10_000, "1K<n<10K"),
        (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"),
        (10_000_000, "1M<n<10M"),
    ]
    for limit, label in bounds:
        if n_rows < limit:
            return label
    return "n>10M"


def dataset_card(repo_id, cfg, counts, skipped):
    subreddits = cfg.get("subreddits") or sorted({
        name for kind in KINDS for name in counts.get(kind, pd.DataFrame()).get("subreddit", pd.Series(dtype=str)).dropna().unique()
    })
    collection = cfg.get("collection") or {}
    after = collection.get("after")
    before = collection.get("before")
    total_rows = sum(counts[kind] for kind in KINDS)
    config_blocks = []
    for kind in KINDS:
        if counts[kind] <= 0:
            continue
        config_blocks.append(
            f"- config_name: {kind}\n"
            f"  data_files:\n"
            f"  - split: train\n"
            f"    path: {kind}/*"
        )
    subreddit_lines = "\n".join(f"- r/{name}" for name in subreddits) or "- (see the subreddit column)"
    skipped_lines = "\n".join(f"- `{name}`" for name in skipped) or "- none"
    window = f"{after or 'unspecified'} to {before or 'the time of collection'}"
    return f"""---
license: other
language:
- en
tags:
- reddit
- sri-lanka
size_categories:
- {size_category(total_rows)}
configs:
{chr(10).join(config_blocks)}
---

# {repo_id.split("/")[-1]}

Public Reddit posts and comments collected from the Arctic Shift archive for later analysis.
This dataset does not label accounts as bots.

## Subreddits

{subreddit_lines}

## Collection window

{window}

## Counts

- Posts: {counts["posts"]:,}
- Comments: {counts["comments"]:,}

## Load

```python
from datasets import load_dataset

posts = load_dataset({json.dumps(repo_id)}, "posts", split="train")
comments = load_dataset({json.dumps(repo_id)}, "comments", split="train")
```

## Empty files skipped

{skipped_lines}

Usernames in this dataset are public Reddit account names. Do not use the data to infer real identities.
"""


def stage_dataset(raw_dir, staging_dir, cfg, repo_id):
    """Write a Hub dataset folder. Returns row counts and skipped filenames."""
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    skipped = []
    counts = {}
    for kind in KINDS:
        frame, kind_skipped = load_kind(raw_dir, kind)
        skipped.extend(kind_skipped)
        frames[kind] = frame
        counts[kind] = len(frame)
        if frame.empty:
            continue
        kind_dir = staging_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(kind_dir / "train-00000-of-00001.parquet", index=False)
    if counts["posts"] == 0 and counts["comments"] == 0:
        raise FileNotFoundError(
            f"No non-empty raw parquet files in {raw_dir}. Run collect_arctic.py first."
        )
    (staging_dir / "README.md").write_text(
        dataset_card(repo_id, cfg, counts, skipped),
        encoding="utf-8",
    )
    return {"counts": counts, "skipped": skipped}


def upload_staged(repo_id, staging_dir, *, private, token, commit_message, api):
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )


def resolve_token(token=None):
    resolved = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not resolved:
        raise RuntimeError(
            "Set HF_TOKEN to a Hugging Face write token before uploading."
        )
    return resolved


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def main(argv=None):
    args = parse_args(argv)
    token = resolve_token()
    cfg = load_config(args.config)
    staging = Path(args.staging_dir) if args.staging_dir else Path("data/hf_staging")
    summary = stage_dataset(args.raw_dir, staging, cfg, args.repo)
    from huggingface_hub import HfApi

    upload_staged(
        args.repo,
        staging,
        private=not args.public,
        token=token,
        commit_message=args.commit_message,
        api=HfApi(),
    )
    privacy = "public" if args.public else "private"
    print(f"Uploaded {privacy} dataset {args.repo}")
    print(f"Posts: {summary['counts']['posts']:,}")
    print(f"Comments: {summary['counts']['comments']:,}")
    if summary["skipped"]:
        print("Skipped empty files: " + ", ".join(summary["skipped"]))
    print(f"https://huggingface.co/datasets/{args.repo}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Dataset id, for example username/sri-lanka-reddit")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--staging-dir", default=None, help="Folder staged for upload. Defaults to data/hf_staging.")
    parser.add_argument("--public", action="store_true", help="Publish the dataset. The default upload is private.")
    parser.add_argument("--commit-message", default="Upload scraped subreddit activity")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
