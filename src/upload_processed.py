"""Upload analysis files that are already in data/processed.

This does not rerun feature, text, or report scripts. Posts, comments, and
report files already in the dataset repo stay in place. Authenticate with the
Colab secret HF_TOKEN or the HF_TOKEN environment variable.

Example:

    python src/upload_processed.py --repo your-username/sri-lanka-reddit
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from upload_hf import resolve_token
from upload_reports import fetch_dataset_card, replace_section, upload_report_folder

PROCESSED_FILES = (
    "activity.parquet",
    "account_features.csv",
    "interaction_edges.csv",
    "shared_thread_edges.csv",
    "near_duplicate_pairs.parquet",
    "semantic_clusters.parquet",
    "semantic_cluster_summary.csv",
)
PROCESSED_HEADING = "## Processed files"


def row_count(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return len(pd.read_parquet(path))
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    return None


def processed_markdown(processed_dir, names):
    processed_dir = Path(processed_dir)
    lines = [
        PROCESSED_HEADING,
        "",
        "These files were uploaded from an existing analysis run. The upload does not recompute them.",
        "",
    ]
    for name in names:
        count = row_count(processed_dir / name)
        detail = f" — {count:,} rows" if count is not None else ""
        lines.append(f"- [{name}](processed/{name}){detail}")
    return "\n".join(lines) + "\n"


def stage_processed_upload(processed_dir, staging_dir, card_text):
    processed_dir = Path(processed_dir)
    staging_dir = Path(staging_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"{processed_dir} is missing. The processed files must already be on disk.")
    names = [name for name in PROCESSED_FILES if (processed_dir / name).exists()]
    if not names:
        raise FileNotFoundError(
            f"No processed files in {processed_dir}. Expected one of: {', '.join(PROCESSED_FILES)}."
        )
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    destination = staging_dir / "processed"
    destination.mkdir(parents=True)
    for name in names:
        shutil.copy2(processed_dir / name, destination / name)
    (staging_dir / "README.md").write_text(card_text, encoding="utf-8")
    return names


def main(argv=None):
    args = parse_args(argv)
    token = resolve_token()
    root = Path(args.root)
    processed_dir = Path(args.processed_dir) if args.processed_dir else root / "data" / "processed"
    staging_dir = Path(args.staging_dir) if args.staging_dir else root / "data" / "processed_upload"
    names = [name for name in PROCESSED_FILES if (processed_dir / name).exists()]
    if not names:
        raise FileNotFoundError(
            f"No processed files in {processed_dir}. Expected one of: {', '.join(PROCESSED_FILES)}."
        )
    card = fetch_dataset_card(args.repo, token)
    updated = replace_section(card, PROCESSED_HEADING, processed_markdown(processed_dir, names))
    uploaded = stage_processed_upload(processed_dir, staging_dir, updated)
    from huggingface_hub import HfApi
    upload_report_folder(
        args.repo,
        staging_dir,
        token=token,
        commit_message=args.commit_message,
        api=HfApi(),
    )
    print(f"Uploaded {len(uploaded)} existing processed files to https://huggingface.co/datasets/{args.repo}")
    for name in uploaded:
        print(f"- processed/{name}")
    missing = [name for name in PROCESSED_FILES if name not in uploaded]
    if missing:
        print("Left out because they are not on disk: " + ", ".join(missing))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Dataset id, for example username/sri-lanka-reddit")
    parser.add_argument("--root", default=".")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--staging-dir", default=None)
    parser.add_argument("--commit-message", default="Upload existing processed analysis files")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
