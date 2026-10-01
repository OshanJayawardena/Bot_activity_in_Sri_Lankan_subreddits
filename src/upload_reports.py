"""Upload generated reports and show their results on the dataset card.

Existing posts and comments in the dataset repo are left in place. Authenticate
with the Colab secret HF_TOKEN or the HF_TOKEN environment variable.

Example:

    python src/upload_reports.py --repo your-username/sri-lanka-reddit
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from upload_hf import load_config, resolve_token

FIGURES = (
    ("Daily activity", "reports/figures/daily_activity.png"),
    ("New-account share", "reports/figures/new_account_share.png"),
    ("Topic rates", "reports/figures/topic_rates.png"),
    ("Interaction network", "reports/figures/interaction_network.png"),
)
RESULTS_HEADING = "## Results"


def markdown_cell(value):
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_number(value, spec):
    if pd.isna(value):
        return ""
    return format(float(value), spec)


def result_stats(processed_dir):
    processed_dir = Path(processed_dir)
    lines = []
    activity_path = processed_dir / "activity.parquet"
    if activity_path.exists():
        activity = pd.read_parquet(activity_path, columns=["author"])
        lines.append(f"- Activity rows: {len(activity):,}")
        lines.append(f"- Accounts: {activity['author'].nunique():,}")
    for label, name in (
        ("Near-duplicate cross-account pairs", "near_duplicate_pairs.parquet"),
        ("Semantic-clustered rows", "semantic_clusters.parquet"),
    ):
        path = processed_dir / name
        if path.exists():
            lines.append(f"- {label}: {len(pd.read_parquet(path)):,}")
    summary_path = processed_dir / "semantic_cluster_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if len(summary):
            lines.append(f"- Semantic clusters: {len(summary):,}")
    return lines


def candidate_table(candidates, limit=20):
    preview = candidates.head(limit)
    rows = [
        "| Account | Activity | Posts | Comments | New ≤7d | New ≤30d | Median gap (min) | Max subreddit share | Signal score |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for account, row in preview.iterrows():
        rows.append(
            "| "
            + " | ".join([
                markdown_cell(account),
                format_number(row.get("activity_count"), ".0f"),
                format_number(row.get("post_count"), ".0f"),
                format_number(row.get("comment_count"), ".0f"),
                markdown_cell(row.get("new_account_7d")),
                markdown_cell(row.get("new_account_30d")),
                format_number(row.get("median_interarrival_min"), ".1f"),
                format_number(row.get("max_subreddit_share"), ".2f"),
                format_number(row.get("coordination_score"), ".3f"),
            ])
            + " |"
        )
    return "\n".join(rows)


def results_markdown(cfg, root, reports_dir, processed_dir, preview_rows=20):
    root = Path(root)
    reports_dir = Path(reports_dir)
    subreddits = ", ".join("r/" + str(name) for name in cfg.get("subreddits", []))
    candidates = pd.read_csv(reports_dir / "candidate_accounts.csv", index_col=0)
    figures = []
    for title, path in FIGURES:
        if (root / path).exists():
            figures.append(f"### {title}\n\n![{title}]({path})")
    stats = "\n".join(result_stats(processed_dir))
    return f"""{RESULTS_HEADING}

Signal scores are observable coordination signals for manual review. They are not a determination that an account is a bot.

Subreddits: {subreddits}.

{stats}

Full report: [reports/report.html](reports/report.html)

Candidate table: [reports/candidate_accounts.csv](reports/candidate_accounts.csv)

### Highest signal scores

{candidate_table(candidates, preview_rows)}

{chr(10).join(figures)}
"""


def merge_card(card, results):
    """Keep the existing dataset-card front matter and replace any Results section."""
    results = results.strip() + "\n"
    if card.startswith("---"):
        end = card.find("\n---", 3)
        if end != -1:
            front = card[: end + 4]
            body = card[end + 4 :]
        else:
            front, body = "", card
    else:
        front, body = "", card
    marker = f"\n{RESULTS_HEADING}\n"
    if body.lstrip().startswith(RESULTS_HEADING):
        body = ""
    elif marker in body:
        body = body[: body.index(marker)]
    return front + body.rstrip() + "\n\n" + results


def stage_report_upload(root, reports_dir, staging_dir, card_text):
    root = Path(root)
    reports_dir = Path(reports_dir)
    staging_dir = Path(staging_dir)
    if not (reports_dir / "report.html").exists():
        raise FileNotFoundError(f"{reports_dir / 'report.html'} is missing. Run src/report.py first.")
    if not (reports_dir / "candidate_accounts.csv").exists():
        raise FileNotFoundError(f"{reports_dir / 'candidate_accounts.csv'} is missing. Run src/report.py first.")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    destination = staging_dir / "reports"
    destination.mkdir(parents=True)
    for name in ("report.html", "candidate_accounts.csv"):
        shutil.copy2(reports_dir / name, destination / name)
    for _title, path in FIGURES:
        source = root / path
        if source.exists():
            target = staging_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (staging_dir / "README.md").write_text(card_text, encoding="utf-8")
    return staging_dir


def fetch_dataset_card(repo_id, token):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
    return Path(path).read_text(encoding="utf-8")


def upload_report_folder(repo_id, staging_dir, *, token, commit_message, api):
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )


def main(argv=None):
    args = parse_args(argv)
    token = resolve_token()
    root = Path(args.root)
    cfg = load_config(args.config)
    reports_dir = Path(args.reports_dir) if args.reports_dir else root / "reports"
    processed_dir = Path(args.processed_dir) if args.processed_dir else root / "data" / "processed"
    staging_dir = Path(args.staging_dir) if args.staging_dir else root / "data" / "report_upload"
    results = results_markdown(cfg, root, reports_dir, processed_dir)
    card = fetch_dataset_card(args.repo, token)
    updated = merge_card(card, results)
    stage_report_upload(root, reports_dir, staging_dir, updated)
    from huggingface_hub import HfApi
    upload_report_folder(
        args.repo,
        staging_dir,
        token=token,
        commit_message=args.commit_message,
        api=HfApi(),
    )
    print(f"Uploaded reports to https://huggingface.co/datasets/{args.repo}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Dataset id, for example username/sri-lanka-reddit")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--staging-dir", default=None)
    parser.add_argument("--commit-message", default="Add analysis report and show results on the dataset card")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
