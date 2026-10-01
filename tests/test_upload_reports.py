import pandas as pd

import upload_reports


def write_reports(root):
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True)
    (reports / "report.html").write_text("<html>report</html>", encoding="utf-8")
    (figures / "daily_activity.png").write_bytes(b"png")
    pd.DataFrame({
        "activity_count": [4],
        "post_count": [1],
        "comment_count": [3],
        "new_account_7d": [True],
        "new_account_30d": [False],
        "median_interarrival_min": [1.5],
        "max_subreddit_share": [0.5],
        "coordination_score": [0.812],
    }, index=pd.Index(["a|b"], name="author")).to_csv(reports / "candidate_accounts.csv")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"author": ["a", "a", "b"]}).to_parquet(processed / "activity.parquet", index=False)
    pd.DataFrame({"id": [1]}).to_parquet(processed / "near_duplicate_pairs.parquet", index=False)
    return reports, processed


def test_results_markdown_includes_stats_candidates_and_existing_figures(tmp_path):
    reports, processed = write_reports(tmp_path)
    text = upload_reports.results_markdown(
        {"subreddits": ["ask_srilanka", "srilanka"]},
        tmp_path,
        reports,
        processed,
        preview_rows=5,
    )

    assert "r/ask_srilanka, r/srilanka" in text
    assert "Activity rows: 3" in text
    assert "Accounts: 2" in text
    assert "Near-duplicate cross-account pairs: 1" in text
    assert "a\\|b" in text
    assert "0.812" in text
    assert "![Daily activity](reports/figures/daily_activity.png)" in text
    assert "Interaction network" not in text
    assert "not a determination that an account is a bot" in text


def test_merge_card_keeps_front_matter_and_replaces_old_results():
    card = """---
configs:
- config_name: posts
---

# Dataset

## Counts

- Posts: 1

## Results

old table
"""
    updated = upload_reports.merge_card(card, "## Results\n\nnew table\n")

    assert "config_name: posts" in updated
    assert "## Counts" in updated
    assert "- Posts: 1" in updated
    assert "old table" not in updated
    assert updated.strip().endswith("new table")


def test_stage_report_upload_copies_report_files(tmp_path):
    reports, _processed = write_reports(tmp_path)
    staging = tmp_path / "stage"

    upload_reports.stage_report_upload(tmp_path, reports, staging, "card text")

    assert (staging / "README.md").read_text() == "card text"
    assert (staging / "reports" / "report.html").read_text() == "<html>report</html>"
    assert (staging / "reports" / "figures" / "daily_activity.png").read_bytes() == b"png"
    assert not (staging / "reports" / "figures" / "topic_rates.png").exists()


def test_upload_report_folder_sends_the_staging_directory():
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)

    upload_reports.upload_report_folder(
        "Oshan/sri-lanka-reddit",
        "/tmp/stage",
        token="hf_test",
        commit_message="Add analysis report",
        api=FakeApi(),
    )

    assert calls == [{
        "folder_path": "/tmp/stage",
        "repo_id": "Oshan/sri-lanka-reddit",
        "repo_type": "dataset",
        "commit_message": "Add analysis report",
        "token": "hf_test",
    }]
