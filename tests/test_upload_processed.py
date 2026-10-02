import pandas as pd

import upload_processed
import upload_reports


def write_processed(root):
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"author": ["a", "b"]}).to_parquet(processed / "activity.parquet", index=False)
    pd.DataFrame({"weight": [2, 3]}).to_csv(processed / "interaction_edges.csv", index=False)
    return processed


def test_processed_markdown_counts_existing_files(tmp_path):
    processed = write_processed(tmp_path)
    text = upload_processed.processed_markdown(processed, ["activity.parquet", "interaction_edges.csv"])

    assert "activity.parquet" in text
    assert "2 rows" in text
    assert "does not recompute" in text


def test_replace_section_keeps_results_when_processed_files_are_added():
    card = """---
configs:
- config_name: posts
---

# Dataset

## Results

signal table
"""
    updated = upload_reports.replace_section(card, "## Processed files", "## Processed files\n\nactivity.parquet\n")

    assert "config_name: posts" in updated
    assert "signal table" in updated
    assert "activity.parquet" in updated

    again = upload_reports.replace_section(updated, "## Processed files", "## Processed files\n\nedges.csv\n")

    assert "signal table" in again
    assert "activity.parquet" not in again
    assert "edges.csv" in again


def test_stage_processed_upload_copies_only_files_already_on_disk(tmp_path):
    processed = write_processed(tmp_path)
    staging = tmp_path / "stage"

    names = upload_processed.stage_processed_upload(processed, staging, "card text")

    assert names == ["activity.parquet", "interaction_edges.csv"]
    assert (staging / "README.md").read_text() == "card text"
    assert (staging / "processed" / "activity.parquet").exists()
    assert not (staging / "processed" / "semantic_clusters.parquet").exists()


def test_stage_processed_upload_refuses_to_invent_missing_outputs(tmp_path):
    empty = tmp_path / "data" / "processed"
    empty.mkdir(parents=True)

    try:
        upload_processed.stage_processed_upload(empty, tmp_path / "stage", "card")
    except FileNotFoundError as exc:
        assert "No processed files" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_upload_does_not_delete_other_dataset_files():
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)

    upload_processed.upload_report_folder(
        "Oshan/sri-lanka-reddit",
        "/tmp/stage",
        token="hf_test",
        commit_message="Upload existing processed analysis files",
        api=FakeApi(),
    )

    assert "delete_patterns" not in calls[0]
    assert calls[0]["repo_type"] == "dataset"
