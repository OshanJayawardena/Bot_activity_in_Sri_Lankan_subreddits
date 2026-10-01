import pandas as pd
import pytest

import upload_hf


def write_raw(raw_dir, name, rows):
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(raw_dir / name, index=False)


def test_stage_dataset_splits_posts_and_comments(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw, "srilanka_posts.parquet", [
        {"id": "p1", "author": "alice", "created_utc": 10, "subreddit": "srilanka", "title": "Hello"},
    ])
    write_raw(raw, "Colombo_posts.parquet", [
        {"id": "p2", "author": "bob", "created_utc": 11, "title": "Road"},
    ])
    write_raw(raw, "srilanka_comments.parquet", [
        {"id": "c1", "author": "cara", "created_utc": 12, "subreddit": "srilanka", "body": "A comment"},
    ])
    write_raw(raw, "AskSriLanka_posts.parquet", [])
    cfg = {"subreddits": ["AskSriLanka", "srilanka", "Colombo"], "collection": {"after": "2026-01-01", "before": None}}

    summary = upload_hf.stage_dataset(raw, tmp_path / "stage", cfg, "owner/sri-lanka-reddit")

    posts = pd.read_parquet(tmp_path / "stage" / "posts" / "train-00000-of-00001.parquet")
    comments = pd.read_parquet(tmp_path / "stage" / "comments" / "train-00000-of-00001.parquet")
    card = (tmp_path / "stage" / "README.md").read_text()

    assert summary["counts"] == {"posts": 2, "comments": 1}
    assert summary["skipped"] == ["AskSriLanka_posts.parquet"]
    assert list(posts["id"]) == ["p2", "p1"]
    assert posts.loc[posts["id"].eq("p2"), "subreddit"].item() == "Colombo"
    assert list(posts.columns).index("id") < list(posts.columns).index("title")
    assert list(comments["body"]) == ["A comment"]
    assert "config_name: posts" in card
    assert "config_name: comments" in card
    assert "r/AskSriLanka" in card
    assert "r/srilanka" in card
    assert "r/Colombo" in card
    assert "2026-01-01" in card
    assert "AskSriLanka_posts.parquet" in card
    assert 'load_dataset("owner/sri-lanka-reddit", "posts"' in card


def test_stage_dataset_rejects_an_empty_raw_directory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="No non-empty raw parquet"):
        upload_hf.stage_dataset(raw, tmp_path / "stage", {}, "owner/repo")


def test_upload_staged_creates_a_private_dataset_by_default():
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    upload_hf.upload_staged(
        "owner/sri-lanka-reddit",
        "/tmp/stage",
        private=True,
        token="hf_test",
        commit_message="Upload scraped subreddit activity",
        api=FakeApi(),
    )

    assert calls[0][0] == "create_repo"
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["private"] is True
    assert calls[0][1]["exist_ok"] is True
    assert calls[0][1]["token"] == "hf_test"
    assert calls[1][0] == "upload_folder"
    assert calls[1][1]["repo_id"] == "owner/sri-lanka-reddit"
    assert calls[1][1]["folder_path"] == "/tmp/stage"


def test_resolve_token_reads_the_environment(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(upload_hf, "read_colab_secret", lambda name="HF_TOKEN": None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        upload_hf.resolve_token()

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert upload_hf.resolve_token() == "hf_from_env"


def test_resolve_token_reads_the_colab_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(upload_hf, "read_colab_secret", lambda name="HF_TOKEN": "hf_from_colab")

    assert upload_hf.resolve_token() == "hf_from_colab"


def test_colab_secret_requires_notebook_access():
    class Denied:
        def get(self, name):
            raise type("NotebookAccessError", (Exception,), {})("denied")

    with pytest.raises(RuntimeError, match="notebook access"):
        upload_hf.secret_from_userdata(Denied(), "HF_TOKEN")


def test_upload_is_private_unless_public_is_set():
    args = upload_hf.parse_args(["--repo", "owner/sri-lanka-reddit"])
    assert args.public is False
    assert upload_hf.parse_args(["--repo", "owner/sri-lanka-reddit", "--public"]).public is True


def test_size_category_bounds():
    assert upload_hf.size_category(0) == "n<1K"
    assert upload_hf.size_category(1000) == "1K<n<10K"
    assert upload_hf.size_category(10_000_000) == "n>10M"
