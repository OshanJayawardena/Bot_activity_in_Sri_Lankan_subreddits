\
import argparse
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml
from tqdm import tqdm

BASE = "https://arctic-shift.photon-reddit.com/api"

POST_FIELDS = ",".join([
    "id","author","author_fullname","created_utc","retrieved_on","subreddit",
    "subreddit_id","score","title","selftext","url","link_flair_text",
    "num_comments","crosspost_parent"
])

COMMENT_FIELDS = ",".join([
    "id","author","author_fullname","created_utc","retrieved_on","subreddit",
    "subreddit_id","score","body","link_id","parent_id","distinguished"
])

def _is_slowdown(response):
    """Arctic Shift returns 422 when a query times out and asks the client to pause."""
    if response.status_code != 422:
        return False
    body = (getattr(response, "text", "") or "").lower()
    return "timeout" in body or "slow down" in body


def _is_retryable(response):
    if response.status_code == 429 or _is_slowdown(response):
        return True
    return response.status_code in (500, 502, 503, 504)


def _retry_wait(response, attempt, sleep_seconds):
    if response is not None and response.status_code == 429:
        try:
            wait = float(response.headers.get("X-RateLimit-Reset", "10"))
        except (TypeError, ValueError):
            wait = 10.0
        return max(wait, sleep_seconds)
    # 2s, 4s, 8s, ... capped, so a "slow down" response actually backs off.
    return min(60.0, max(sleep_seconds, 2.0) * (2 ** attempt))


def get_json(session, endpoint, params, sleep_seconds=0.35, retries=8):
    url = f"{BASE}/{endpoint}"
    last_status = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            last_status = "network"
            if attempt == retries - 1:
                raise
            time.sleep(_retry_wait(None, attempt, sleep_seconds))
            continue
        if _is_retryable(r):
            last_status = r.status_code
            time.sleep(_retry_wait(r, attempt, sleep_seconds))
            continue
        if r.status_code >= 400:
            body = (getattr(r, "text", "") or "").strip()
            message = f"{r.status_code} Error for url: {r.url}"
            if body:
                message = f"{message}: {body[:500]}"
            raise requests.HTTPError(message, response=r)
        time.sleep(sleep_seconds)
        return r.json()
    raise RuntimeError(f"Repeated rate limiting or timeouts ({last_status}) from {url}")


def next_after(created_utc):
    """Integer epoch second just after ``created_utc``.

    Arctic Shift accepts epoch seconds and ISO-8601 dates. A fractional epoch
    such as ``1767300949.001`` is rejected with HTTP 400. Reddit timestamps
    are whole seconds, so the next second keeps every later post.
    """
    return str(int(float(created_utc)) + 1)


def normalize_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # Some APIs return {data: {children: [...]}}
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("children"), list):
            return data["children"]
    return []

def collect(kind, subreddit, after, before, limit, max_pages, sleep):
    endpoint = "posts/search" if kind == "posts" else "comments/search"
    fields = POST_FIELDS if kind == "posts" else COMMENT_FIELDS
    rows = []
    session = requests.Session()
    params = {
        "subreddit": subreddit,
        "after": after,
        "limit": limit,
        "sort": "asc",
        "fields": fields,
        "format": "json",
    }
    if before:
        params["before"] = before

    seen_ids = set()
    last_created = None
    for _ in tqdm(range(max_pages), desc=f"{kind} r/{subreddit}"):
        payload = get_json(session, endpoint, params, sleep)
        batch = normalize_payload(payload)
        if not batch:
            break
        for row in batch:
            row_id = row.get("id")
            if row_id is not None and row_id in seen_ids:
                continue
            if row_id is not None:
                seen_ids.add(row_id)
            rows.append(row)

        created = [x.get("created_utc") for x in batch if x.get("created_utc") is not None]
        current_max = max(created) if created else None
        if current_max is None or current_max == last_created:
            break
        last_created = current_max
        params["after"] = next_after(current_max)

        if len(batch) < limit:
            break

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    after = cfg["collection"]["after"]
    before = cfg["collection"].get("before")
    limit = cfg["collection"]["limit_per_request"]
    sleep = cfg["collection"]["sleep_seconds"]
    pages = cfg["collection"]["max_pages_per_subreddit"]

    for sr in cfg["subreddits"]:
        for kind in ("posts", "comments"):
            rows = collect(kind, sr, after, before, limit, pages, sleep)
            df = pd.DataFrame(rows)
            path = out / f"{sr}_{kind}.parquet"
            df.to_parquet(path, index=False)
            print(f"{kind:9s} r/{sr}: {len(df):,} rows -> {path}")

if __name__ == "__main__":
    main()
