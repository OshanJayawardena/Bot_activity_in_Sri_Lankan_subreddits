import requests

import collect_arctic


class FakeResponse:
    def __init__(self, status, payload, headers=None, text="", url="https://example.test/api"):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.url = url

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return self.responses.pop(0)


def test_next_after_is_the_following_integer_second():
    assert collect_arctic.next_after(1767300949) == "1767300950"
    assert collect_arctic.next_after(1767300949.9) == "1767300950"
    assert "." not in collect_arctic.next_after(20)


def test_normalize_payload_shapes():
    assert collect_arctic.normalize_payload([{"id": "a"}]) == [{"id": "a"}]
    assert collect_arctic.normalize_payload({"data": [{"id": "a"}]}) == [{"id": "a"}]
    assert collect_arctic.normalize_payload({"results": [{"id": "b"}]}) == [{"id": "b"}]
    assert collect_arctic.normalize_payload({"items": [{"id": "c"}]}) == [{"id": "c"}]
    children = [{"kind": "t3", "data": {"id": "d"}}]
    assert collect_arctic.normalize_payload({"data": {"children": children}}) == children
    assert collect_arctic.normalize_payload({"data": {"after": None}}) == []
    assert collect_arctic.normalize_payload({"unexpected": 1}) == []
    assert collect_arctic.normalize_payload(None) == []
    assert collect_arctic.normalize_payload("nope") == []


def test_get_json_returns_body_and_sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(collect_arctic.time, "sleep", lambda seconds: slept.append(seconds))
    session = FakeSession([FakeResponse(200, {"data": [{"id": "1"}]})])

    payload = collect_arctic.get_json(session, "posts/search", {"limit": 1}, sleep_seconds=0.35)

    assert payload == {"data": [{"id": "1"}]}
    assert session.calls[0]["url"] == f"{collect_arctic.BASE}/posts/search"
    assert session.calls[0]["params"] == {"limit": 1}
    assert session.calls[0]["timeout"] == 60
    assert slept == [0.35]


def test_get_json_retries_rate_limit_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(collect_arctic.time, "sleep", lambda seconds: slept.append(seconds))
    session = FakeSession([
        FakeResponse(429, {}, headers={"X-RateLimit-Reset": "2"}),
        FakeResponse(200, [{"id": "ok"}]),
    ])

    payload = collect_arctic.get_json(session, "comments/search", {"limit": 5}, sleep_seconds=0.35)

    assert payload == [{"id": "ok"}]
    assert len(session.calls) == 2
    assert slept == [2.0, 0.35]


def test_get_json_rate_limit_uses_sleep_floor_when_reset_is_smaller(monkeypatch):
    slept = []
    monkeypatch.setattr(collect_arctic.time, "sleep", lambda seconds: slept.append(seconds))
    session = FakeSession([
        FakeResponse(429, {}, headers={"X-RateLimit-Reset": "0.1"}),
        FakeResponse(200, []),
    ])

    collect_arctic.get_json(session, "posts/search", {}, sleep_seconds=0.35)

    assert slept[0] == 0.35


def test_get_json_raises_after_repeated_rate_limits(monkeypatch):
    monkeypatch.setattr(collect_arctic.time, "sleep", lambda seconds: None)
    session = FakeSession([FakeResponse(429, {}) for _ in range(3)])

    try:
        collect_arctic.get_json(session, "posts/search", {}, sleep_seconds=0, retries=3)
    except RuntimeError as exc:
        assert "Repeated rate limiting" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert len(session.calls) == 3


def test_get_json_does_not_retry_http_errors(monkeypatch):
    slept = []
    monkeypatch.setattr(collect_arctic.time, "sleep", lambda seconds: slept.append(seconds))
    session = FakeSession([
        FakeResponse(400, {}, text='{"error":"Invalid date"}', url="https://example.test/posts"),
    ])

    try:
        collect_arctic.get_json(session, "posts/search", {}, sleep_seconds=0.35)
    except requests.HTTPError as exc:
        assert "Invalid date" in str(exc)
        assert "400" in str(exc)
    else:
        raise AssertionError("expected HTTPError")
    assert slept == []
    assert len(session.calls) == 1


def test_collect_paginates_until_short_page(monkeypatch):
    pages = [
        [
            {"id": "1", "created_utc": 10},
            {"id": "2", "created_utc": 20},
        ],
        [{"id": "3", "created_utc": 30}],
    ]
    calls = []

    def fake_get_json(session, endpoint, params, sleep_seconds=0.35, retries=5):
        calls.append((endpoint, dict(params), sleep_seconds))
        return pages[len(calls) - 1]

    monkeypatch.setattr(collect_arctic, "get_json", fake_get_json)

    rows = collect_arctic.collect("posts", "srilanka", "2026-01-01", None, limit=2, max_pages=10, sleep=0.2)

    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert len(calls) == 2
    assert calls[0][0] == "posts/search"
    assert calls[0][1]["subreddit"] == "srilanka"
    assert calls[0][1]["after"] == "2026-01-01"
    assert calls[0][1]["limit"] == 2
    assert calls[0][1]["sort"] == "asc"
    assert calls[0][1]["fields"] == collect_arctic.POST_FIELDS
    assert "before" not in calls[0][1]
    assert calls[0][2] == 0.2
    assert calls[1][1]["after"] == "21"


def test_collect_comments_include_before_and_comment_fields(monkeypatch):
    calls = []

    def fake_get_json(session, endpoint, params, sleep_seconds=0.35, retries=5):
        calls.append((endpoint, dict(params)))
        return []

    monkeypatch.setattr(collect_arctic, "get_json", fake_get_json)

    rows = collect_arctic.collect(
        "comments", "Colombo", "2026-01-01", "2026-02-01", limit=100, max_pages=5, sleep=0.1
    )

    assert rows == []
    assert calls[0][0] == "comments/search"
    assert calls[0][1]["before"] == "2026-02-01"
    assert calls[0][1]["fields"] == collect_arctic.COMMENT_FIELDS


def test_collect_stops_when_cursor_does_not_advance(monkeypatch):
    calls = []

    def fake_get_json(session, endpoint, params, sleep_seconds=0.35, retries=5):
        calls.append(dict(params))
        return [{"id": "1", "created_utc": 50}, {"id": "2", "created_utc": 50}]

    monkeypatch.setattr(collect_arctic, "get_json", fake_get_json)

    rows = collect_arctic.collect("posts", "srilanka", 0, None, limit=2, max_pages=10, sleep=0)

    # The repeated page is deduped, then the unchanged timestamp stops the loop.
    assert [row["id"] for row in rows] == ["1", "2"]
    assert len(calls) == 2
    assert calls[1]["after"] == "51"


def test_collect_stops_when_created_utc_is_missing(monkeypatch):
    calls = []

    def fake_get_json(session, endpoint, params, sleep_seconds=0.35, retries=5):
        calls.append(1)
        return [{"id": "1"}, {"id": "2"}]

    monkeypatch.setattr(collect_arctic, "get_json", fake_get_json)

    rows = collect_arctic.collect("posts", "srilanka", 0, None, limit=2, max_pages=10, sleep=0)

    assert [row["id"] for row in rows] == ["1", "2"]
    assert calls == [1]


def test_collect_stops_at_max_pages(monkeypatch):
    calls = []

    def fake_get_json(session, endpoint, params, sleep_seconds=0.35, retries=5):
        calls.append(params["after"])
        start = 1000 + len(calls) * 10
        return [
            {"id": f"{len(calls)}a", "created_utc": start},
            {"id": f"{len(calls)}b", "created_utc": start + 1},
        ]

    monkeypatch.setattr(collect_arctic, "get_json", fake_get_json)

    rows = collect_arctic.collect("posts", "srilanka", "start", None, limit=2, max_pages=3, sleep=0)

    assert len(rows) == 6
    assert len(calls) == 3
    assert calls[0] == "start"
