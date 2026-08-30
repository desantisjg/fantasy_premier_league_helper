"""Local UI server contracts.

Offline only — nothing here calls the Anthropic API. The streaming endpoints are
covered for their failure path (no credentials) rather than a live run, since a
real agent turn costs money and is exercised by hand.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fplr.web import server


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'data-theme="dark"' in body, "the UI must default to dark mode"
    assert "Hi Jordan" in body, "the page must greet by name on load"


def test_context_reports_the_gameweek_and_latest_brief(client):
    payload = client.get("/api/context").json()
    assert payload["manager"] == "Jordan"
    if "error" in payload:
        pytest.skip(f"no snapshot available: {payload['error']}")
    assert isinstance(payload["gameweek"], int)
    assert payload["deadline"]


def test_latest_brief_picks_the_highest_gameweek(tmp_path, monkeypatch):
    """Briefs must sort numerically, or gw10 would rank below gw9."""
    monkeypatch.setattr(server, "REPORTS_DIR", tmp_path)
    for gameweek in (3, 9, 10):
        (tmp_path / f"gw{gameweek:02d}_brief.md").write_text(f"# Gameweek {gameweek}\n")

    brief = server.latest_brief()
    assert brief["gameweek"] == 10, "sorted as text, gw09 would beat gw10"


def test_latest_brief_is_none_when_there_are_no_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPORTS_DIR", tmp_path)
    assert server.latest_brief() is None


def test_reset_clears_only_the_named_session(client):
    server.SESSIONS["keep"] = [{"role": "user", "content": "a"}]
    server.SESSIONS["drop"] = [{"role": "user", "content": "b"}]

    client.post("/api/reset", json={"session_id": "drop", "message": ""})
    assert "drop" not in server.SESSIONS
    assert "keep" in server.SESSIONS
    server.SESSIONS.pop("keep", None)


def test_chat_streams_a_clean_error_without_credentials(client, monkeypatch):
    """A missing key must surface as a readable message, not a stack trace."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with client.stream(
        "POST", "/api/chat", json={"session_id": "test-noauth", "message": "hi"}
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    server.SESSIONS.pop("test-noauth", None)
    errors = [e for e in events if e["type"] == "error"]
    assert errors, f"expected an error event, got {[e['type'] for e in events]}"
    assert "credentials" in errors[0]["message"].lower()
