"""A local web app for talking to the assistant.

Runs on localhost only. The Anthropic key stays server-side and is never sent to
the browser, which is the whole reason this is a small server rather than a single
HTML file: a page that called the API directly would have to ship the key to the
client, where anyone with devtools can read it.

Agent turns take upwards of a minute, so responses stream over Server-Sent Events
and the UI shows which tools are being called as it goes. A spinner with no detail
for ninety seconds is indistinguishable from a hang.

Conversation state is per-session and in memory: this is a single-user local app,
and a restart losing chat history is the right trade for having no database.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import REPORTS_DIR, load_dotenv
from ..agent.runner import WEB_SEARCH_TOOL, AgentError, stream_agent
from ..agent.tools import AGENT_TOOLS
from ..agent.weekly import SYSTEM_PROMPT, brief_task, gameweek_context

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

#: Appended to the standard system prompt when the assistant is being chatted with
#: rather than asked for the weekly write-up.
CHAT_ADDENDUM = """

## This conversation

You are in a live chat with Jordan. He has just been shown the latest gameweek \
brief, so do not repeat it — answer what he actually asks.

Keep replies short: a few sentences, or a small table when comparing players. Reach \
for a tool whenever a claim needs a number; do not answer from memory about form, \
prices or fixtures, because they change. If a question is not about FPL, just answer \
it briefly and move on.\
"""

#: In-memory conversation store, keyed by session id.
SESSIONS: dict[str, list[dict]] = {}

app = FastAPI(title="FPL Assistant")


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def latest_brief() -> dict | None:
    """The most recent brief on disk, newest gameweek first."""
    if not REPORTS_DIR.exists():
        return None
    briefs = sorted(
        REPORTS_DIR.glob("gw*_brief.md"),
        key=lambda p: int(re.search(r"gw(\d+)", p.name).group(1)),
    )
    if not briefs:
        return None

    path = briefs[-1]
    text = path.read_text(encoding="utf-8")
    gameweek = int(re.search(r"gw(\d+)", path.name).group(1))
    return {
        "gameweek": gameweek,
        "markdown": text,
        "path": str(path),
        "modified": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/context")
def context() -> dict:
    """What the page needs on load: the gameweek, and the latest brief if any."""
    payload: dict = {"manager": "Jordan", "brief": latest_brief()}
    try:
        gameweek, deadline = gameweek_context()
        payload["gameweek"] = gameweek
        payload["deadline"] = deadline
    except (FileNotFoundError, RuntimeError) as error:
        payload["error"] = str(error)
    return payload


def _run(messages: list[dict], system_text: str) -> Iterator[str]:
    """Drive the agent and translate its events into SSE frames."""
    system = [
        {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
    ]
    try:
        for event in stream_agent(
            messages=messages,
            system=system,
            tools=list(AGENT_TOOLS) + [WEB_SEARCH_TOOL],
        ):
            if event["type"] in ("tools", "paused", "text"):
                yield _sse(event)
            elif event["type"] == "done":
                yield _sse({"type": "done", "usage": event["usage"]})
    except AgentError as error:
        yield _sse({"type": "error", "message": str(error)})
    except Exception as error:  # never leave the UI hanging on an unexpected fault
        yield _sse({"type": "error", "message": f"Unexpected failure: {error}"})


@app.post("/api/brief/generate")
def generate_brief() -> StreamingResponse:
    """Produce a fresh brief, streaming progress."""

    def stream() -> Iterator[str]:
        try:
            gameweek, deadline = gameweek_context()
        except (FileNotFoundError, RuntimeError) as error:
            yield _sse({"type": "error", "message": str(error)})
            return

        messages: list[dict] = [
            {"role": "user", "content": brief_task(gameweek, deadline)}
        ]
        collected = ""
        for frame in _run(messages, SYSTEM_PROMPT):
            payload = json.loads(frame.removeprefix("data: ").strip())
            if payload.get("type") == "text":
                collected = payload["text"]
            yield frame

        if collected:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / f"gw{gameweek:02d}_brief.md"
            header = (
                f"# Gameweek {gameweek} brief\n\n"
                f"*Deadline {deadline} · generated "
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}*\n\n"
            )
            path.write_text(header + collected + "\n", encoding="utf-8")
            yield _sse({"type": "saved", "path": str(path), "gameweek": gameweek})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """One conversational turn, with history preserved per session."""
    session_id = request.session_id or uuid.uuid4().hex
    messages = SESSIONS.setdefault(session_id, [])

    # Seed a new conversation with the brief the user is looking at, so follow-up
    # questions like "why him?" have something to refer back to.
    if not messages:
        brief = latest_brief()
        if brief:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "For context, this is the gameweek brief I am currently "
                        f"looking at:\n\n{brief['markdown']}"
                    ),
                }
            )
            messages.append(
                {"role": "assistant", "content": "Understood — ask me anything about it."}
            )

    messages.append({"role": "user", "content": request.message})

    def stream() -> Iterator[str]:
        yield _sse({"type": "session", "session_id": session_id})
        yield from _run(messages, SYSTEM_PROMPT + CHAT_ADDENDUM)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/reset")
def reset(request: ChatRequest) -> dict:
    """Forget a conversation."""
    if request.session_id:
        SESSIONS.pop(request.session_id, None)
    return {"ok": True}


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run(
        "fplr.web.server:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )
