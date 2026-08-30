"""The shared agent loop.

Both the weekly brief and the chat UI drive the same conversation loop, so it lives
here once. Duplicating it would mean duplicating the `pause_turn` handling, which is
the subtlest part: web search is a server-side tool, and a long search turn can stop
with `stop_reason: "pause_turn"`. The SDK's tool runner only continues after a
*client* tool returns a result, so a paused turn silently ends the loop and hands
back a truncated answer with no error at all. The remedy is to mirror the
conversation and restart the runner, which is what `stream_agent` does.

Events are yielded as they happen rather than returned at the end, because a full
run takes upwards of a minute and a UI showing nothing for that long looks broken.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic

MODEL = "claude-opus-5"
MAX_TOKENS = 16_000
MAX_PAUSE_RESTARTS = 5

#: Bounded so a single run cannot spend unboundedly on search.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 8,
}


class AgentError(RuntimeError):
    """A run that failed for a reason the user can act on."""


def describe_api_error(error: Exception) -> str:
    """Turn an SDK exception into something actionable.

    Handled most-specific first: the difference between "no credit", "bad key" and
    "rate limited" is the whole point, and collapsing them would send someone to
    check the wrong thing.
    """
    if isinstance(error, anthropic.AuthenticationError):
        return (
            "Anthropic rejected the credentials. Check ANTHROPIC_API_KEY in .env, "
            "or run `ant auth login`."
        )
    if isinstance(error, anthropic.PermissionDeniedError):
        return "This key is not permitted to use the Messages API or this model."
    if isinstance(error, anthropic.RateLimitError):
        return "Rate limited by Anthropic. Wait for the retry-after window and retry."
    if isinstance(error, anthropic.BadRequestError):
        message = str(getattr(error, "message", "") or error)
        if "credit balance" in message.lower():
            return (
                "The Anthropic account has no credits, so the request was refused "
                "before any work was done — nothing was charged. Add credits under "
                "Plans & Billing at console.anthropic.com. The key itself "
                "authenticated correctly."
            )
        return f"Anthropic rejected the request: {message}"
    if isinstance(error, anthropic.APIConnectionError):
        return f"Could not reach the Anthropic API: {error}"
    if isinstance(error, anthropic.APIStatusError):
        return f"Anthropic returned {error.status_code}: {error}"
    return str(error)


def build_client(env: dict[str, str] | None = None) -> anthropic.Anthropic:
    """An SDK client, with a clear error if no credentials are configured."""
    import os

    source = env if env is not None else os.environ
    if not (source.get("ANTHROPIC_API_KEY") or source.get("ANTHROPIC_AUTH_TOKEN")):
        raise AgentError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env, or run "
            "`ant auth login` to store a profile the SDK picks up automatically."
        )
    return anthropic.Anthropic()


def stream_agent(
    *,
    messages: list[dict],
    system: list[dict] | str,
    tools: list[Any],
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> Iterator[dict]:
    """Run the agent, yielding progress events and finally the answer.

    Yields dicts of:
      {"type": "tools",  "names": [...]}   a turn that called tools
      {"type": "paused", "restart": n}     a server-tool turn resumed
      {"type": "text",   "text": "..."}    the final answer
      {"type": "done",   "usage": {...}, "messages": [...]}

    `messages` is mutated in place so the caller ends up holding the full
    conversation, which is what makes multi-turn chat work.
    """
    client = client or build_client()
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    final = None
    restarts = 0

    while True:
        try:
            runner = client.beta.messages.tool_runner(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            )

            last = None
            for message in runner:
                last = message
                for key in usage:
                    usage[key] += getattr(message.usage, key, 0) or 0

                # Server-side tools arrive as `server_tool_use`, not `tool_use`;
                # matching only the latter hides web search entirely.
                called = [
                    block.name
                    for block in message.content
                    if block.type in ("tool_use", "server_tool_use")
                ]
                if called:
                    yield {"type": "tools", "names": called}

                # Mirror the history: the runner keeps its own copy and does not
                # expose it, so a restart after a pause needs our own record.
                messages.append({"role": "assistant", "content": message.content})
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    messages.append(tool_response)
        except anthropic.APIError as error:
            raise AgentError(describe_api_error(error)) from error

        final = last
        if final is None or final.stop_reason != "pause_turn":
            break

        restarts += 1
        if restarts > MAX_PAUSE_RESTARTS:
            raise AgentError(
                f"The turn was still paused after {MAX_PAUSE_RESTARTS} restarts. "
                "This usually means web search is looping; retry or lower max_uses."
            )
        yield {"type": "paused", "restart": restarts}

    text = "\n".join(
        block.text for block in (final.content if final else []) if block.type == "text"
    ).strip()
    yield {"type": "text", "text": text}
    yield {"type": "done", "usage": usage, "messages": messages}
