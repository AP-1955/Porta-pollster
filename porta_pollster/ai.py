"""AI analyst: turns a forecast into a plain-English briefing using Claude.

Needs the `anthropic` package and credentials (ANTHROPIC_API_KEY or `ant auth login`).
The model only explains numbers the forecast engine produced; it never invents polls.
"""

from __future__ import annotations

import json

from .forecast import ForecastResult

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a nonpartisan election analyst explaining the output of a statistical forecast.

Rules:
- Use only the numbers in the forecast JSON you are given. Never invent polls, results, or quotes.
- Treat every party and candidate the same way. Do not root for anyone or editorialize.
- Lead with the bottom line: who is favored and how confident the model is, in plain words.
- Explain what is driving the projection: the poll average, the fundamentals (approval, economy),
  pollster house effects, and how voter memory has shrunk the effect of older events.
- Be candid about uncertainty. A 70% favorite loses 3 times in 10. Say so when a race is close.
- Point out weak spots in the data (few polls, one pollster dominating, big undecided share).
- Keep it under 350 words unless the user asks a specific question that needs more."""


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("The AI analyst needs the anthropic package: pip install 'porta-pollster[ai]'") from e
    return anthropic.Anthropic()


def explain(results: list[ForecastResult], question: str | None = None, client=None) -> str:
    """Ask Claude for a briefing on one or more race forecasts."""
    client = client or _client()
    payload = json.dumps([r.to_dict() for r in results], indent=2)
    prompt = f"<forecast>\n{payload}\n</forecast>\n\n"
    prompt += question or "Write a briefing on these races."

    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        # If a request is declined by a safety classifier, retry on a fallback model automatically.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError("The AI analyst declined this request.")
    text = "".join(block.text for block in message.content if block.type == "text").strip()
    if message.stop_reason == "max_tokens":
        text += "\n\n[truncated]"
    return text
