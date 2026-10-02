"""
Generates the structured post-call summary for the company employee.

Takes the full transcript (in whatever language(s) it happened in) and
produces a single structured report in English, regardless of the call's
language, so any employee can read it.
"""

import os
import json
import re
from datetime import datetime, timezone

import google.generativeai as genai

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

SUMMARY_SYSTEM_PROMPT = """You turn a customer call transcript into a \
short, clear internal summary for a company employee who did not hear \
the call. The transcript may be in any language or mix of languages —
always write the summary itself in English.

Extract:
- customer_need: one or two sentences on what the customer actually wants
- key_requirements: a list of specific requirements, constraints, or \
must-haves the customer mentioned (empty list if none were given)
- deadline: the deadline or timeframe the customer mentioned, or null \
if none was given
- budget: the budget the customer mentioned, or null if none was given
- ambiguities: a list of things that are still unclear or need follow-up \
(empty list if the requirements were fully clear)
- next_action: one clear, concrete sentence on what the company employee \
should do next
- customer_confirmed: true if the transcript shows the customer explicitly \
confirming the summarized details before the call ended, false otherwise
- languages_used: list of the language(s) the customer spoke during the call

Output ONLY a single JSON object with exactly these keys, nothing else — \
no markdown fences, no preamble, no commentary.
"""


def generate_summary(transcript: str, api_key: str | None = None) -> dict:
    genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SUMMARY_SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

    response = model.generate_content(
        f"Transcript:\n\n{transcript}\n\nProduce the JSON summary now."
    )

    raw = response.text.strip()
    cleaned = raw

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {
            "customer_need": raw,
            "key_requirements": [],
            "deadline": None,
            "budget": None,
            "ambiguities": ["Summary could not be parsed automatically — review transcript manually."],
            "next_action": "Review the raw transcript manually.",
            "customer_confirmed": False,
            "languages_used": [],
        }

    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return data


def save_summary(summary: dict, transcript: str, out_dir: str = "summaries") -> str:
    """Saves the summary + transcript to a local JSON file and returns the path.

    This is a stand-in for real delivery (Slack, email, a dashboard). Swap
    this function's body out once you wire up real delivery — everything
    else in the pipeline calls this one function, so that's the only place
    you need to change.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/call_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {"summary": summary, "transcript": transcript}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filename


def format_for_employee(summary: dict) -> str:
    """Human-readable version of the summary, e.g. for a Slack message or email body."""
    lines = [
        "New customer call summary",
        "--------------------------",
        f"Customer need: {summary.get('customer_need', 'N/A')}",
    ]

    reqs = summary.get("key_requirements") or []
    if reqs:
        lines.append("Key requirements:")
        lines.extend(f"  - {r}" for r in reqs)

    lines.append(f"Deadline: {summary.get('deadline') or 'Not specified'}")
    lines.append(f"Budget: {summary.get('budget') or 'Not specified'}")

    ambiguities = summary.get("ambiguities") or []
    if ambiguities:
        lines.append("Needs follow-up on:")
        lines.extend(f"  - {a}" for a in ambiguities)

    lines.append(f"Next action: {summary.get('next_action', 'N/A')}")
    lines.append(f"Customer confirmed details: {'Yes' if summary.get('customer_confirmed') else 'No'}")
    lines.append(f"Language(s) used: {', '.join(summary.get('languages_used') or []) or 'Unknown'}")

    return "\n".join(lines)
