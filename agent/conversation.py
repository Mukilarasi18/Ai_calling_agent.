"""
The conversation agent: the "brain" of the call.

Responsibilities (per the internship spec):
  - Talk naturally, understand the customer's language and requirements
  - Ask clarifying questions when something is missing or unclear
  - Understand informal / vague phrasing ("something like what you did
    before", "not too expensive", "soonish")
  - Reply in whatever language the customer is currently using, and
    switch languages mid-conversation if the customer does
  - Remember the whole conversation so far
  - Signal when the call is ready to end (customer has confirmed details)

This module is transport-agnostic: it doesn't know or care whether the
text came from a phone call (via STT) or a terminal (the text demo). That
keeps it easy to test without any telephony setup.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are a warm, efficient phone agent for a company \
that takes on custom projects for customers (e.g. web/app development, \
design, or similar services — infer the domain from context if the \
customer doesn't say).

Your job on this call:
1. Greet the customer naturally and ask how you can help.
2. Understand what they need, even when they describe it informally, \
vaguely, or incompletely (e.g. "something like a website for my shop", \
"not sure exactly", "similar to what you did for my friend").
3. Ask short, specific clarifying questions one at a time to fill gaps in:
   - What exactly they need (scope/features)
   - Any deadline or timeframe
   - Any budget, if they mention one
   - Any must-have requirements or constraints
   Do not interrogate them with a long list at once — ask naturally, the \
way a helpful human would, and only ask about what's actually missing.
4. Reply in the SAME language the customer is currently using. If they \
switch languages mid-conversation, switch with them immediately and \
naturally, without commenting on the switch.
5. Before ending the call, summarize back what you understood in one or \
two sentences and explicitly ask the customer to confirm it's correct.
6. Only end the call after the customer confirms. Once they confirm, \
thank them and let them know the team will follow up, then end the call.

If the customer asks whether you are a real person,an AI, or a bot, \
say honestly and briefly that you're an AI assistant helping with calls,\
headers in your replies; speak in plain natural sentences.

Keep every reply short (1-3 sentences) — this is a spoken phone \
conversation, not a chat message. Never use bullet points, markdown, or \
headers in your replies; speak in plain natural sentences.

You must also silently track whether the call is ready to end. After \
your reply, decide: has the customer explicitly confirmed the summary \
you gave them? Output your response as a single JSON object with keys:
  "reply": the natural-language reply to speak to the customer
  "call_complete": true only if the customer just confirmed the final \
summary and the call should end now, otherwise false
  "detected_language": the language the customer is currently speaking, \
as a plain English name (e.g. "Spanish", "Tamil", "English")

Output ONLY the JSON object, nothing else — no markdown fences, no \
preamble.
"""


@dataclass
class Turn:
    role: str  # "customer" or "agent"
    text: str


@dataclass
class ConversationState:
    """Everything the agent needs to remember about one call."""
    turns: list[Turn] = field(default_factory=list)
    call_complete: bool = False
    last_detected_language: str = "English"

    def add(self, role: str, text: str) -> None:
        self.turns.append(Turn(role=role, text=text))

    def as_transcript(self) -> str:
        lines = []
        for t in self.turns:
            speaker = "Customer" if t.role == "customer" else "Agent"
            lines.append(f"{speaker}: {t.text}")
        return "\n".join(lines)


class ConversationAgent:
    """
    Wraps the LLM call so both the text demo and the real phone pipeline
    can share identical conversation logic.
    """

    def __init__(self, api_key: Optional[str] = None):
        genai.configure(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
    transport="rest",
    )
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

    def opening_line(self) -> str:
        """A fixed, fast opening greeting so the call doesn't start with LLM
        latency — the LLM takes over from the customer's first reply."""
        return "Hi, thanks for calling! How can I help you today?"

    def respond(self, state: ConversationState, customer_message: str) -> dict:
        """
        Feed the customer's latest message plus full history to the LLM
        and get back the agent's reply, whether the call should end, and
        the detected language.

        Returns a dict: {"reply": str, "call_complete": bool,
                          "detected_language": str}
        """
        state.add("customer", customer_message)

        prompt = self._build_prompt(state)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=400,
            ),
        )

        raw = response.text.strip()
        parsed = self._parse_json_response(raw)

        state.add("agent", parsed["reply"])
        state.call_complete = parsed.get("call_complete", False)
        state.last_detected_language = parsed.get(
            "detected_language", state.last_detected_language
        )
        return parsed

    def _build_prompt(self, state: ConversationState) -> str:
        transcript = state.as_transcript()
        return (
            "Here is the conversation so far (most recent message last). "
            "Respond as the agent to the customer's latest message, "
            "following your instructions exactly.\n\n"
            f"{transcript}"
        )

    def _parse_json_response(self, raw: str) -> dict:
        # Defensive parsing: strip accidental markdown fences if the model
        # adds them despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fall back gracefully rather than crashing the call.
            return {
                "reply": raw,
                "call_complete": False,
                "detected_language": "English",
            }