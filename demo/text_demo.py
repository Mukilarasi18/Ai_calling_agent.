"""
Text-based demo of the full workflow, minus the phone call itself.

Run this first, before touching Twilio/Deepgram/ElevenLabs. It proves out
the two hardest parts of the assignment — natural multilingual
conversation with requirement understanding, and post-call summarization —
using your keyboard instead of a phone line.

Usage:
    python demo/text_demo.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agent.conversation import ConversationAgent, ConversationState
from agent.summary import generate_summary, save_summary, format_for_employee


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY in your .env file first (see .env.example).")
        return

    agent = ConversationAgent()
    state = ConversationState()

    print("=" * 60)
    print("AI Calling Agent — text demo")
    print("Type as the customer. Type 'hangup' to end without confirming.")
    print("Try switching languages mid-conversation, and being vague on")
    print("purpose — that's exactly what this is meant to handle.")
    print("=" * 60)

    opening = agent.opening_line()
    print(f"\nAgent: {opening}")
    state.add("agent", opening)

    while True:
        try:
            customer_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not customer_message:
            continue
        if customer_message.lower() == "hangup":
            print("\n[Call ended by customer without confirmation]")
            break

        result = agent.respond(state, customer_message)
        print(f"\nAgent: {result['reply']}")

        if result.get("call_complete"):
            print("\n[Call complete — customer confirmed details]")
            break

    print("\nGenerating summary for the company employee...\n")
    transcript = state.as_transcript()
    summary = generate_summary(transcript)
    path = save_summary(summary, transcript)

    print("=" * 60)
    print(format_for_employee(summary))
    print("=" * 60)
    print(f"\nFull summary + transcript saved to: {path}")


if __name__ == "__main__":
    main()
