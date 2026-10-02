# AI Calling Agent — INT/26/082

An AI voice agent that answers customer calls, understands requirements in
whatever language the customer speaks (switching mid-call if needed), asks
clarifying questions, confirms details, and produces a structured summary
for a company employee after the call ends.

## Architecture

```
Customer phone call
      |
      v
Twilio (PSTN + Media Streams)  <-- gets you a real/test phone number
      |  (raw audio, websocket)
      v
Speech-to-Text (Deepgram, streaming, auto language detect)
      |  (text)
      v
Conversation Agent (LLM, agent/conversation.py)
      |  (text reply, in the customer's language)
      v
Text-to-Speech (ElevenLabs)
      |  (audio, websocket)
      v
Back to customer over the call
      |
   [call ends]
      v
Summary Agent (agent/summary.py) -> structured JSON summary -> sent to
company employee (email / Slack / dashboard - you wire this last)
```

## Two ways to run this

### 1. Text demo (do this first — no telephony/audio setup needed)

This proves the *conversation understanding + summary* half of the
pipeline works, using your terminal instead of a phone call. All you need
is one LLM API key.

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY
python demo/text_demo.py
```

Talk to it like a customer ("hi, i need... umm something like a website
for my shop, not sure how big"), try switching languages mid-conversation,
and end with something like "yes that's right, go ahead." It will print a
structured summary at the end.

### 2. Real phone call demo (do this second, once step 1 works)

This wires the same conversation agent up to an actual test phone number
using Twilio Media Streams, Deepgram for speech-to-text, and ElevenLabs
for speech-to-speech.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ALL keys: Anthropic, Deepgram, ElevenLabs, Twilio
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then:
1. Expose your local server publicly (e.g. `ngrok http 8000`) since Twilio
   needs a public URL to reach you.
2. In the Twilio console, buy/use a trial number and set its "A call comes
   in" webhook to `https://<your-ngrok-url>/incoming-call`.
3. Call the Twilio number from your phone. You should hear the agent
   answer and be able to talk to it.
4. When the call ends, check `summaries/` for the generated JSON summary.

## Project layout

```
ai-calling-agent/
├── app.py                  # FastAPI server: Twilio webhook + media stream
├── agent/
│   ├── conversation.py     # The LLM conversation agent + system prompt
│   ├── summary.py          # Post-call summary generation
│   ├── stt.py              # Deepgram streaming speech-to-text wrapper
│   └── tts.py              # ElevenLabs text-to-speech wrapper
├── demo/
│   └── text_demo.py        # Terminal-based demo, no telephony needed
├── requirements.txt
└── .env.example
```

## What's real vs. what you still need to fill in

- The conversation agent, its system prompt, and the summary generator
  (`agent/conversation.py`, `agent/summary.py`) are complete and work as
  soon as you add an LLM API key — try them with `demo/text_demo.py`.
- The Twilio/Deepgram/ElevenLabs wiring in `app.py`, `agent/stt.py`, and
  `agent/tts.py` follows each provider's documented streaming pattern, but
  you'll need to:
  - Create accounts and get API keys for Twilio, Deepgram, and ElevenLabs.
  - Double-check the exact audio encoding your Deepgram/ElevenLabs plan
    expects (this code assumes 8kHz mu-law, which is what Twilio sends).
  - Add real delivery for the summary (currently it just saves to a local
    `summaries/` folder as JSON) — e.g. send it to Slack, email, or a
    dashboard your company team actually looks at.

## Suggested next steps (in order)

1. Run the text demo. Try several messy, informal, multilingual test
   conversations. Read the generated summaries — tune the system prompts
   in `agent/conversation.py` and `agent/summary.py` until they consistently
   capture the right details.
2. Sign up for Twilio (free trial gives you a test number), Deepgram, and
   ElevenLabs. Add the keys to `.env`.
3. Run `app.py` behind `ngrok` and make a real test call to yourself.
   Iterate on latency and interruption handling — this is the hardest part
   of a voice agent to get feeling natural.
4. Record 2–3 test calls (ideally one that switches languages mid-call,
   one with vague/informal requirements) and keep the transcript + summary
   pairs — these make the best demo evidence for your internship report.
5. Add real summary delivery (Slack webhook or email) instead of writing
   to a local file.
6. Only after all of the above works reliably: talk to your company about
   pointing a real business number at this system.
