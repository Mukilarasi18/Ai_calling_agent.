"""
FastAPI server that answers real phone calls via Twilio and runs the full
pipeline: Twilio audio -> Deepgram STT -> ConversationAgent (LLM) ->
ElevenLabs TTS -> back to Twilio audio. When the call ends, it generates
and saves a summary for the company employee.

Run with:  uvicorn app:app --host 0.0.0.0 --port 8000
Then point a Twilio number's voice webhook at:
  https://<your-public-url>/incoming-call
(use ngrok or similar for local testing — see README.md)
"""

import os
import json
import base64
import asyncio

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse

from agent.conversation import ConversationAgent, ConversationState
from agent.stt import DeepgramStreamingSTT
from agent.tts import ElevenLabsTTS
from agent.summary import generate_summary, save_summary, format_for_employee

app = FastAPI()

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio hits this webhook when a call comes in. We respond with TwiML
    that opens a bidirectional Media Stream to our websocket endpoint,
    which is where the actual conversation happens.
    """
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}/media-stream" />
    </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio opens this websocket for the duration of the call and streams
    raw audio frames to us, and expects audio frames streamed back.
    """
    await websocket.accept()

    conversation_agent = ConversationAgent()
    state = ConversationState()
    tts = ElevenLabsTTS()
    stream_sid: str | None = None

    async def handle_transcript(text: str, language: str):
        """Called by Deepgram STT each time the customer finishes an utterance."""
        result = conversation_agent.respond(state, text)
        reply_text = result["reply"]

        audio_bytes = await tts.synthesize(reply_text)
        if stream_sid:
            message = ElevenLabsTTS.to_twilio_media_message(stream_sid, audio_bytes)
            await websocket.send_text(json.dumps(message))

        if result.get("call_complete"):
            await asyncio.sleep(2)  # let the final audio finish playing
            await websocket.close()

    stt = DeepgramStreamingSTT(on_transcript=handle_transcript)
    await stt.connect()

    try:
        while True:
            raw_message = await websocket.receive_text()
            data = json.loads(raw_message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                # Speak the opening line immediately so the caller isn't
                # met with silence.
                opening = conversation_agent.opening_line()
                state.add("agent", opening)
                audio_bytes = await tts.synthesize(opening)
                await websocket.send_text(
                    json.dumps(ElevenLabsTTS.to_twilio_media_message(stream_sid, audio_bytes))
                )

            elif event == "media":
                mulaw_bytes = base64.b64decode(data["media"]["payload"])
                await stt.send_audio_chunk(mulaw_bytes)

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        await stt.close()
        await _finalize_call(state)


async def _finalize_call(state: ConversationState):
    """Runs once the call ends: generate and save the summary."""
    if not state.turns:
        return  # call ended with no conversation at all

    transcript = state.as_transcript()
    summary = generate_summary(transcript)
    path = save_summary(summary, transcript)

    print("\n" + "=" * 60)
    print("CALL ENDED — SUMMARY FOR COMPANY EMPLOYEE")
    print("=" * 60)
    print(format_for_employee(summary))
    print(f"\nSaved to: {path}")
    print("=" * 60 + "\n")

    # TODO: replace this print-and-save with real delivery, e.g.:
    #   - send `format_for_employee(summary)` to a Slack webhook
    #   - email it to os.environ["SUMMARY_NOTIFY_EMAIL"]
    #   - POST `summary` to your internal dashboard/API


@app.get("/health")
async def health():
    return {"status": "ok"}
