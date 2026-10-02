"""
Streaming speech-to-text via Deepgram, wired for Twilio Media Streams
audio (8kHz mu-law, base64-encoded chunks arriving over a websocket).

Deepgram's streaming API supports automatic language detection, which
handles the "understand multiple languages / handle language switches"
requirement on the input side (the LLM handles it on the output side by
replying in whatever language was detected).

Docs: https://developers.deepgram.com/docs/live-streaming-audio
"""

import os
import json
import asyncio
from typing import Callable, Awaitable

import websockets

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=mulaw&sample_rate=8000&channels=1"
    "&punctuate=true&interim_results=false"
    "&detect_language=true&smart_format=true"
)


class DeepgramStreamingSTT:
    """
    One instance per active call. Feed it raw audio chunks as they arrive
    from Twilio; it calls `on_transcript` with each finalized utterance.
    """

    def __init__(self, on_transcript: Callable[[str, str], Awaitable[None]], api_key: str | None = None):
        """
        on_transcript: async callback(transcript_text, detected_language)
        called once per finalized utterance (i.e. once the customer pauses).
        """
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self.on_transcript = on_transcript
        self._ws = None
        self._recv_task = None

    async def connect(self):
        self._ws = await websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {self.api_key}"},
        )
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def send_audio_chunk(self, mulaw_bytes: bytes):
        if self._ws is not None:
            await self._ws.send(mulaw_bytes)

    async def _receive_loop(self):
        async for message in self._ws:
            data = json.loads(message)
            try:
                alt = data["channel"]["alternatives"][0]
                transcript = alt.get("transcript", "").strip()
                # Deepgram returns a detected language code when
                # detect_language=true; default to "en" if absent.
                language = data.get("channel", {}).get("detected_language", "en")
                is_final = data.get("is_final", False)
            except (KeyError, IndexError):
                continue

            if transcript and is_final:
                await self.on_transcript(transcript, language)

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws is not None:
            await self._ws.close()
