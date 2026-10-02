"""
Text-to-speech via ElevenLabs, producing audio in the mu-law 8kHz format
Twilio Media Streams expects, so it can be piped straight back to the
caller.

Docs: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
"""

import os
import base64
import httpx

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class ElevenLabsTTS:
    def __init__(self, api_key: str | None = None, voice_id: str | None = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID")

    async def synthesize(self, text: str) -> bytes:
        """
        Returns raw mu-law 8kHz audio bytes for the given text, ready to
        be base64-encoded and sent back over the Twilio media stream.

        ElevenLabs' multilingual model (eleven_multilingual_v2 or newer)
        auto-detects the language of the input text, so no separate
        language parameter is needed here — just make sure the LLM's
        reply text is in the right language before calling this.
        """
        url = ELEVENLABS_API_URL.format(voice_id=self.voice_id)
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "output_format": "ulaw_8000",  # matches Twilio's expected format
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    @staticmethod
    def to_twilio_media_message(stream_sid: str, audio_bytes: bytes) -> dict:
        """Wraps raw audio into the JSON message shape Twilio's Media
        Streams websocket expects for outbound audio."""
        return {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(audio_bytes).decode("utf-8")},
        }
