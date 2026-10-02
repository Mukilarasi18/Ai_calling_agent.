"""
Local microphone demo of the FULL voice pipeline — real speech in, real
speech out — using your laptop's mic and speakers instead of a phone call.
"""

import os
import io
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS

from agent.conversation import ConversationAgent, ConversationState
from agent.summary import generate_summary, save_summary, format_for_employee

SAMPLE_RATE = 16000
DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?encoding=linear16&sample_rate=16000&channels=1"
    "&punctuate=true&smart_format=true&detect_language=true"
)
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def record_until_enter() -> bytes:
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback
    )
    with stream:
        input()

    if not frames:
        return b""

    audio = np.concatenate(frames, axis=0)
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    response = httpx.post(DEEPGRAM_URL, headers=headers, content=wav_bytes, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""



def speak(text: str):
    """Converts text to speech via Google's free gTTS and plays it."""
    tts = gTTS(text=text, lang="en")
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)

    audio_data, sample_rate = sf.read(buffer)
    sd.play(audio_data, sample_rate)
    sd.wait()


def main():
    missing = [
        k for k in ("GEMINI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
        if not os.environ.get(k)
    ]
    if missing:
        print(f"Missing from .env: {', '.join(missing)}. Fill these in first.")
        return

    agent = ConversationAgent()
    state = ConversationState()

    print("=" * 60)
    print("AI Calling Agent — live microphone demo")
    print("=" * 60)

    opening = agent.opening_line()
    print(f"\nAgent: {opening}")
    state.add("agent", opening)
    speak(opening)

    while True:
        input("\nPress Enter, then speak. Press Enter again when done...")
        print("Recording... press Enter to stop.")
        wav_bytes = record_until_enter()

        if not wav_bytes:
            print("No audio captured, try again.")
            continue

        print("Transcribing...")
        transcript = transcribe(wav_bytes)
        if not transcript:
            print("Couldn't understand that, please try again.")
            continue
        print(f"You said: {transcript}")

        result = agent.respond(state, transcript)
        print(f"Agent: {result['reply']}")
        speak(result["reply"])

        if result.get("call_complete"):
            print("\n[Call complete — customer confirmed details]")
            break

    print("\nGenerating summary for the company employee...\n")
    full_transcript = state.as_transcript()
    summary = generate_summary(full_transcript)
    path = save_summary(summary, full_transcript)

    print("=" * 60)
    print(format_for_employee(summary))
    print("=" * 60)
    print(f"\nFull summary + transcript saved to: {path}")


if __name__ == "__main__":
    main()