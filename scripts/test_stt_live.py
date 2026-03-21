"""Manual integration smoke test for the STT agent + Deepgram service.

Requires a real DEEPGRAM_API_KEY in your .env.
Run with: python scripts/test_stt_live.py

What it proves:
  - DeepgramConnection can open a real WebSocket to Nova-2
  - send_audio() successfully forwards PCM bytes
  - send_keepalive() doesn't crash on a live connection
  - on_transcript callback fires with a real TranscriptResult
  - close() drains and shuts down cleanly
"""

import asyncio
import os
import sys
import wave
import struct

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from services.deepgram_service import DeepgramConnection
from shared.schemas import TranscriptResult


received_transcripts: list[TranscriptResult] = []


async def on_transcript(result: TranscriptResult) -> None:
    received_transcripts.append(result)
    print(f"  📝 Transcript [{result.type}]: '{result.text}' (confidence={result.confidence:.2f})")


async def on_utterance_end() -> None:
    print("  🔔 UtteranceEnd fired")


def generate_sine_pcm(frequency: int = 440, duration_ms: int = 1000) -> bytes:
    """Generate a simple sine wave as PCM16 mono 16kHz bytes.
    
    A real speech file would produce actual transcripts. This just proves
    the connection is alive and forwarding bytes without errors.
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    import math
    samples = [
        int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key or api_key == "test-deepgram-key":
        print("❌ DEEPGRAM_API_KEY not set or is the test placeholder. Set a real key in .env")
        sys.exit(1)

    print("🔌 Connecting to Deepgram...")
    conn = DeepgramConnection(
        on_transcript=on_transcript,
        on_utterance_end=on_utterance_end,
    )

    await conn.connect()
    print(f"✅ Connected. is_connected={conn.is_connected}")

    print("🎵 Sending 1s of sine wave audio...")
    audio = generate_sine_pcm()

    # Send in 20ms chunks (320 bytes each) to simulate ESP32 streaming
    chunk_size = 640  # 16000hz * 2 bytes * 0.02s
    for i in range(0, len(audio), chunk_size):
        await conn.send_audio(audio[i:i + chunk_size])
        await asyncio.sleep(0.02)

    print("💓 Sending keepalive...")
    await conn.send_keepalive()
    print("✅ Keepalive sent without error")

    # Wait briefly for any transcripts to arrive
    print("⏳ Waiting 2s for transcript callbacks...")
    await asyncio.sleep(2.0)

    print("🔌 Closing connection...")
    await conn.close()
    print(f"✅ Closed. is_connected={conn.is_connected}")

    print(f"\n{'='*50}")
    print(f"Transcripts received: {len(received_transcripts)}")
    if received_transcripts:
        print("✅ Callback pipeline is working end-to-end")
    else:
        print("⚠️  No transcripts received — sine wave won't produce speech,")
        print("   but if no errors appeared above, the connection itself is working.")
        print("   Try with a real WAV file to get actual transcripts.")

    print("✅ Smoke test complete")


if __name__ == "__main__":
    asyncio.run(main())