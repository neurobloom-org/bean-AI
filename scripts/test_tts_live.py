"""Live integration test for TTS agent — requires real ElevenLabs API key.

Usage:
    python scripts/test_tts_live.py

Needs a .env file with:
    ELEVENLABS_API_KEY=your_key_here
    ELEVENLABS_VOICE_ID=your_voice_id_here
    ELEVENLABS_MODEL_ID=eleven_turbo_v2
"""

import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env before importing anything from the project
from dotenv import load_dotenv
load_dotenv()

# Minimal env check before going further
required = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL_ID"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}")
    sys.exit(1)

from agents.tts.agent import synthesize_speech, stream_tts_chunks


async def test_full_audio() -> None:
    print("\n── Test 1: Full audio (synthesize_speech) ──────────────────")
    result = await synthesize_speech("Hello, I am BEAN. How are you feeling today?")

    if result["status"] == "error":
        print(f"❌ Error: {result['error']}")
        return

    print(f"✅ Status      : {result['status']}")
    print(f"   Voice ID    : {result['voice_id']}")
    print(f"   Text length : {result['text_length']} chars")
    print(f"   Audio bytes : {result['audio_length_bytes']} bytes")
    print(f"   Cache hit   : {result['cache_hit']}")
    print(f"   Turn ID     : {result['turn_id']}")

    # Save to file so you can actually listen to it
    audio_bytes = base64.b64decode(result["audio_b64"])
    output_path = "scripts/test_tts_output_full.mp3"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"   Saved to    : {output_path}  ← open this and listen")


async def test_streaming() -> None:
    print("\n── Test 2: Streaming chunks (stream_tts_chunks) ────────────")
    chunk_count = 0
    total_bytes = 0

    try:
        async for chunk in stream_tts_chunks("Streaming test. One two three."):
            chunk_count += 1
            total_bytes += len(chunk.audio_chunk)
            print(
                f"   Chunk {chunk_count:02d}: {len(chunk.audio_chunk):5d} bytes"
                f"  is_final={chunk.is_final}"
            )
    except Exception as exc:
        print(f"❌ Stream failed: {exc}")
        return

    print(f"✅ Total chunks : {chunk_count}")
    print(f"   Total bytes  : {total_bytes}")


# correct — check the returned dict instead
async def test_empty_text() -> None:
    result = await synthesize_speech("   ")
    if result["status"] == "error" and "empty" in result["error"].lower():
        print(f"✅ Empty text handled cleanly: {result['error']}")
    else:
        print(f"❌ Unexpected result: {result}")


async def test_error_handling() -> None:
    print("\n── Test 4: Error handling (bad voice_id) ───────────────────")
    result = await synthesize_speech("test", voice_id="invalid_voice_id_xyz")
    if result["status"] == "error":
        print(f"✅ Error caught cleanly: {result['error_type']}: {result['error']}")
    else:
        # Some accounts may accept any voice ID — not necessarily a failure
        print(f"⚠️  Returned ok (voice may have fallen back): {result['voice_id']}")


async def main() -> None:
    print("═" * 60)
    print("  BEAN AI — TTS Agent Live Test")
    print("═" * 60)

    await test_full_audio()
    await test_streaming()
    await test_empty_text()
    await test_error_handling()

    print("\n═" * 60)
    print("  Done.")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())