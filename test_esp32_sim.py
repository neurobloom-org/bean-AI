#!/usr/bin/env python3
"""
BEAN AI — Fake ESP32 Simulator
Simulates the ESP32 hardware client so you can test the full pipeline
(STT → Orchestrator → TTS) from your PC without physical hardware.

Requirements:
    pip install websockets pyaudio supabase python-dotenv

Usage:
    python test_esp32_sim.py --email you@email.com --password yourpassword
    python test_esp32_sim.py --email you@email.com --password yourpassword --url ws://localhost:8080/ws
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import threading
import wave
from pathlib import Path

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger("esp32_sim")

# ── Audio settings (must match what Deepgram expects) ─────────────────────────
SAMPLE_RATE = 16000      # 16kHz
CHANNELS = 1             # Mono
SAMPLE_WIDTH = 2         # 16-bit PCM
CHUNK_SIZE = 3200        # 200ms chunks (3200 bytes = 1600 samples * 2 bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEAN AI Fake ESP32 Simulator")
    parser.add_argument("--email", required=True, help="Supabase user email")
    parser.add_argument("--password", required=True, help="Supabase user password")
    parser.add_argument(
        "--url",
        default="ws://localhost:8080/ws",
        help="WebSocket URL (default: ws://localhost:8080/ws)",
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="Optional: path to a WAV file to send instead of live mic",
    )
    return parser.parse_args()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


async def get_supabase_token(email: str, password: str) -> str:
    """Sign in to Supabase and return JWT access token."""
    import httpx

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not supabase_url or not supabase_anon_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

    url = f"{supabase_url.rstrip('/')}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": supabase_anon_key,
        "Content-Type": "application/json",
    }
    body = {"email": email, "password": password}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=body, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(
            f"Supabase sign-in failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {data}")

    logger.info("✓ Signed in as %s", email)
    return token


class AudioPlayer:
    """Plays PCM16 audio through speakers."""

    def __init__(self):
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._playing = False
        self._thread: threading.Thread | None = None

    def add_chunk(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)
        if not self._playing:
            self._start()

    def _start(self) -> None:
        self._playing = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self) -> None:
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=22050,  # ElevenLabs outputs at 22050Hz
                output=True,
            )
            while True:
                with self._lock:
                    if not self._buffer:
                        break
                    chunk = bytes(self._buffer[:4096])
                    self._buffer = self._buffer[4096:]
                stream.write(chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as exc:
            logger.error("Audio playback error: %s", exc)
        finally:
            self._playing = False


class MicRecorder:
    """Records from microphone and yields PCM16 chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self._running = False

    def record_chunks(self):
        """Generator that yields PCM16 audio chunks from mic."""
        try:
            import pyaudio
        except ImportError:
            raise RuntimeError("pyaudio not installed. Run: pip install pyaudio")

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.chunk_size // SAMPLE_WIDTH,
        )

        logger.info("🎤 Microphone open — speak now! (Ctrl+C to stop)")
        self._running = True

        try:
            while self._running:
                data = stream.read(
                    self.chunk_size // SAMPLE_WIDTH,
                    exception_on_overflow=False,
                )
                yield data
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def stop(self) -> None:
        self._running = False


def load_wav_chunks(wav_path: str, chunk_size: int = CHUNK_SIZE):
    """Yield PCM16 chunks from a WAV file."""
    with wave.open(wav_path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("WAV file must be 16-bit PCM")
        if wf.getnchannels() != 1:
            raise ValueError("WAV file must be mono")
        logger.info(
            "📁 Loading WAV: %s (rate=%dHz, frames=%d)",
            wav_path,
            wf.getframerate(),
            wf.getnframes(),
        )
        while True:
            data = wf.readframes(chunk_size // SAMPLE_WIDTH)
            if not data:
                break
            yield data


async def run_simulator(
    ws_url: str,
    token: str,
    wav_path: str | None = None,
) -> None:
    """Main simulator loop."""
    try:
        import websockets
    except ImportError:
        raise RuntimeError("websockets not installed. Run: pip install websockets")

    logger.info("Connecting to %s ...", ws_url)

    player = AudioPlayer()
    audio_buffer = bytearray()
    receiving_tts = False

    # Pass token as query param — subprotocol with long JWT causes timeout
    ws_url_with_token = f"{ws_url}?token={token}"

    async with websockets.connect(
        ws_url_with_token,
        open_timeout=30,
        ping_interval=20,
        ping_timeout=10,
    ) as ws:
        logger.info("✓ WebSocket connected")

        async def receive_loop():
            nonlocal receiving_tts, audio_buffer
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        # TTS audio chunk
                        player.add_chunk(message)
                        print("🔊", end="", flush=True)
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type", "")

                        if msg_type == "connected":
                            logger.info(
                                "✓ Session started: %s", data.get("session_id", "")[:8]
                            )

                        elif msg_type == "listening":
                            print("\n👂 BEAN is listening...")

                        elif msg_type == "active_listen":
                            filler = data.get("text", "")
                            if filler:
                                print(f"\n💬 [filler] {filler}")

                        elif msg_type == "response_text":
                            text = data.get("text", "")
                            route = data.get("route", "unknown")
                            print(f"\n🤖 BEAN [{route}]: {text}")

                        elif msg_type == "tts_start":
                            print("\n🔊 [TTS streaming...]", end="", flush=True)

                        elif msg_type == "tts_end":
                            print(" [done]")

                        elif msg_type == "alert_notification":
                            print(f"\n🚨 ALERT: {data.get('message', '')}")

                        elif msg_type == "error":
                            logger.error(
                                "Server error [%s]: %s",
                                data.get("code", ""),
                                data.get("message", ""),
                            )

                        elif msg_type == "pong":
                            pass  # keepalive

                        else:
                            logger.debug("Server message: %s", data)

            except websockets.ConnectionClosed as exc:
                logger.info("Connection closed: %s", exc)

        # Start receive loop in background
        receive_task = asyncio.create_task(receive_loop())

        # Send audio
        try:
            if wav_path:
                # Send WAV file chunks
                for chunk in load_wav_chunks(wav_path):
                    await ws.send(chunk)
                    await asyncio.sleep(0.2)  # 200ms per chunk (real-time)
                logger.info("✓ WAV file sent")
                # Wait a bit for response
                await asyncio.sleep(10)
            else:
                # Live mic recording
                recorder = MicRecorder()
                loop = asyncio.get_event_loop()

                print("\n" + "="*50)
                print("🎤 SPEAK TO BEAN — Press Ctrl+C to stop")
                print("="*50 + "\n")

                def send_mic_audio():
                    try:
                        for chunk in recorder.record_chunks():
                            asyncio.run_coroutine_threadsafe(
                                ws.send(chunk), loop
                            )
                    except Exception as exc:
                        logger.error("Mic error: %s", exc)

                mic_thread = threading.Thread(target=send_mic_audio, daemon=True)
                mic_thread.start()

                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                finally:
                    recorder.stop()

        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            receive_task.cancel()
            try:
                await ws.send(json.dumps({"type": "end_session"}))
            except Exception:
                pass

    logger.info("✓ Session ended")


async def main() -> None:
    args = parse_args()
    load_env()

    print("\n" + "="*50)
    print("  BEAN AI — Fake ESP32 Simulator")
    print("="*50)

    # Get auth token
    try:
        token = await get_supabase_token(args.email, args.password)
    except Exception as exc:
        logger.error("Auth failed: %s", repr(exc))
        import traceback
        traceback.print_exc()
        logger.info(
            "Tip: Make sure you have a user registered in Supabase. "
            "Go to Supabase Dashboard → Authentication → Users → Add user"
        )
        sys.exit(1)

    # Run simulator
    try:
        await run_simulator(
            ws_url=args.url,
            token=token,
            wav_path=args.wav,
        )
    except Exception as exc:
        logger.error("Simulator error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())