#!/usr/bin/env python3
"""
BEAN AI — ESP32 Simulator with Laptop Audio Playback

Simulates the ESP32 robot from your laptop:
  - Mic input  → PCM16 → WebSocket → Server → Deepgram STT
  - Server     → play_audio URL    → HTTP MP3 stream → Laptop speakers

Push-to-talk:
    Press ENTER to start recording
    Press ENTER again to stop and trigger AI response
    Ctrl+C to quit

Usage:
    python test_esp32_sim.py --email you@email.com --password yourpass
    python test_esp32_sim.py --email you@email.com --password yourpass --url ws://localhost:8080/ws
    python test_esp32_sim.py --email you@email.com --password yourpass --wav test.wav
    python test_esp32_sim.py --device-id ESP32_BEAN_001 --url ws://localhost:8080/ws

Dependencies:
    pip install websockets pyaudio pygame httpx python-dotenv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger("bean_sim")

# ── Audio settings (must match Deepgram config) ───────────────────────────────
SAMPLE_RATE = 16000   # Hz
CHANNELS    = 1       # mono
SAMPLE_WIDTH = 2      # 16-bit
CHUNK_SIZE  = 1600    # 100ms of audio at 16kHz (1600 samples × 2 bytes = 3200 bytes)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEAN AI ESP32 Simulator")
    auth = p.add_mutually_exclusive_group()
    auth.add_argument("--email",     help="Supabase account email")
    auth.add_argument("--device-id", help="Pre-registered device_id (skips JWT auth)")
    p.add_argument("--password", help="Supabase account password (required with --email)")
    p.add_argument(
        "--url",
        default="ws://localhost:8080/ws",
        help="WebSocket server URL (default: ws://localhost:8080/ws)",
    )
    p.add_argument(
        "--wav",
        default=None,
        help="Path to a 16kHz mono WAV file to send instead of live mic",
    )
    p.add_argument(
        "--volume",
        type=int,
        default=80,
        help="Playback volume 0-100 (default 80)",
    )
    return p.parse_args()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────


async def get_jwt(email: str, password: str) -> str:
    """Sign in to Supabase and return a JWT access token."""
    import httpx

    url  = os.environ.get("SUPABASE_URL", "").rstrip("/")
    anon = os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not anon:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{url}/auth/v1/token?grant_type=password",
            headers={"apikey": anon, "Content-Type": "application/json"},
            json={"email": email, "password": password},
        )
    if r.status_code != 200:
        raise RuntimeError(f"Auth failed ({r.status_code}): {r.text}")

    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {r.text}")

    logger.info("✓ Signed in as %s", email)
    return token


# ─────────────────────────────────────────────────────────────────────────────
# MP3 playback via pygame
# ─────────────────────────────────────────────────────────────────────────────


def _init_pygame(volume: int) -> bool:
    """Initialise pygame mixer. Returns True if available."""
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.set_volume(max(0, min(100, volume)) / 100)
        logger.info("✓ Audio output ready (pygame mixer)")
        return True
    except Exception as exc:
        logger.warning("pygame not available — audio will not play: %s", exc)
        logger.warning("Install with: pip install pygame")
        return False


async def play_mp3_from_url(url: str, jwt: str | None = None) -> None:
    """Download MP3 stream from URL and play through laptop speakers."""
    try:
        import httpx
        import pygame
    except ImportError as exc:
        logger.warning("Cannot play audio: %s", exc)
        return

    print("  🔊 Downloading audio...", end="", flush=True)
    headers = {}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Stream the HTTP response (chunked) and buffer it
            audio_bytes = b""
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 404:
                    print(" [stream not found]")
                    return
                if response.status_code == 401:
                    print(" [401 unauthorized — JWT may have expired]")
                    return
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    audio_bytes += chunk

        if not audio_bytes:
            print(" [empty response]")
            return

        # Write to a temp MP3 file and play with pygame
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        print(f" {len(audio_bytes)//1024}KB — playing...", end="", flush=True)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

        print(" ✓")

    except asyncio.CancelledError:
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass
        raise
    except Exception as exc:
        print(f" [error: {exc}]")
    finally:
        try:
            os.unlink(tmp_path)  # type: ignore[possibly-undefined]
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Microphone input
# ─────────────────────────────────────────────────────────────────────────────


def mic_chunks(chunk_samples: int = CHUNK_SIZE):
    """Generator that yields raw PCM16 chunks from the default microphone."""
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError("pyaudio not installed — run: pip install pyaudio")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=chunk_samples,
    )
    try:
        while True:
            yield stream.read(chunk_samples, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def wav_chunks(path: str, chunk_samples: int = CHUNK_SIZE):
    """Yield PCM16 chunks from a 16kHz mono WAV file."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("WAV must be 16-bit PCM")
        if wf.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        logger.info(
            "WAV: %s  rate=%dHz  frames=%d",
            path, wf.getframerate(), wf.getnframes(),
        )
        while True:
            data = wf.readframes(chunk_samples)
            if not data:
                break
            yield data


# ─────────────────────────────────────────────────────────────────────────────
# Push-to-talk keyboard thread
# ─────────────────────────────────────────────────────────────────────────────


class PushToTalk:
    """Monitors Enter key to toggle recording state."""

    def __init__(self) -> None:
        self._recording = threading.Event()
        self._quit      = threading.Event()
        self._thread    = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._quit.set()

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    def _loop(self) -> None:
        while not self._quit.is_set():
            try:
                input()  # blocks until Enter
            except EOFError:
                break
            if self._recording.is_set():
                self._recording.clear()   # stop recording
            else:
                self._recording.set()     # start recording

    def wait_for_start(self) -> None:
        """Block until the user presses Enter to begin recording."""
        self._recording.wait()

    def wait_for_stop(self) -> None:
        """Block until the user presses Enter again to stop."""
        while self._recording.is_set():
            import time
            time.sleep(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator
# ─────────────────────────────────────────────────────────────────────────────


async def run_simulator(
    ws_url: str,
    *,
    jwt: str | None = None,
    device_id: str | None = None,
    wav_path: str | None = None,
    volume: int = 80,
) -> None:
    try:
        import websockets
    except ImportError:
        raise RuntimeError("websockets not installed — run: pip install websockets")

    pygame_ok = _init_pygame(volume)

    # Build connection URL and headers
    connect_kwargs: dict = {
        "open_timeout": 30,
        "ping_interval": 20,
        "ping_timeout": 10,
    }

    if device_id:
        final_url = f"{ws_url}?device_id={device_id}"
        logger.info("Connecting as device_id=%s ...", device_id)
    elif jwt:
        final_url = ws_url
        connect_kwargs["subprotocols"] = [f"bearer.{jwt}"]
        logger.info("Connecting with JWT ...")
    else:
        raise RuntimeError("Must provide either jwt or device_id")

    async with websockets.connect(final_url, **connect_kwargs) as ws:
        logger.info("✓ WebSocket connected to %s", ws_url)

        # ── Receive loop ──────────────────────────────────────────────────────
        play_audio_task: asyncio.Task | None = None
        tts_buffer: bytearray = bytearray()
        in_tts: bool = False

        async def _play_tts_buffer(buf: bytes) -> None:
            """Write buffered TTS binary frames to a temp file and play."""
            if not buf or not pygame_ok:
                return
            try:
                import pygame
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(buf)
                    tmp_path = f.name
                print(f"  🔊 Playing TTS ({len(buf)//1024}KB)...", end="", flush=True)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
                print(" done")
            except Exception as exc:
                print(f" [error: {exc}]")
            finally:
                try:
                    os.unlink(tmp_path)  # type: ignore[possibly-undefined]
                except Exception:
                    pass

        async def receive_loop() -> None:
            nonlocal play_audio_task, tts_buffer, in_tts
            async for message in ws:
                if isinstance(message, bytes):
                    if in_tts:
                        # Binary TTS chunk — buffer until response_audio_end
                        tts_buffer.extend(message)
                    else:
                        # Binary = background music chunk
                        print("🎵", end="", flush=True)
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "connected":
                    sid = data.get("session_id", "")[:8]
                    print(f"\n✓ Session started  [id={sid}]")

                elif msg_type == "transcript_partial":
                    text = data.get("text", "")
                    print(f"\r  🎤 Partial: {text:<60}", end="", flush=True)

                elif msg_type == "transcript_final":
                    text = data.get("text", "")
                    conf = data.get("confidence", 0)
                    print(f"\n  📝 Final [{conf:.0%}]: {text}")

                elif msg_type == "response_text":
                    text  = data.get("text", "")
                    route = data.get("route", "?")
                    print(f"\n  🤖 BEAN [{route}]: {text}")
                    # Next binary frames will be TTS audio
                    in_tts = True
                    tts_buffer.clear()

                elif msg_type == "response_audio_end":
                    # TTS stream complete — play the buffered audio
                    in_tts = False
                    if tts_buffer:
                        buf = bytes(tts_buffer)
                        tts_buffer.clear()
                        if play_audio_task and not play_audio_task.done():
                            play_audio_task.cancel()
                        play_audio_task = asyncio.create_task(_play_tts_buffer(buf))

                elif msg_type == "play_audio":
                    # HTTP URL mode (feature/fixing server)
                    in_tts = False
                    url = data.get("url", "")
                    if url and pygame_ok:
                        if play_audio_task and not play_audio_task.done():
                            play_audio_task.cancel()
                        play_audio_task = asyncio.create_task(play_mp3_from_url(url, jwt=jwt))
                    elif url:
                        print(f"\n  🔊 [audio URL — pygame not available]: {url}")

                elif msg_type == "alert_notification":
                    print(f"\n  🚨 ALERT: {data.get('message', '')}")

                elif msg_type == "music_start":
                    title = data.get("title", "Unknown")
                    mood  = data.get("mood", "")
                    print(f"\n  🎵 Music: {title}  [{mood}]")

                elif msg_type == "music_stopped":
                    print("\n  ⏹  Music stopped")

                elif msg_type == "music_volume":
                    print(f"\n  🔉 Volume: {data.get('volume')}%")

                elif msg_type == "music_unavailable":
                    print(f"\n  ⚠  Music unavailable: {data.get('message','')}")

                elif msg_type == "error":
                    code = data.get("code", "")
                    msg  = data.get("message", "")
                    print(f"\n  ❌ Error [{code}]: {msg}")

                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))

                elif msg_type == "pong":
                    pass

                else:
                    logger.debug("Server: %s", data)

        receive_task = asyncio.create_task(receive_loop())
        loop = asyncio.get_event_loop()

        try:
            if wav_path:
                # ── WAV file mode ─────────────────────────────────────────────
                print(f"\n  Sending WAV file: {wav_path}")
                for chunk in wav_chunks(wav_path):
                    await ws.send(chunk)
                    await asyncio.sleep(0.1)   # 100ms per chunk
                print("  ✓ WAV sent — sending stop_recording")
                await ws.send(json.dumps({"type": "stop_recording"}))
                # Wait for response + playback
                await asyncio.sleep(15)

            else:
                # ── Live mic push-to-talk mode ────────────────────────────────
                ptt = PushToTalk()
                ptt.start()

                print("\n" + "=" * 52)
                print("  Press  ENTER  to start talking")
                print("  Press  ENTER  again to stop and get response")
                print("  Ctrl+C to quit")
                print("=" * 52)

                while True:
                    # Wait for user to press Enter to start
                    print("\n  [IDLE] Press ENTER to start recording...", flush=True)
                    await asyncio.get_event_loop().run_in_executor(
                        None, ptt.wait_for_start
                    )

                    print("  [RECORDING] 🎤 Speak now — press ENTER to stop", flush=True)

                    # Stream mic audio in a background thread
                    stop_flag = threading.Event()

                    def stream_mic() -> None:
                        try:
                            for chunk in mic_chunks():
                                if stop_flag.is_set():
                                    break
                                asyncio.run_coroutine_threadsafe(
                                    ws.send(chunk), loop
                                )
                        except Exception as exc:
                            logger.error("Mic stream error: %s", exc)

                    mic_thread = threading.Thread(target=stream_mic, daemon=True)
                    mic_thread.start()

                    # Wait for Enter to stop recording
                    await asyncio.get_event_loop().run_in_executor(
                        None, ptt.wait_for_stop
                    )

                    # Stop mic and tell server
                    stop_flag.set()
                    mic_thread.join(timeout=1.0)

                    print("  [WAITING] ⏳ Processing...", flush=True)
                    await ws.send(json.dumps({"type": "stop_recording"}))

                    # Wait a bit before next round so response can play
                    await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n  Stopping...")
        finally:
            try:
                await ws.send(json.dumps({"type": "end_session"}))
            except Exception:
                pass
            receive_task.cancel()
            if play_audio_task and not play_audio_task.done():
                play_audio_task.cancel()
            try:
                await asyncio.gather(receive_task, return_exceptions=True)
            except Exception:
                pass

    logger.info("✓ Session ended")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    args = parse_args()
    load_env()

    print()
    print("=" * 52)
    print("   BEAN AI - ESP32 Simulator v2")
    print("   Microphone -> Server -> Laptop Speakers")
    print("=" * 52)

    jwt: str | None = None
    device_id: str | None = args.device_id

    if args.email:
        if not args.password:
            print("ERROR: --password is required with --email")
            sys.exit(1)
        try:
            jwt = await get_jwt(args.email, args.password)
        except Exception as exc:
            logger.error("Auth failed: %s", exc)
            print(
                "\nTip: Make sure SUPABASE_URL and SUPABASE_ANON_KEY are set in .env\n"
                "     and you have a registered user in Supabase Auth."
            )
            sys.exit(1)
    elif not device_id:
        print("ERROR: provide --email/--password or --device-id")
        sys.exit(1)

    try:
        await run_simulator(
            ws_url=args.url,
            jwt=jwt,
            device_id=device_id,
            wav_path=args.wav,
            volume=args.volume,
        )
    except Exception as exc:
        logger.error("Simulator error: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
