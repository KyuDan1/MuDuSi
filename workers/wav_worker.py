"""A deterministic speech/noise stimulus source, not a language model.

Run: python -m workers.wav_worker --wav your-seed.wav --port 9001
The file must be 24 kHz mono uncompressed PCM16 WAV. Playback advances one 80 ms
block per received simulation input. Every new connection starts at the file's
beginning. After playback, the source emits silence unless --loop is requested.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from duplex_arena.adapters import FRAME_SIZE, SAMPLE_RATE, validate_pcm


def load_stimulus(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth(), source.getcomptype()) != (1, SAMPLE_RATE, 2, "NONE"):
            raise ValueError("Stimulus must be an uncompressed 24 kHz mono PCM16 WAV")
        if not 0 < source.getnframes() <= SAMPLE_RATE * 600:
            raise ValueError("Stimulus duration must be positive and at most 10 minutes")
        raw = source.readframes(source.getnframes())
        if len(raw) != source.getnframes() * 2:
            raise ValueError("WAV data is truncated")
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def stimulus_session(websocket, audio: np.ndarray, *, loop: bool = False) -> None:
    cursor = 0
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=10)
        if not isinstance(raw, str):
            raise ValueError("First message must be a JSON init event")
        init = json.loads(raw)
        expected = {"type": "init", "protocol": "duplex-arena.pcm.v1",
                    "sample_rate": SAMPLE_RATE, "channels": 1,
                    "frame_size": FRAME_SIZE, "format": "f32le"}
        if not isinstance(init, dict) or any(init.get(key) != value for key, value in expected.items()):
            raise ValueError("Expected duplex-arena.pcm.v1 24 kHz mono PCM initialization")
        await websocket.send(json.dumps({"type": "status", "state": "ready", "worker": "wav-stimulus"}))
        async for message in websocket:
            if not isinstance(message, bytes) or len(message) != FRAME_SIZE * 4:
                raise ValueError("Expected one 1920-sample float32 binary input block")
            validate_pcm(np.frombuffer(message, dtype="<f4"))
            frame = np.zeros(FRAME_SIZE, dtype=np.float32)
            offset = 0
            while offset < FRAME_SIZE and cursor < audio.size:
                count = min(FRAME_SIZE - offset, audio.size - cursor)
                frame[offset:offset + count] = audio[cursor:cursor + count]
                cursor += count
                offset += count
                if loop and cursor == audio.size:
                    cursor = 0
            await websocket.send(frame.astype("<f4", copy=False).tobytes())
    except (ValueError, asyncio.TimeoutError) as exc:
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
            await websocket.close(code=1008, reason="Invalid stimulus protocol")
        except ConnectionClosed:
            pass
    except ConnectionClosed:
        pass


async def run(host: str, port: int, audio: np.ndarray, loop: bool) -> None:
    async def handler(websocket):
        await stimulus_session(websocket, audio, loop=loop)
    async with serve(handler, host, port, max_size=FRAME_SIZE * 4, max_queue=16):
        print(f"WAV stimulus at ws://{host}:{port}/stream: {audio.size / SAMPLE_RATE:.2f}s, loop={loop}", flush=True)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9001, type=int)
    parser.add_argument("--loop", action="store_true", help="Repeat the file; default plays once")
    parser.add_argument("--delay-seconds", type=float, default=0, help="Initial silence per connection")
    args = parser.parse_args()
    if not np.isfinite(args.delay_seconds) or not 0 <= args.delay_seconds <= 600:
        parser.error("delay-seconds must be finite and in [0, 600]")
    try:
        audio = load_stimulus(args.wav)
    except (ValueError, wave.Error, OSError) as exc:
        parser.error(str(exc))
    if args.delay_seconds:
        audio = np.concatenate((np.zeros(round(args.delay_seconds * SAMPLE_RATE), dtype=np.float32), audio))
    asyncio.run(run(args.host, args.port, audio, args.loop))


if __name__ == "__main__":
    main()
