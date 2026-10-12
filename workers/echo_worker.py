"""CPU protocol probe, not a speech or ASR model.

Run: python -m workers.echo_worker --port 9001
Binary input/output is normalized mono little-endian float32 at 24 kHz. In a
custom model worker, replace the sender's gain operation with streaming model
inference while leaving receiver and sender independent.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from duplex_arena.adapters import FRAME_SIZE, SAMPLE_RATE, validate_pcm


async def echo_session(websocket, *, gain: float = 0.5, delay_ms: float = 0) -> None:
    pending: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=32)
    sender_task: asyncio.Task | None = None
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=10)
        if not isinstance(raw, str):
            raise ValueError("First message must be a JSON init event")
        init = json.loads(raw)
        expected = {"type": "init", "protocol": "duplex-arena.pcm.v1",
                    "sample_rate": SAMPLE_RATE, "channels": 1,
                    "frame_size": FRAME_SIZE, "format": "f32le"}
        if not isinstance(init, dict) or any(init.get(key) != value for key, value in expected.items()):
            raise ValueError("Expected duplex-arena.pcm.v1, 24kHz, mono, 1920 samples, f32le")
        await websocket.send(json.dumps({"type": "status", "state": "ready", "worker": "cpu-echo"}))

        async def sender() -> None:
            first = True
            while True:
                pcm = await pending.get()
                try:
                    if delay_ms:
                        await asyncio.sleep(delay_ms / 1000)
                    await websocket.send((pcm * gain).astype("<f4").tobytes())
                    if first:
                        await websocket.send(json.dumps({"type": "text", "text": "CPU echo transport; no speech model or ASR loaded."}))
                        first = False
                finally:
                    pending.task_done()

        sender_task = asyncio.create_task(sender())
        async for message in websocket:
            if sender_task.done():
                sender_task.result()
            if not isinstance(message, bytes) or len(message) != FRAME_SIZE * 4:
                raise ValueError(f"Each audio message must be {FRAME_SIZE * 4} binary bytes")
            audio = validate_pcm(np.frombuffer(message, dtype="<f4"))
            try:
                pending.put_nowait(audio)
            except asyncio.QueueFull as exc:
                raise ValueError("CPU echo queue overflow") from exc
    except (ValueError, asyncio.TimeoutError) as exc:
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
            await websocket.close(code=1008, reason="Invalid audio protocol")
        except ConnectionClosed:
            pass
    except ConnectionClosed:
        pass
    finally:
        if sender_task is not None:
            sender_task.cancel()
            await asyncio.gather(sender_task, return_exceptions=True)


async def run(host: str, port: int, gain: float, delay_ms: float) -> None:
    async def handler(websocket):
        await echo_session(websocket, gain=gain, delay_ms=delay_ms)

    async with serve(handler, host, port, max_size=FRAME_SIZE * 4, max_queue=16):
        print(f"CPU echo worker listening at ws://{host}:{port}/stream (no model)", flush=True)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=float, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.gain) or not 0 <= args.gain <= 1:
        parser.error("gain must be finite and in [0, 1]")
    if not np.isfinite(args.delay_ms) or not 0 <= args.delay_ms <= 10000:
        parser.error("delay-ms must be finite and in [0, 10000]")
    asyncio.run(run(args.host, args.port, args.gain, args.delay_ms))


if __name__ == "__main__":
    main()
