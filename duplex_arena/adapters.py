"""Full-duplex transports. No model weights or GPU libraries run in the arena.

Each adapter owns independent send/receive tasks. The simulation never waits for
generated audio: pull_audio returns available samples and pads an underrun with
silence, while transport/model failures raise AdapterError instead of pretending
that a model is still connected.
"""

from __future__ import annotations

import asyncio
from collections import deque
import importlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920
AGENT_IDS = frozenset({"persona", "moshi", "custom"})


class AdapterError(RuntimeError):
    """Connection, protocol, codec, or model failure requiring caller attention."""


class AdapterBackpressureError(AdapterError):
    """Input cannot keep up; stop the rollout rather than silently dropping input."""


def validate_pcm(pcm: np.ndarray, frame_size: int | None = FRAME_SIZE) -> np.ndarray:
    """Validate mono normalized samples; copy to protect against caller mutation."""
    audio = np.asarray(pcm)
    if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("PCM must be a one-dimensional floating-point mono array")
    if frame_size is not None and audio.size != frame_size:
        raise ValueError(f"Expected {frame_size} audio samples, received {audio.size}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("PCM contains NaN or infinity")
    if audio.size and np.max(np.abs(audio)) > 1.00001:
        raise ValueError("PCM samples must be normalized to [-1, 1]")
    return np.array(audio, dtype=np.float32, order="C", copy=True)


class _AudioBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.chunks: deque[np.ndarray] = deque()

    def append(self, audio: np.ndarray) -> int:
        if audio.size:
            self.chunks.append(audio)
            self.size += audio.size
        dropped = max(0, self.size - self.capacity)
        self._take(dropped)
        return dropped

    def _take(self, count: int) -> np.ndarray:
        result = np.zeros(count, dtype=np.float32)
        cursor = 0
        while cursor < count and self.chunks:
            chunk = self.chunks.popleft()
            take = min(count - cursor, chunk.size)
            result[cursor:cursor + take] = chunk[:take]
            if take < chunk.size:
                self.chunks.appendleft(chunk[take:])
            self.size -= take
            cursor += take
        return result

    def pull(self, count: int) -> tuple[np.ndarray, int]:
        available = min(count, self.size)
        return self._take(count), count - available


class StreamAdapter:
    """Streaming interface for native Moshi/PersonaPlex or a custom PCM worker.

    start/push_audio/close are asynchronous; pull_audio and drain_text are
    synchronous and nonblocking. Use one adapter instance per model session.
    Queue bounds are intentional: input overflow is an error; old output is
    discarded on overflow and counted, preventing unlimited latency growth.
    """

    def __init__(self, url: str, *, max_input_frames: int = 32,
                 max_output_frames: int = 64, startup_timeout: float = 120.0):
        parsed = urlsplit(url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError("Worker URL must be an absolute ws:// or wss:// URL")
        if parsed.username or parsed.password:
            raise ValueError("Put worker authentication in a private proxy, not a URL")
        if not isinstance(max_input_frames, int) or not 1 <= max_input_frames <= 1000:
            raise ValueError("max_input_frames must be in [1, 1000]")
        if not isinstance(max_output_frames, int) or not 1 <= max_output_frames <= 1000:
            raise ValueError("max_output_frames must be in [1, 1000]")
        if not np.isfinite(startup_timeout) or not 0 < startup_timeout <= 600:
            raise ValueError("startup_timeout must be in (0, 600]")
        self.url = url
        self.startup_timeout = float(startup_timeout)
        self.max_input_frames = max_input_frames
        self.max_output_frames = max_output_frames
        self._ws: Any = None
        self._tasks: list[asyncio.Task] = []
        self._closing = False
        self._ready = False
        self._error: str | None = None
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._worker_name: str | None = None
        self._input: asyncio.Queue[np.ndarray] = asyncio.Queue(self.max_input_frames)
        self._audio = _AudioBuffer(self.max_output_frames * FRAME_SIZE)
        self._text: deque[str] = deque(maxlen=256)
        self._counters = {
            "input_frames": 0, "sent_frames": 0, "received_samples": 0,
            "played_samples": 0, "underrun_frames": 0, "underrun_samples": 0,
            "output_dropped_samples": 0, "input_overflow_frames": 0,
            "text_dropped_messages": 0, "decoded_clipped_samples": 0,
        }

    @property
    def ready(self) -> bool:
        return self._ready and self._error is None and not self._closing

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def stats(self) -> dict[str, Any]:
        return {**self._counters, "ready": self.ready, "error": self.error,
                "worker_name": self._worker_name,
                "queued_input_frames": self._input.qsize(),
                "queued_output_samples": self._audio.size}

    def _check_ready(self) -> None:
        if self._error:
            raise AdapterError(self._error)
        if not self.ready:
            raise AdapterError("Adapter has not completed start() or is closed")

    async def start(self) -> None:
        if self.ready:
            return
        if self._ws is not None:
            await self.close()
        self._closing = False
        self._error = None
        self._reset_buffers()
        try:
            from websockets.asyncio.client import connect
            self._prepare()
            self._ws = await connect(self.url, open_timeout=self.startup_timeout,
                                     max_size=1_048_576, max_queue=16,
                                     close_timeout=2, ping_timeout=30)
            await asyncio.wait_for(self._handshake(), self.startup_timeout)
            self._ready = True
            self._tasks = [asyncio.create_task(self._send_loop()),
                           asyncio.create_task(self._receive_loop())]
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception as exc:
            self._error = f"{type(self).__name__} startup failed: {exc}"
            await self.close()
            raise AdapterError(self._error) from exc

    def _prepare(self) -> None:
        pass

    async def _handshake(self) -> None:
        raise NotImplementedError

    def _encode(self, pcm: np.ndarray) -> bytes:
        raise NotImplementedError

    def _on_message(self, message: str | bytes) -> None:
        raise NotImplementedError

    async def _send_loop(self) -> None:
        try:
            while True:
                pcm = await self._input.get()
                try:
                    payload = self._encode(pcm)
                    if payload:
                        await self._ws.send(payload)
                    self._counters["sent_frames"] += 1
                finally:
                    self._input.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(f"Worker send failed: {exc}")

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                self._on_message(message)
            if not self._closing:
                await self._fail("Worker closed the streaming session")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(f"Worker receive failed: {exc}")

    async def _fail(self, message: str) -> None:
        if self._closing:
            return
        self._error = self._error or message
        self._ready = False
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def push_audio(self, pcm: np.ndarray) -> None:
        self._check_ready()
        audio = validate_pcm(pcm)
        try:
            self._input.put_nowait(audio)
        except asyncio.QueueFull as exc:
            self._counters["input_overflow_frames"] += 1
            self._error = "Worker input queue overflow; rollout cannot preserve timing"
            self._ready = False
            raise AdapterBackpressureError(self._error) from exc
        self._counters["input_frames"] += 1

    def _append_audio(self, pcm: np.ndarray) -> None:
        audio = validate_pcm(pcm, frame_size=None)
        self._counters["received_samples"] += audio.size
        self._counters["output_dropped_samples"] += self._audio.append(audio)

    def _append_text(self, text: str) -> None:
        if len(self._text) == self._text.maxlen:
            self._counters["text_dropped_messages"] += 1
        self._text.append(text)

    def pull_audio(self, frame_size: int = FRAME_SIZE) -> np.ndarray:
        self._check_ready()
        if not isinstance(frame_size, int) or not 0 < frame_size <= SAMPLE_RATE * 10:
            raise ValueError("frame_size must be a positive sample count <= 240000")
        audio, missing = self._audio.pull(frame_size)
        self._counters["played_samples"] += frame_size - missing
        if missing:
            self._counters["underrun_frames"] += 1
            self._counters["underrun_samples"] += missing
        return audio

    def drain_text(self) -> list[str]:
        self._check_ready()
        result = list(self._text)
        self._text.clear()
        return result

    async def close(self) -> None:
        self._closing = True
        self._ready = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        while not self._input.empty():
            self._input.get_nowait()
            self._input.task_done()
        self._audio = _AudioBuffer(self.max_output_frames * FRAME_SIZE)
        self._text.clear()


class PCMWebSocketAdapter(StreamAdapter):
    """24kHz mono float32 little-endian WebSocket worker protocol v1."""

    async def _handshake(self) -> None:
        await self._ws.send(json.dumps({"type": "init", "protocol": "duplex-arena.pcm.v1",
                                        "sample_rate": SAMPLE_RATE, "channels": 1,
                                        "frame_size": FRAME_SIZE, "format": "f32le"}))
        message = await self._ws.recv()
        if not isinstance(message, str):
            raise AdapterError("PCM worker must acknowledge init with JSON status")
        status = json.loads(message)
        if not isinstance(status, dict) or status.get("type") != "status" or status.get("state") != "ready":
            raise AdapterError(f"PCM worker rejected init: {message[:300]}")
        worker_name = status.get("worker")
        if isinstance(worker_name, str):
            self._worker_name = worker_name[:256]

    def _encode(self, pcm: np.ndarray) -> bytes:
        return pcm.astype("<f4", copy=False).tobytes()

    def _on_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            if not message or len(message) % 4:
                raise AdapterError("Worker audio must contain complete float32 samples")
            self._append_audio(np.frombuffer(message, dtype="<f4"))
            return
        event = json.loads(message)
        if not isinstance(event, dict):
            raise AdapterError("Worker JSON event must be an object")
        kind = event.get("type")
        if kind == "text" and isinstance(event.get("text"), str):
            self._append_text(event["text"])
        elif kind == "error":
            raise AdapterError(str(event.get("message", "Unspecified worker error")))
        elif kind == "status" and event.get("state") in {"ready", "listening", "speaking"}:
            return
        else:
            raise AdapterError(f"Unknown worker event: {event!r}")


class MoshiWebSocketAdapter(StreamAdapter):
    """Bridge to official moshi.server's /api/chat binary Opus protocol."""

    def _prepare(self) -> None:
        try:
            sphn = importlib.import_module("sphn")
        except ImportError as exc:
            raise AdapterError("Native models require optional codec: pip install 'duplex-arena[models]'") from exc
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

    async def _handshake(self) -> None:
        message = await self._ws.recv()
        if message != b"\x00":
            raise AdapterError("Expected native Moshi/PersonaPlex handshake byte 0")

    def _encode(self, pcm: np.ndarray) -> bytes:
        payload = self._opus_writer.append_pcm(pcm)
        # sphn 0.1 uses append + read; sphn 0.2 returns bytes directly.
        if payload is None:
            payload = self._opus_writer.read_bytes()
        return b"\x01" + payload if payload else b""

    def _on_message(self, message: str | bytes) -> None:
        if not isinstance(message, bytes) or not message:
            raise AdapterError("Native worker messages must be nonempty binary frames")
        kind, payload = message[0], message[1:]
        if kind == 1:
            pcm = self._opus_reader.append_bytes(payload)
            if pcm is None:
                pcm = self._opus_reader.read_pcm()
            if np.asarray(pcm).size:
                # Lossy decoding can overshoot a valid normalized source. Apply
                # the playback DAC limit explicitly and expose clipping counts.
                self._counters["decoded_clipped_samples"] += int(np.count_nonzero(np.abs(pcm) > 1))
                self._append_audio(np.clip(pcm, -1, 1))
        elif kind == 2:
            self._append_text(payload.decode("utf-8", errors="strict"))
        else:
            raise AdapterError(f"Unexpected native worker message kind {kind}")


class RaonWebSocketAdapter(StreamAdapter):
    """Official Raon realtime protocol: kind byte followed by float32 PCM."""

    async def _handshake(self) -> None:
        message = await self._ws.recv()
        if not isinstance(message, bytes) or not message or message[0] != 0:
            raise AdapterError("Expected Raon ready frame")
        self._worker_name = "Raon-SpeechChat-9B (SGLang)"

    def _encode(self, pcm: np.ndarray) -> bytes:
        return b"\x01" + pcm.astype("<f4", copy=False).tobytes()

    def _on_message(self, message: str | bytes) -> None:
        if not isinstance(message, bytes) or not message:
            raise AdapterError("Expected binary Raon frame")
        kind, payload = message[0], message[1:]
        if kind == 1:
            if not payload or len(payload) % 4:
                raise AdapterError("Invalid Raon PCM length")
            self._append_audio(np.frombuffer(payload, dtype="<f4"))
        elif kind == 2:
            self._append_text(payload.decode("utf-8"))
        elif kind in (5, 6):
            raise AdapterError("Raon stopped: " + payload.decode("utf-8", errors="replace"))
        elif kind != 8:  # PONG
            raise AdapterError(f"Unexpected Raon message kind {kind}")


class PersonaPlexWebSocketAdapter(MoshiWebSocketAdapter):
    def __init__(self, url: str, *, voice_prompt: str = "NATF2.pt",
                 text_prompt: str = "You enjoy having a good conversation.", **kwargs: Any):
        if not isinstance(voice_prompt, str) or not voice_prompt or "/" in voice_prompt or "\\" in voice_prompt:
            raise ValueError("voice_prompt must name one voice file in the worker's voice directory")
        if not isinstance(text_prompt, str):
            raise ValueError("text_prompt must be a string")
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.update(voice_prompt=voice_prompt, text_prompt=text_prompt)
        super().__init__(urlunsplit(parts._replace(query=urlencode(query))), **kwargs)


def build_adapters(config_path: str | Path) -> dict[str, StreamAdapter]:
    """Load explicit agent transports. Missing agents remain caller-controlled.

    No model starts until its adapter's start() is awaited. Unknown backends or
    agent IDs and duplicate IDs are errors; a typo never activates a fake model.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("agents"), list):
        raise ValueError("Worker config must be an object containing an agents list")
    adapters: dict[str, StreamAdapter] = {}
    for item in config["agents"]:
        if not isinstance(item, dict):
            raise ValueError("Every agent config must be an object")
        agent_id = item.get("id")
        if not isinstance(agent_id, str) or agent_id not in AGENT_IDS:
            raise ValueError(f"Unknown agent ID {agent_id!r}; use persona, moshi, or custom")
        if agent_id in adapters:
            raise ValueError(f"Duplicate agent ID: {agent_id}")
        backend, url = item.get("backend"), item.get("url")
        if not isinstance(url, str):
            raise ValueError(f"Agent {agent_id} requires a worker URL")
        kwargs = {key: item[key] for key in ("max_input_frames", "max_output_frames", "startup_timeout") if key in item}
        if backend == "pcm":
            adapter = PCMWebSocketAdapter(url, **kwargs)
        elif backend == "moshi":
            adapter = MoshiWebSocketAdapter(url, **kwargs)
        elif backend == "personaplex":
            for key in ("voice_prompt", "text_prompt"):
                if key in item:
                    kwargs[key] = item[key]
            adapter = PersonaPlexWebSocketAdapter(url, **kwargs)
        elif backend == "raon":
            adapter = RaonWebSocketAdapter(url, **kwargs)
        else:
            raise ValueError(f"Unknown backend {backend!r} for agent {agent_id}")
        label = item.get("label", "Raon" if backend == "raon" else None)
        if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 80):
            raise ValueError("Agent label must be a nonempty string of at most 80 characters")
        adapter.display_name = label
        adapters[agent_id] = adapter
    if not adapters:
        raise ValueError("Configure at least one worker")
    return adapters
