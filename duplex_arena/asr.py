"""Optional, delayed ASR over one receiver's actual mixed microphone signal.

Only ``start()`` imports the optional backend or downloads its model. The arena
and its synthetic captions work without this module's optional dependencies.
An injected transcriber accepts one mono float32 24 kHz window and returns text;
it replaces the entire backend, including its production 16 kHz resampling.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
import math

import numpy as np


class ASRObserver:
    sample_rate = 24_000
    window_seconds = 5.0

    def __init__(self, model_name: str = "tiny", device: str = "cpu", agent_id: str = "moshi",
                 transcriber: Callable[[np.ndarray], str] | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.agent_id = agent_id
        self._transcriber = transcriber
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._events: deque[dict] = deque(maxlen=100)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0
        self._last_end: float | None = None
        self._generation = 0
        self._sequence = 0
        self._task: asyncio.Task | None = None
        self._closed = False
        self._start_lock = asyncio.Lock()
        self._stats = {"ready": False, "loading": False, "dropped_windows": 0, "processed_windows": 0,
                       "discontinuities": 0, "error": None}

    @property
    def stats(self) -> dict:
        return {**self._stats, "model": self.model_name, "device": self.device, "agent_id": self.agent_id,
                "window_seconds": self.window_seconds, "pending_windows": self._queue.qsize(),
                "buffered_seconds": len(self._buffer) / self.sample_rate}

    def _load_transcriber(self) -> Callable[[np.ndarray], str]:
        try:
            from faster_whisper import WhisperModel
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("Receiver ASR requires the optional dependencies: pip install '.[asr]'") from exc
        model = WhisperModel(self.model_name, device=self.device,
                             compute_type="int8" if self.device == "cpu" else "float16")

        def transcribe(audio_24k: np.ndarray) -> str:
            audio_16k = resample_poly(audio_24k, 2, 3).astype(np.float32)
            segments, _ = model.transcribe(audio_16k, beam_size=1, vad_filter=True,
                                           condition_on_previous_text=False)
            # faster-whisper inference occurs while iterating this generator;
            # keep that work inside the worker thread as well.
            return " ".join(segment.text.strip() for segment in segments).strip()

        return transcribe

    async def start(self) -> None:
        """Load the model off the event loop, then start one inference worker."""
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("ASR observer is closed")
            if self._task is not None and not self._task.done():
                return
            self._stats["loading"] = True
            try:
                if self._transcriber is None:
                    self._transcriber = await asyncio.to_thread(self._load_transcriber)
                if self._closed:
                    return
                self._stats.update(ready=True, error=None)
                self._task = asyncio.create_task(self._worker(), name=f"asr-{self.agent_id}")
            except Exception as exc:
                self._stats.update(ready=False, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                self._stats["loading"] = False

    def feed(self, pcm: np.ndarray, time_end: float) -> None:
        """Queue contiguous 5 s windows without blocking the simulation clock.

        Invoke on the same asyncio event loop as ``start``. The caller supplies
        the receiver's post-propagation PCM, not a clean speaker output. When
        inference falls behind, discard the oldest of two waiting windows.
        """
        block = np.asarray(pcm, dtype=np.float32)
        if block.ndim != 1 or not np.all(np.isfinite(block)):
            raise ValueError("ASR input must be a finite mono PCM array")
        if not isinstance(time_end, (int, float)) or not math.isfinite(time_end):
            raise ValueError("time_end must be a finite simulation timestamp")
        if not self._stats["ready"] or self._closed or not len(block):
            return
        start = float(time_end) - len(block) / self.sample_rate
        if self._last_end is not None and abs(start - self._last_end) > 1.1 / self.sample_rate:
            # Never join two different episodes, skipped time ranges, or
            # discontinuous microphone fragments into a fictitious utterance.
            self._buffer = np.empty(0, dtype=np.float32)
            self._stats["discontinuities"] += 1
        if not len(self._buffer):
            self._buffer_start = start
        self._last_end = float(time_end)
        self._buffer = np.concatenate((self._buffer, block))
        size = round(self.sample_rate * self.window_seconds)
        while len(self._buffer) >= size:
            window = self._buffer[:size].copy()
            self._buffer = self._buffer[size:].copy()
            window_start = self._buffer_start
            window_end = window_start + self.window_seconds
            self._buffer_start = window_end
            item = (self._generation, window_start, window_end, window)
            if self._queue.full():
                self._queue.get_nowait()
                self._queue.task_done()
                self._stats["dropped_windows"] += 1
            self._queue.put_nowait(item)

    async def _worker(self) -> None:
        while True:
            generation, start, end, audio = await self._queue.get()
            try:
                if generation != self._generation or self._closed:
                    continue
                text = await asyncio.to_thread(self._transcriber, audio)
                if generation != self._generation or self._closed:
                    continue
                if not isinstance(text, str):
                    raise TypeError("ASR transcriber must return a string")
                self._stats["processed_windows"] += 1
                self._stats["error"] = None
                if text.strip():
                    self._sequence += 1
                    self._events.append({"id": f"asr-{generation}-{self._sequence}", "time": round(end, 6),
                                         "start": round(start, 6), "end": round(end, 6), "agent_id": self.agent_id,
                                         "text": text.strip(), "kind": "asr", "type": "asr"})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if generation == self._generation and not self._closed:
                    self._stats["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def drain_events(self) -> list[dict]:
        """Return and remove newly completed actual-ASR events."""
        events = list(self._events)
        self._events.clear()
        return events

    def _clear_pending(self) -> None:
        self._generation += 1
        self._buffer = np.empty(0, dtype=np.float32)
        self._last_end = None
        self._buffer_start = 0.0
        self._events.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    async def reset(self) -> None:
        """Forget the episode and ignore any obsolete in-progress inference.

        The active inference thread cannot safely be forcibly stopped. Keeping
        the single worker alive lets it finish, reject its stale generation,
        and proceed with new audio without concurrent use of the model.
        """
        self._clear_pending()
        self._stats.update(dropped_windows=0, processed_windows=0, discontinuities=0, error=None)

    async def close(self) -> None:
        """Cancel the async worker; late thread results can never be published."""
        self._closed = True
        self._clear_pending()
        self._stats["ready"] = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
