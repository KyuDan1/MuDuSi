"""One authoritative simulation clock, many read-only observers.

The UI never invents audio levels. A browser disconnect cannot advance or reset the
environment. Live workers consume previous received blocks and generate output
independently; a 1-block control boundary prevents within-step future leakage.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
import io
import os
import time
import wave
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .engine import ArenaEngine

IDS = ("persona", "moshi", "custom")


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: Literal["persona", "moshi", "custom"]
    position: tuple[float, float, float] | None = None
    muted: bool | None = None


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["play", "pause", "step", "reset", "configure"]
    speed: Literal[0.5, 1.0, 2.0] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32-1)
    scenario: Literal["roundtable", "interruption", "cocktail", "quiet"] | None = None
    noise_db: float | None = Field(default=None, ge=-90, le=-10)
    reverb: float | None = Field(default=None, ge=0, le=0.8)
    agents: list[AgentUpdate] | None = None


class Session:
    def __init__(self, seed=42, scenario="roundtable", models: str | None = None, asr=None):
        self.engine = ArenaEngine(seed=seed, scenario=scenario)
        self.running = False
        self.speed = 1.0
        self.error: str | None = None
        self.models = models
        self.adapters = {}
        self.trace: deque = deque(maxlen=3000)
        self.recordings = {k: deque(maxlen=750) for k in IDS}
        self.observers: set[asyncio.Queue] = set()
        self.audio_observers = {k: set() for k in IDS}
        self.lock = asyncio.Lock()
        self.last_step_ms = 0.0
        self.overruns = 0
        self.model_events: deque = deque(maxlen=80)
        self.model_event_sequence = 0
        self.model_text = {k: "" for k in IDS}
        self.adc_clipped_samples = {k: 0 for k in IDS}
        self.asr = asr
        self.asr_events: deque = deque(maxlen=40)
        self.asr_text = {k: "" for k in IDS}

    async def connect_models(self):
        await self.close_models()
        if not self.models:
            return
        from .adapters import build_adapters
        self.adapters = build_adapters(self.models)
        try:
            # Concurrent startup avoids serializing model warm-up across GPUs.
            results = await asyncio.gather(*(a.start() for a in self.adapters.values()), return_exceptions=True)
            failures = [r for r in results if isinstance(r, BaseException)]
            if failures:
                raise RuntimeError(str(failures[0]))
        except Exception:
            await self.close_models()
            raise

    async def close_models(self):
        await asyncio.gather(*(a.close() for a in self.adapters.values()), return_exceptions=True)
        self.adapters = {}

    def snapshot(self):
        state = deepcopy(self.engine.snapshot())
        state.update(running=self.running, speed=self.speed, error=self.error,
                     mode="live" if self.models else "demo",
                     performance={"step_ms": round(self.last_step_ms, 2), "overruns": self.overruns,
                                  "adc_clipped_samples": self.adc_clipped_samples.copy()},
                     recording={"max_seconds": 60, "trace_max_frames": 3000})
        if self.models:
            state["events"] = list(self.model_events)
            for agent in state["agents"]:
                adapter = self.adapters.get(agent["id"])
                if adapter and getattr(adapter, "display_name", None):
                    agent["name"] = adapter.display_name
                agent["transcript"] = self.model_text[agent["id"]]
                agent["transcript_kind"] = "model"
                agent["model"] = (getattr(adapter, "stats", {}).get("worker_name") or agent["name"]) if adapter else "Unconnected · silent"
                agent["transport"] = type(adapter).__name__ if adapter else None
                agent["worker"] = getattr(adapter, "stats", {}) if adapter else {"connected": False}
                if agent["worker"].get("worker_name") == "wav-stimulus" and not getattr(adapter, "display_name", None):
                    agent["name"] = "WAV stimulus"
        state["asr"] = {"enabled": self.asr is not None, **(self.asr.stats if self.asr else {})}
        state["asr_events"] = list(self.asr_events)
        for agent in state["agents"]:
            agent["asr_transcript"] = self.asr_text[agent["id"]]
        return state

    def collect_asr(self):
        if self.asr:
            for event in self.asr.drain_events():
                self.asr_events.append(event)
                self.asr_text[event["agent_id"]] = event["text"]

    @staticmethod
    def offer(queue, value):
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(value)

    def publish(self):
        state = self.snapshot()
        for queue in self.observers:
            self.offer(queue, state)
        return state

    async def step(self):
        started = time.perf_counter()
        if self.models:
            if not self.adapters:
                raise RuntimeError("No model workers connected. Check configuration and reset.")
            for key, adapter in self.adapters.items():
                if getattr(adapter, "error", None):
                    raise RuntimeError(f"{key}: {adapter.error}")
            inputs = {}
            for key in self.adapters:
                raw = self.engine.last_inputs[key]
                self.adc_clipped_samples[key] += int(np.count_nonzero(np.abs(raw) > 1))
                inputs[key] = np.clip(raw, -1, 1).astype(np.float32)
            await asyncio.gather(*(adapter.push_audio(inputs[key]) for key, adapter in self.adapters.items()))
            actions = {key: adapter.pull_audio(self.engine.frame_size) for key, adapter in self.adapters.items()}
            self.engine.step(actions)
            for key, adapter in self.adapters.items():
                for token in adapter.drain_text():
                    self.model_text[key] = (self.model_text[key] + token)[-600:]
                    self.model_event_sequence += 1
                    self.model_events.append({"id": f"model-{self.model_event_sequence}", "time": self.engine.time,
                                              "agent_id": key, "type": "text", "text": token, "kind": "model"})
        else:
            self.engine.step()
        if self.asr:
            self.asr.feed(self.engine.last_inputs[self.asr.agent_id], self.engine.time)
            self.collect_asr()
        self.last_step_ms = (time.perf_counter() - started) * 1000
        for key in IDS:
            frame = self.engine.last_inputs[key].astype("<f4", copy=True)
            self.recordings[key].append(frame)
            for queue in self.audio_observers[key]:
                self.offer(queue, frame.tobytes())
        state = self.snapshot()
        self.trace.append(state)
        if state.get("done"):
            self.running = False
        return self.publish()

    async def reset(self, seed=None, scenario=None):
        self.running = False
        self.engine.reset(seed=seed, scenario=scenario)
        self.trace.clear()
        self.model_events.clear()
        self.model_event_sequence = 0
        self.model_text = {k: "" for k in IDS}
        self.asr_events.clear()
        self.asr_text = {k: "" for k in IDS}
        self.adc_clipped_samples = {k: 0 for k in IDS}
        if self.asr:
            await self.asr.reset()
        for recording in self.recordings.values():
            recording.clear()
        for queues in self.audio_observers.values():
            for queue in queues:
                while not queue.empty():
                    queue.get_nowait()
        self.error = None
        self.overruns = 0
        await self.connect_models()

    async def clock(self):
        while True:
            started = time.perf_counter()
            async with self.lock:
                self.collect_asr()
                if self.running:
                    try:
                        await self.step()
                    except Exception as exc:
                        self.error = str(exc)
                        self.running = False
                        self.publish()
                elif self.asr:
                    self.publish()
            interval = self.engine.frame_size / self.engine.sample_rate / self.speed
            elapsed = time.perf_counter() - started
            if self.running and elapsed > interval:
                self.overruns += 1
            await asyncio.sleep(max(0.001, interval - elapsed))


def create_app(seed=42, scenario="roundtable", models=None, asr=None, replay=None):
    if replay:
        if models or asr:
            raise ValueError("Replay cannot connect models or ASR")
        from .replay import ReplaySession
        session = ReplaySession(replay)
    else:
        session = Session(seed, scenario, models, asr)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await session.connect_models()
        except Exception as exc:
            session.error = str(exc)
        if session.asr:
            try:
                await session.asr.start()
            except Exception:
                # Optional observer failure is exposed through ASR stats; the
                # acoustic experiment remains available for inspection.
                pass
        task = asyncio.create_task(session.clock())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await session.close_models()
        if session.asr:
            await session.asr.close()

    app = FastAPI(title="MuDuSi — Multi-agent Duplex Simulator", version="0.1.0", lifespan=lifespan)
    app.state.session = session

    presentation = None

    def get_presentation():
        nonlocal presentation
        if not replay:
            raise HTTPException(409, "Presentation requires --replay of a completed recording")
        if presentation is None:
            from .presentation import load_presentation
            presentation = load_presentation(replay)
        return presentation

    @app.get("/api/presentation")
    async def presentation_data():
        return get_presentation()[0]

    @app.get("/api/presentation/audio.wav")
    async def presentation_audio():
        return Response(get_presentation()[1], media_type="audio/wav")

    @app.get("/api/health")
    async def health():
        return {"status": "error" if session.error else "ok", "mode": "replay" if replay else "live" if models else "demo", "error": session.error}

    @app.get("/api/state")
    async def state():
        return session.snapshot()

    @app.post("/api/control")
    async def control(command: Control):
        async with session.lock:
            try:
                if replay and (command.action == "configure" or any(value is not None for value in
                    (command.seed, command.scenario, command.noise_db, command.reverb, command.agents))):
                    raise ValueError("Recorded replay is read-only; run another batch to change acoustics")
                if command.speed is not None:
                    if models and command.speed != 1:
                        raise ValueError("Live model streaming runs at 1× wall-clock speed.")
                    session.speed = command.speed
                if command.action == "reset" or command.scenario is not None:
                    await session.reset(seed=command.seed, scenario=command.scenario)
                options = command.model_dump(exclude_none=True, exclude={"action", "speed", "seed", "scenario"})
                if options:
                    session.engine.configure(**options)
                if command.action == "play":
                    if session.error:
                        raise ValueError("Reset to reconnect workers before playing: " + session.error)
                    session.running = True
                elif command.action == "pause":
                    session.running = False
                elif command.action == "step":
                    session.running = False
                    await session.step()
                return session.publish()
            except (ValueError, KeyError) as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                session.error = str(exc)
                session.running = False
                session.publish()
                raise HTTPException(503, str(exc)) from exc

    @app.get("/api/export")
    async def export():
        session.collect_asr()
        return {"schema_version": 1, "meta": {"mode": "replay" if replay else "live" if models else "demo", "seed": session.engine.seed,
                "sample_rate": session.engine.sample_rate, "frame_size": session.engine.frame_size,
                "trace_limit_frames": 3000, "acoustics": "direct delay, inverse-distance gain, partition loss, simple echo",
                "transcripts": "generated model text, not ASR" if models or replay else "scripted demo captions, not ASR",
                "reward": "heuristic acoustic proxy, not semantic quality"}, "frames": list(session.trace),
                "asr_events": list(session.asr_events), "asr": session.snapshot()["asr"]}

    @app.get("/api/audio/{agent_id}.wav")
    async def audio(agent_id: str):
        if agent_id not in IDS:
            raise HTTPException(404, "Unknown receiver")
        frames = list(session.recordings[agent_id])
        if not frames:
            raise HTTPException(409, "Run the simulation before exporting receiver audio.")
        pcm = (np.clip(np.concatenate(frames), -1, 1) * 32767).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(session.engine.sample_rate)
            wav.writeframes(pcm.tobytes())
        return Response(output.getvalue(), media_type="audio/wav",
                        headers={"Content-Disposition": f'attachment; filename="{agent_id}-received.wav"'})

    @app.websocket("/ws")
    async def states(ws: WebSocket):
        await ws.accept()
        queue = asyncio.Queue(maxsize=2)
        session.observers.add(queue)
        try:
            await ws.send_json(session.snapshot())
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=5)
                except asyncio.TimeoutError:
                    frame = session.snapshot()
                await ws.send_json(frame)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session.observers.discard(queue)

    @app.websocket("/ws/audio/{agent_id}")
    async def stream_audio(ws: WebSocket, agent_id: str):
        if agent_id not in IDS:
            await ws.close(code=1008)
            return
        await ws.accept()
        queue = asyncio.Queue(maxsize=4)
        session.audio_observers[agent_id].add(queue)
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=5)
                except asyncio.TimeoutError:
                    frame = b""
                await ws.send_bytes(frame)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session.audio_observers[agent_id].discard(queue)

    dist = Path(os.environ.get(
        "MUDUSI_WEB_DIR", os.environ.get("DUPLEX_ARENA_WEB_DIR", Path.cwd() / "web" / "dist")
    ))
    if not dist.exists():
        dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        async def index():
            return FileResponse(dist / "index.html")
    else:
        @app.get("/")
        async def unbuilt():
            return {"message": "Build the viewer: cd web && npm ci && npm run build", "api": "/docs"}
    return app
