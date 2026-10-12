"""Read-only playback of completed batch artifacts; never contacts a worker."""
from copy import deepcopy
import json
from pathlib import Path
import wave

import numpy as np

from .server import IDS, Session


class ReplaySession(Session):
    def __init__(self, directory):
        super().__init__()
        root = Path(directory)
        self.frames = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
        if not self.frames:
            raise ValueError("Replay needs at least one recorded frame")
        self.engine.reset(seed=self.frames[0]["seed"], scenario=self.frames[0]["scenario"])
        self.audio = {}
        for key in IDS:
            with wave.open(str(root / f"{key}-received.wav")) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2):
                    raise ValueError("Replay audio must be 24 kHz mono PCM16")
                self.audio[key] = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32767
                if self.audio[key].size != len(self.frames) * 1920:
                    raise ValueError("Replay audio and trace lengths differ")
        self.cursor = 0
        self.offline_asr = json.loads((root / "asr.json").read_text()) if (root / "asr.json").exists() else None

    def snapshot(self):
        state = deepcopy(self.frames[max(0, self.cursor - 1)])
        for agent in state["agents"]:
            if agent.get("worker", {}).get("worker_name") == "wav-stimulus":
                agent["name"] = "WAV stimulus"
        state.update(running=self.running, speed=self.speed,
                     replay={"enabled": True, "frame": self.cursor, "frames": len(self.frames)},
                     error=self.error)
        if self.offline_asr:
            state["asr"] = {"enabled": True, "ready": True, "kind": "offline_receiver_asr",
                            "model": self.offline_asr["model"], "error": None}
            events = []
            for agent in state["agents"]:
                record = self.offline_asr["files"].get(f"{agent['id']}-received.wav", {})
                segments = [s for s in record.get("segments", []) if s["end"] <= state["time"]]
                agent["asr_transcript"] = " ".join(s["text"].strip() for s in segments[-2:])
                for s in segments:
                    events.append({"id": f"offline-{agent['id']}-{s['id']}", "time": s["end"],
                                   "start": s["start"], "end": s["end"], "agent_id": agent["id"],
                                   "text": s["text"], "kind": "asr"})
            state["asr_events"] = sorted(events, key=lambda e: e["time"])[-40:]
        return state

    async def step(self):
        if self.cursor >= len(self.frames):
            self.running = False
            return self.publish()
        start = self.cursor * 1920
        for key in IDS:
            pcm = self.audio[key][start:start + 1920]
            self.recordings[key].append(pcm)
            for queue in self.audio_observers[key]:
                self.offer(queue, pcm.astype("<f4").tobytes())
        self.cursor += 1
        self.trace.append(self.snapshot())
        if self.cursor == len(self.frames):
            self.running = False
        return self.publish()

    async def reset(self, seed=None, scenario=None):
        self.running = False
        self.cursor = 0
        self.trace.clear()
        for recording in self.recordings.values():
            recording.clear()
        for queues in self.audio_observers.values():
            for queue in queues:
                while not queue.empty():
                    queue.get_nowait()
