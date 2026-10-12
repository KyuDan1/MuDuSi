"""A reproducible three-agent full-duplex acoustic simulation.

The default demo emits synthetic harmonic tones with a speech-like envelope.
Its captions are a script, never ASR or model-generated text. Real adapters pass
their PCM emissions to ``step(actions)`` and consume ``last_inputs`` afterwards.
"""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

from .acoustics import AcousticMixer, Divider, FRAME_SIZE, SAMPLE_RATE, dbfs, rms, validate_audio

AGENT_IDS = ("persona", "moshi", "custom")
SCENARIOS = ("roundtable", "interruption", "cocktail", "quiet")
ROOM = {"width": 12.0, "depth": 9.0, "height": 3.2}
AGENT_DEFAULTS = (
    {"id": "persona", "name": "PersonaPlex", "model": "PersonaPlex · demo adapter", "color": "#B7D7C0", "position": [-3.0, 1.2, -1.2]},
    {"id": "moshi", "name": "Moshi", "model": "Moshi · demo adapter", "color": "#C9C0E8", "position": [2.7, 1.2, -1.1]},
    {"id": "custom", "name": "Custom SLM", "model": "Custom SLM · demo adapter", "color": "#E8C6AA", "position": [0.0, 1.2, 2.3]},
)

# start, duration, speaker, display-only script; each scenario repeats after 18s.
SCRIPTS = {
    "roundtable": (
        (0.4, 2.8, "persona", "Let's test how voices travel through this room."),
        (3.5, 2.5, "moshi", "I can hear you from the other side."),
        (6.5, 2.6, "custom", "I'll take the next turn. What changes with distance?"),
        (9.7, 2.3, "persona", "Moving closer should improve the signal."),
        (12.4, 2.4, "moshi", "Let's leave a little space between turns."),
        (15.4, 1.8, "custom", "Ready for another round."),
    ),
    "interruption": (
        (0.4, 4.2, "persona", "I have a longer thought about conversational timing—"),
        (2.0, 2.4, "moshi", "One moment, can I add something?"),
        (4.9, 2.5, "custom", "That was an interruption. Both streams stayed active."),
        (8.0, 3.6, "moshi", "A listener can answer before a speaker has finished—"),
        (9.8, 2.0, "persona", "Yes, a brief acknowledgment can be useful."),
        (13.0, 3.0, "custom", "Overlap is a measurement, not always a mistake."),
    ),
    "cocktail": (
        (0.3, 4.0, "persona", "There are several voices in this noisy room."),
        (1.1, 4.0, "moshi", "The divider changes how clearly we hear each other."),
        (2.1, 3.5, "custom", "I can still listen while I produce audio."),
        (7.0, 4.8, "persona", "Try moving the agents or changing the noise floor."),
        (8.3, 4.3, "custom", "Every microphone receives its own mixture."),
        (9.5, 4.0, "moshi", "Distance, delay and overlap all matter here."),
        (15.0, 2.3, "custom", "Now there is a short quiet gap."),
    ),
    "quiet": (
        (1.0, 2.8, "persona", "A quiet room makes a useful baseline."),
        (5.0, 2.4, "moshi", "The received signal is easier to separate."),
        (9.0, 2.8, "custom", "We can compare this with a noisy episode."),
        (14.0, 2.0, "persona", "Let's listen to the room."),
    ),
}


class ArenaEngine:
    sample_rate = SAMPLE_RATE
    frame_size = FRAME_SIZE

    def __init__(self, seed: int = 42, scenario: str = "roundtable", horizon: int = 750) -> None:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer number of frames")
        self.horizon = horizon
        self.seed = int(seed)
        self.scenario = scenario
        self.reset()

    @property
    def time(self) -> float:
        return self.tick * self.frame_size / self.sample_rate

    def reset(self, seed: int | None = None, scenario: str | None = None) -> dict:
        next_seed = self.seed if seed is None else int(seed)
        next_scenario = self.scenario if scenario is None else scenario
        if next_scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {', '.join(SCENARIOS)}")
        self.seed, self.scenario = next_seed, next_scenario
        self.tick = 0
        self.mode = "demo"
        self.noise_db = {"roundtable": -55.0, "interruption": -55.0, "cocktail": -35.0, "quiet": -72.0}[self.scenario]
        self.reverb = 0.18 if self.scenario != "quiet" else 0.06
        self.agents = {agent["id"]: {**copy.deepcopy(agent), "muted": False} for agent in AGENT_DEFAULTS}
        self.dividers = [Divider()] if self.scenario == "cocktail" else []
        self.mixer = AcousticMixer(AGENT_IDS, seed=self.seed, noise_db=self.noise_db, reverb=self.reverb,
                                   room_width=ROOM["width"], dividers=self.dividers)
        self.last_inputs = {agent: np.zeros(self.frame_size, dtype=np.float32) for agent in AGENT_IDS}
        self.last_outputs = {agent: block.copy() for agent, block in self.last_inputs.items()}
        self._events: list[dict] = []
        self._utterances_seen: set[tuple[int, int]] = set()
        self._transcripts = {agent: "" for agent in AGENT_IDS}
        self._cumulative = {agent: 0.0 for agent in AGENT_IDS}
        self._rewards = {agent: {"audibility": 0.0, "overlap": 0.0, "energy": 0.0, "total": 0.0} for agent in AGENT_IDS}
        self._overlap_ticks = 0
        self._speech_ticks = 0
        self._active_speakers = 0
        self._links: list[dict] = []
        # Derive initial geometry without consuming noise or advancing the clock.
        preview = AcousticMixer(AGENT_IDS, noise_db=-np.inf, reverb=0, dividers=self.dividers)
        preview.render(self.last_outputs, self._positions())
        self._links = preview.last_links
        return self.snapshot()

    def _positions(self) -> dict:
        return {agent: settings["position"] for agent, settings in self.agents.items()}

    def configure(self, noise_db: float | None = None, reverb: float | None = None,
                  agents: list[dict] | None = None, scenario: str | None = None) -> dict:
        """Apply validated controls. Switching scenes starts a fresh episode."""
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if noise_db is not None and (not np.isfinite(noise_db) or not -120 <= noise_db <= 0):
            raise ValueError("noise_db must be between -120 and 0 dBFS")
        if reverb is not None and (not np.isfinite(reverb) or not 0 <= reverb <= 1):
            raise ValueError("reverb must be between 0 and 1")
        changes: dict[str, dict] = {}
        for change in agents or []:
            if not isinstance(change, dict) or change.get("id") not in AGENT_IDS:
                raise ValueError("Each agent update must include a known id")
            agent_id = change["id"]
            validated = {}
            if "position" in change:
                position = np.asarray(change["position"], dtype=np.float64)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    raise ValueError("position must be finite [x, y, z]")
                if not (-6 <= position[0] <= 6 and 0 <= position[1] <= 3.2 and -4.5 <= position[2] <= 4.5):
                    raise ValueError("position must be inside the 12 × 9 × 3.2 m room")
                validated["position"] = position.tolist()
            if "muted" in change:
                if not isinstance(change["muted"], bool):
                    raise ValueError("muted must be a boolean")
                validated["muted"] = change["muted"]
            changes[agent_id] = {**changes.get(agent_id, {}), **validated}
        if scenario is not None and scenario != self.scenario:
            self.reset(scenario=scenario)
        if noise_db is not None:
            self.noise_db = self.mixer.noise_db = float(noise_db)
        if reverb is not None:
            self.reverb = self.mixer.reverb = float(reverb)
        for agent_id, change in changes.items():
            self.agents[agent_id].update(change)
        if any("position" in change for change in changes.values()):
            # Refresh geometry for paused viewers without consuming microphone
            # noise or modifying sound already travelling through the live mixer.
            preview = AcousticMixer(AGENT_IDS, noise_db=-np.inf, reverb=0, dividers=self.dividers)
            preview.render(self.last_outputs, self._positions())
            previous_levels = {(link["source"], link["target"]): link["level_db"] for link in self._links}
            self._links = [dict(link, level_db=previous_levels.get((link["source"], link["target"]), -96.0))
                           for link in preview.last_links]
        return self.snapshot()

    def _demo_outputs(self) -> dict[str, np.ndarray]:
        blocks = {agent: np.zeros(self.frame_size, dtype=np.float32) for agent in AGENT_IDS}
        times = self.time + np.arange(self.frame_size, dtype=np.float64) / self.sample_rate
        # Include both cycles if the PCM block crosses a cycle boundary.
        for cycle in range(int(times[0] // 18), int(times[-1] // 18) + 1):
            for index, (start, duration, agent, text) in enumerate(SCRIPTS[self.scenario]):
                onset = cycle * 18 + start
                local = times - onset
                mask = (local >= 0) & (local < duration)
                if not np.any(mask) or self.agents[agent]["muted"]:
                    continue
                t = local[mask]
                frequency = {"persona": 137.0, "moshi": 181.0, "custom": 223.0}[agent]
                phase = 2 * np.pi * frequency * t + 0.55 * np.sin(2 * np.pi * 2.8 * t)
                voice = np.sin(phase) + 0.36 * np.sin(2 * phase) + 0.14 * np.sin(3 * phase)
                syllables = 0.3 + 0.7 * np.maximum(0, np.sin(2 * np.pi * 3.7 * t))
                envelope = np.minimum(1.0, t / 0.025) * np.minimum(1.0, (duration - t) / 0.045)
                blocks[agent][mask] += (0.22 * voice * syllables * envelope).astype(np.float32)
                key = (cycle, index)
                if key not in self._utterances_seen:
                    self._utterances_seen.add(key)
                    self._transcripts[agent] = text
                    self._events.append({"id": f"{cycle}-{index}", "time": round(onset, 3),
                                         "agent_id": agent, "type": "utterance", "text": text, "kind": "scripted"})
        self._events = self._events[-80:]
        return blocks

    def step(self, actions: Mapping[str, np.ndarray] | None = None) -> dict:
        if self.tick >= self.horizon:
            raise RuntimeError("Episode is done; call reset() before stepping again")
        if actions is None:
            blocks = self._demo_outputs()
            self.mode = "demo"
        else:
            if not isinstance(actions, Mapping):
                raise ValueError("actions must map agent ids to mono PCM blocks")
            unknown = set(actions) - set(AGENT_IDS)
            if unknown:
                raise ValueError(f"Unknown agent ids: {sorted(unknown)}")
            blocks = {agent: validate_audio(actions.get(agent, np.zeros(self.frame_size)), self.frame_size, agent)
                      for agent in AGENT_IDS}
            self.mode = "external"
            self._transcripts = {agent: "" for agent in AGENT_IDS}
        for agent in AGENT_IDS:
            if self.agents[agent]["muted"]:
                blocks[agent] = np.zeros(self.frame_size, dtype=np.float32)
        received = self.mixer.render(blocks, self._positions())
        self.last_outputs = blocks
        self.last_inputs = received
        self._links = self.mixer.last_links
        active = {agent: dbfs(blocks[agent]) > -65 for agent in AGENT_IDS}
        self._active_speakers = sum(active.values())
        self._speech_ticks += int(self._active_speakers > 0)
        self._overlap_ticks += int(self._active_speakers > 1)
        for agent in AGENT_IDS:
            delivered = [link["level_db"] for link in self._links if link["source"] == agent]
            # Use source-attributed received energy before microphone noise, then
            # compare it with the configured noise floor. This is a proxy for
            # detectability, not an intelligibility or speech-quality score.
            audible_floor = max(-65.0, self.noise_db + 6.0)
            audibility = 0.10 * float(np.mean(np.clip((np.asarray(delivered) - audible_floor) / 30.0, 0, 1)))
            overlap = -0.04 if active[agent] and self._active_speakers > 1 else 0.0
            energy = -0.015 * (rms(blocks[agent]) / 0.2) ** 2
            total = audibility + overlap + energy
            self._rewards[agent] = {"audibility": audibility, "overlap": overlap, "energy": energy, "total": total}
            self._cumulative[agent] += total
        self.tick += 1
        return self.snapshot()

    def snapshot(self) -> dict:
        rendered_agents = []
        for agent, settings in self.agents.items():
            tx_db, rx_db = dbfs(self.last_outputs[agent]), dbfs(self.last_inputs[agent])
            speaking = tx_db > -65
            listening = rx_db > max(-75, self.noise_db + 6)
            activity = "duplex" if speaking and listening else "speaking" if speaking else "listening" if listening else "idle"
            rendered_agents.append({
                **copy.deepcopy(settings), "speaking": speaking, "listening": listening,
                "tx_db": round(tx_db, 2), "rx_db": round(rx_db, 2), "activity": activity,
                "transcript": self._transcripts[agent], "transcript_kind": "scripted" if self.mode == "demo" else "none",
                "reward": round(self._rewards[agent]["total"], 6), "cumulative_reward": round(self._cumulative[agent], 6),
            })
        return {
            "time": round(self.time, 3), "tick": self.tick, "seed": self.seed, "scenario": self.scenario,
            "mode": self.mode, "sample_rate": self.sample_rate, "frame_ms": self.frame_size / self.sample_rate * 1000,
            "room": {**ROOM, "obstacles": [d.as_dict() for d in self.dividers]},
            "noise_db": self.noise_db, "reverb": self.reverb, "agents": rendered_agents,
            "links": copy.deepcopy(self._links), "events": copy.deepcopy(self._events),
            "metrics": {"overlap_ratio": round(self._overlap_ticks / max(1, self._speech_ticks), 4),
                        "mean_rx_db": round(float(np.mean([a["rx_db"] for a in rendered_agents])), 2),
                        "episode_reward": round(sum(self._cumulative.values()), 6), "active_speakers": self._active_speakers},
            "rewards": copy.deepcopy(self._rewards), "done": self.tick >= self.horizon,
        }
