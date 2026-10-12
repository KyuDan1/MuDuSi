"""Deterministic, streaming mono acoustics for a small virtual room.

This is a deliberately inexpensive propagation model, not a room-acoustics
solver: distance delay, inverse-distance pressure, divider loss, one image-
source reflection and independent microphone noise. All levels are dBFS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920
DB_FLOOR = -96.0


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def dbfs(audio: np.ndarray) -> float:
    return max(DB_FLOOR, 20.0 * float(np.log10(max(rms(audio), 1e-12))))


def validate_audio(audio: object, frame_size: int, agent_id: str) -> np.ndarray:
    """Copy an emission so later caller mutations cannot alter the simulation."""
    try:
        block = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{agent_id}: audio must be a numeric mono PCM block") from exc
    if block.shape != (frame_size,):
        raise ValueError(f"{agent_id}: expected mono PCM shape ({frame_size},), got {block.shape}")
    if not np.all(np.isfinite(block)):
        raise ValueError(f"{agent_id}: audio must contain only finite samples")
    return block.copy()


@dataclass(frozen=True)
class Divider:
    """A thin vertical acoustic divider parallel to the room's z axis."""

    x: float = 0.0
    z_min: float = -3.8
    z_max: float = -0.8
    height: float = 2.2
    loss_db: float = 14.0

    def blocks(self, source: np.ndarray, target: np.ndarray) -> bool:
        dx = float(target[0] - source[0])
        if abs(dx) < 1e-12:
            return False
        fraction = (self.x - float(source[0])) / dx
        if not 0 < fraction < 1:
            return False
        crossing = source + fraction * (target - source)
        return bool(self.z_min <= crossing[2] <= self.z_max and 0 <= crossing[1] <= self.height)

    def as_dict(self) -> dict:
        return {
            "type": "divider", "position": [self.x, self.height / 2, (self.z_min + self.z_max) / 2],
            "size": [0.15, self.height, self.z_max - self.z_min], "loss_db": self.loss_db,
        }


class AcousticMixer:
    """Mix simultaneous emissions into independent, causally delayed inputs.

    A queue per directed link preserves delayed samples across frame boundaries.
    Geometry is evaluated at emission time; a moved listener does not rewrite
    already-propagating samples. There is deliberately no local sidetone or
    reflected self audio. The reflection is one right-wall image source.
    """

    def __init__(
        self, agent_ids: Sequence[str], *, seed: int = 42, sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE, noise_db: float = -55.0, reverb: float = 0.18,
        room_width: float = 12.0, dividers: Sequence[Divider] = (),
    ) -> None:
        if sample_rate <= 0 or frame_size <= 0:
            raise ValueError("sample_rate and frame_size must be positive")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent ids must be unique")
        self.agent_ids = tuple(agent_ids)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_db = float(noise_db)
        self.reverb = float(reverb)
        self.room_width = room_width
        self.dividers = tuple(dividers)
        self.rng = np.random.default_rng(seed)
        self._queues: dict[tuple[str, str], np.ndarray] = {}
        self.last_links: list[dict] = []
        self.last_components: dict[tuple[str, str], np.ndarray] = {}

    def _schedule(self, key: tuple[str, str], block: np.ndarray, delay: int, gain: float) -> None:
        needed = delay + self.frame_size
        queue = self._queues.get(key)
        if queue is None:
            queue = np.zeros(needed, dtype=np.float64)
        elif len(queue) < needed:
            queue = np.pad(queue, (0, needed - len(queue)))
        queue[delay:needed] += block * gain
        self._queues[key] = queue

    def render(self, outputs: Mapping[str, np.ndarray], positions: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
        unknown = set(outputs) - set(self.agent_ids)
        if unknown:
            raise ValueError(f"Unknown agent ids: {sorted(unknown)}")
        blocks = {agent: validate_audio(outputs.get(agent, np.zeros(self.frame_size)), self.frame_size, agent)
                  for agent in self.agent_ids}
        coords = {agent: np.asarray(positions[agent], dtype=np.float64) for agent in self.agent_ids}
        if any(p.shape != (3,) or not np.all(np.isfinite(p)) for p in coords.values()):
            raise ValueError("Positions must be finite [x, y, z] coordinates")
        inputs = {agent: np.zeros(self.frame_size, dtype=np.float64) for agent in self.agent_ids}
        links: list[dict] = []
        components: dict[tuple[str, str], np.ndarray] = {}
        for source in self.agent_ids:
            for target in self.agent_ids:
                if source == target:
                    continue
                key = (source, target)
                distance = float(np.linalg.norm(coords[source] - coords[target]))
                delay = int(np.ceil(distance / SPEED_OF_SOUND * self.sample_rate))
                occlusion_db = sum(d.loss_db for d in self.dividers if d.blocks(coords[source], coords[target]))
                gain = 10 ** (-occlusion_db / 20) / max(1.0, distance)
                self._schedule(key, blocks[source], delay, gain)
                if self.reverb > 0:
                    image_source = coords[source].copy()
                    image_source[0] = self.room_width - image_source[0]
                    echo_distance = float(np.linalg.norm(image_source - coords[target]))
                    echo_delay = max(delay + 1, int(np.ceil(echo_distance / SPEED_OF_SOUND * self.sample_rate)))
                    echo_gain = self.reverb * 0.45 / max(1.0, echo_distance)
                    self._schedule(key, blocks[source], echo_delay, echo_gain)
                queue = self._queues[key]
                component = queue[:self.frame_size].copy()
                self._queues[key] = queue[self.frame_size:].copy()
                inputs[target] += component
                components[key] = component.astype(np.float32)
                links.append({
                    "source": source, "target": target, "distance": round(distance, 3),
                    "delay_ms": round(delay / self.sample_rate * 1000, 3),
                    "gain_db": round(20 * float(np.log10(max(gain, 1e-12))), 2),
                    "occluded": bool(occlusion_db), "level_db": round(dbfs(component), 2),
                })
        # A disabled noise source (-inf) is useful for exact impulse tests.
        if np.isfinite(self.noise_db):
            noise_scale = 10 ** (self.noise_db / 20)
            for target in self.agent_ids:
                inputs[target] += self.rng.normal(0.0, noise_scale, self.frame_size)
        self.last_links = links
        self.last_components = components
        # Do not clip: retain physical superposition and let adapters handle ADC limits.
        return {agent: block.astype(np.float32) for agent, block in inputs.items()}
