"""Small parallel multi-agent interface for research integration.

``reset`` and ``step`` return PettingZoo-style dictionaries without requiring
PettingZoo or Gymnasium. This is not a registered PettingZoo environment.
Each action is exactly one 80 ms, 24 kHz mono float PCM emission. All emissions
are submitted simultaneously; returned audio is the next observation. Policies
must consume the observation before producing their next action. There is no
text, clean source signal, or future audio in an agent's observation.

Rewards are transparent engineering heuristics: audibility, a tunable-in-code
overlap cost, and emission energy. They do not evaluate dialogue quality or
semantic credit assignment; legitimate backchannels can receive overlap costs.
This module supplies an environment, not a trained policy or an RL algorithm.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .engine import AGENT_IDS, ArenaEngine


class DuplexArenaEnv:
    metadata = {"name": "duplex_arena_v0", "is_parallelizable": True, "render_modes": ["snapshot"]}
    possible_agents = list(AGENT_IDS)

    def __init__(self, seed: int = 42, scenario: str = "roundtable", horizon: int = 750,
                 noise_db: float | None = None, reverb: float | None = None) -> None:
        self.engine = ArenaEngine(seed=seed, scenario=scenario, horizon=horizon)
        self._noise_db = noise_db
        self._reverb = reverb
        self.agents = list(self.possible_agents)
        self._configure()

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @property
    def frame_size(self) -> int:
        return self.engine.frame_size

    def _configure(self) -> None:
        self.engine.configure(noise_db=self._noise_db, reverb=self._reverb)

    def _observations(self) -> dict[str, dict]:
        return {agent: {"audio": self.engine.last_inputs[agent].copy(),
                        "position": np.asarray(self.engine.agents[agent]["position"], dtype=np.float32),
                        "time": self.engine.time} for agent in AGENT_IDS}

    def _infos(self) -> dict[str, dict]:
        snapshot = self.engine.snapshot()
        return {agent: {"reward_terms": snapshot["rewards"][agent], "seed": self.engine.seed,
                        "tick": self.engine.tick, "reward_kind": "acoustic_heuristic"} for agent in AGENT_IDS}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        options = options or {}
        self.engine.reset(seed=seed, scenario=options.get("scenario"))
        self.engine.mode = "external"
        self._configure()
        self.agents = list(self.possible_agents)
        return self._observations(), self._infos()

    def step(self, actions: Mapping[str, np.ndarray]) -> tuple[dict, dict, dict, dict, dict]:
        if not self.agents:
            raise RuntimeError("Episode is done; call reset() before stepping again")
        if actions is None:
            raise ValueError("RL actions must be a mapping; use {} for silence")
        snapshot = self.engine.step(actions)
        observations = self._observations()
        rewards = {agent: snapshot["rewards"][agent]["total"] for agent in AGENT_IDS}
        # A fixed time budget is a truncation, not a task terminal state.
        terminations = {agent: False for agent in AGENT_IDS}
        truncations = {agent: snapshot["done"] for agent in AGENT_IDS}
        infos = self._infos()
        if snapshot["done"]:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> dict:
        return self.engine.snapshot()

    def close(self) -> None:
        self.agents = []


# Short import name for training scripts.
ArenaEnv = DuplexArenaEnv
