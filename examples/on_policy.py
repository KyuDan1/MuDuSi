"""CPU REINFORCE example: learn a stochastic listen/speak gate from received PCM.

This trains three tiny gate policies, NOT speech-language model weights. The
emitted action is a synthetic carrier when the sampled gate is on. Each agent
observes only its own current microphone. Parameters stay frozen for a full
rollout; logged behavior probabilities and version identify the policy that
actually generated each action. Discounted per-agent returns assign temporal
credit. Acoustic heuristic rewards are deliberately not dialogue objectives.

Run: python examples/on_policy.py --episodes 12 --output runs/gates
"""
import argparse
import json
from pathlib import Path

import numpy as np
from duplex_arena.env import DuplexArenaEnv


def features(observation):
    level = float(np.sqrt(np.mean(observation["audio"] ** 2)))
    t = observation["time"]
    return np.array([1.0, min(level * 20, 2), np.sin(t), np.cos(t)])


def train(episodes=12, seed=42, horizon=100, output="runs/gates"):
    rng = np.random.default_rng(seed)
    env = DuplexArenaEnv(seed=seed, horizon=horizon, reverb=0.1)
    weights = {key: rng.normal(0, 0.15, size=4) for key in env.possible_agents}
    baseline = {key: 0.0 for key in weights}
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    history = []
    with (path / "transitions.jsonl").open("w") as stream:
        for version in range(episodes):
            observations, _ = env.reset(seed=seed + version)
            rollout = {key: [] for key in weights}
            for tick in range(horizon):
                pending, actions = {}, {}
                for index, key in enumerate(weights):
                    x = features(observations[key])
                    p = float(1 / (1 + np.exp(-np.clip(weights[key] @ x, -20, 20))))
                    gate = int(rng.random() < p)
                    t = (tick * env.frame_size + np.arange(env.frame_size)) / env.sample_rate
                    actions[key] = (gate * 0.17 * np.sin(2 * np.pi * (140 + 43 * index) * t)).astype(np.float32)
                    pending[key] = (x, p, gate)
                observations, rewards, terminated, truncated, infos = env.step(actions)
                for key, (x, p, gate) in pending.items():
                    rollout[key].append((x, p, gate, rewards[key]))
                    stream.write(json.dumps({"episode": version, "tick": tick, "agent": key,
                        "policy_version": version, "features": x.tolist(), "gate": gate,
                        "behavior_log_prob": float(np.log(p if gate else 1-p)), "reward": rewards[key],
                        "reward_terms": infos[key]["reward_terms"], "truncated": truncated[key]}) + "\n")
            totals = {}
            for key, rows in rollout.items():
                returns = np.zeros(len(rows))
                credit = 0.0
                for t in reversed(range(len(rows))):
                    credit = rows[t][3] + 0.97 * credit
                    returns[t] = credit
                # Baseline is from prior rollouts, independent of current actions.
                gradient = sum((g - baseline[key]) * (gate-p) * x
                               for (x, p, gate, _), g in zip(rows, returns)) / len(rows)
                weights[key] += 0.25 * np.clip(gradient, -2, 2)
                baseline[key] = 0.9 * baseline[key] + 0.1 * float(returns.mean())
                totals[key] = float(sum(row[3] for row in rows))
            result = {"episode": version, "policy_version": version, "returns": totals,
                      "overlap_ratio": env.render()["metrics"]["overlap_ratio"]}
            history.append(result)
            print(json.dumps(result))
    artifact = {"kind": "toy_gate_policy_not_slm", "algorithm": "on-policy REINFORCE", "seed": seed,
                "policy_version": episodes, "weights": {k: v.tolist() for k, v in weights.items()}, "history": history}
    (path / "policy.json").write_text(json.dumps(artifact, indent=2) + "\n")
    env.close()
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output", default="runs/gates")
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("episodes and horizon must be positive")
    train(**vars(args))
