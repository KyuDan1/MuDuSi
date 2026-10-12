import json

import numpy as np
import pytest

from duplex_arena.engine import AGENT_IDS, ArenaEngine
from duplex_arena.env import DuplexArenaEnv


def test_demo_is_reproducible_and_snapshot_is_json_serializable():
    one, two = ArenaEngine(seed=17), ArenaEngine(seed=17)
    for _ in range(60):
        assert one.step() == two.step()
        for agent in AGENT_IDS:
            np.testing.assert_array_equal(one.last_inputs[agent], two.last_inputs[agent])
    snapshot = one.snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["events"]
    assert all(event["kind"] == "scripted" for event in snapshot["events"])
    assert snapshot["sample_rate"] == 24_000
    assert snapshot["frame_ms"] == 80


def test_external_actions_remain_independent_and_missing_means_silence():
    arena = ArenaEngine()
    arena.configure(noise_db=-120, reverb=0)
    emission = np.full(arena.frame_size, 0.15, dtype=np.float32)
    snapshot = arena.step({"persona": emission})
    assert snapshot["mode"] == "external"
    assert not np.any(arena.last_outputs["moshi"])
    assert not np.any(arena.last_outputs["custom"])
    assert np.max(np.abs(arena.last_inputs["persona"])) < 0.00001
    assert np.max(arena.last_inputs["moshi"]) > 0.01
    assert snapshot["agents"][0]["transcript"] == ""
    emission[:] = 0
    assert np.all(arena.last_outputs["persona"] == 0.15)


def test_rl_observations_have_only_local_audio_and_no_future_leakage():
    env = DuplexArenaEnv(horizon=3, noise_db=-120, reverb=0)
    initial, infos = env.reset(seed=4)
    assert set(initial) == set(AGENT_IDS)
    assert set(initial["persona"]) == {"audio", "position", "time"}
    assert np.count_nonzero(initial["moshi"]["audio"]) == 0
    assert infos["moshi"]["reward_kind"] == "acoustic_heuristic"
    emission = np.zeros(env.frame_size, dtype=np.float32)
    emission[-1] = 1
    observed, _, _, _, _ = env.step({"persona": emission})
    # This last-sample impulse cannot reach a remote microphone this tick.
    assert np.max(np.abs(observed["moshi"]["audio"])) < 0.00001
    later, _, _, _, _ = env.step({})
    assert np.max(later["moshi"]["audio"]) > 0.1
    # Previously returned observations are snapshots, not shared receive buffers.
    assert np.count_nonzero(initial["moshi"]["audio"]) == 0


def test_horizon_truncates_all_agents_and_reset_restores_episode():
    env = DuplexArenaEnv(horizon=2)
    env.reset(seed=42)
    _, _, terminated, truncated, _ = env.step({})
    assert not any(terminated.values()) and not any(truncated.values())
    _, _, terminated, truncated, infos = env.step({})
    assert not any(terminated.values()) and all(truncated.values())
    assert env.agents == []
    assert infos["persona"]["tick"] == 2
    with pytest.raises(RuntimeError, match="reset"):
        env.step({})
    observed, _ = env.reset(seed=42)
    assert env.agents == list(AGENT_IDS)
    assert observed["persona"]["time"] == 0


def test_reward_terms_sum_and_overlap_penalty_is_explicit():
    single = DuplexArenaEnv(horizon=10, noise_db=-120, reverb=0)
    overlapping = DuplexArenaEnv(horizon=10, noise_db=-120, reverb=0)
    block = np.full(single.frame_size, 0.1, dtype=np.float32)
    single.reset()
    overlapping.reset()
    _, solo_rewards, _, _, solo_infos = single.step({"persona": block})
    _, overlap_rewards, _, _, overlap_infos = overlapping.step({"persona": block, "moshi": block})
    assert overlap_rewards["persona"] == pytest.approx(solo_rewards["persona"] - 0.04)
    for agent in AGENT_IDS:
        terms = overlap_infos[agent]["reward_terms"]
        assert terms["total"] == pytest.approx(terms["audibility"] + terms["overlap"] + terms["energy"])
    assert solo_infos["persona"]["reward_terms"]["overlap"] == 0


def test_validation_is_transactional():
    arena = ArenaEngine()
    original = arena.snapshot()
    with pytest.raises(ValueError):
        arena.step({"persona": np.zeros(arena.frame_size), "moshi": np.full(arena.frame_size, np.nan)})
    assert arena.snapshot() == original
    with pytest.raises(ValueError):
        arena.configure(noise_db=-20, agents=[{"id": "moshi", "position": [100, 0, 0]}])
    assert arena.snapshot() == original
    with pytest.raises(ValueError):
        arena.step({"unrecognized": np.zeros(arena.frame_size)})


def test_noise_reduces_audibility_credit_without_rewarding_silence():
    quiet = DuplexArenaEnv(noise_db=-72, reverb=0)
    noisy = DuplexArenaEnv(noise_db=-20, reverb=0)
    block = np.full(quiet.frame_size, 0.1, dtype=np.float32)
    quiet.reset()
    noisy.reset()
    _, _, _, _, quiet_infos = quiet.step({"persona": block})
    _, _, _, _, noisy_infos = noisy.step({"persona": block})
    assert quiet_infos["persona"]["reward_terms"]["audibility"] > noisy_infos["persona"]["reward_terms"]["audibility"]
    assert quiet_infos["moshi"]["reward_terms"]["total"] == 0
    assert noisy_infos["moshi"]["reward_terms"]["total"] == 0


def test_paused_movement_updates_geometry_without_noise_or_clock_progress():
    moved, reference = ArenaEngine(seed=5), ArenaEngine(seed=5)
    before = moved.snapshot()
    updated = moved.configure(agents=[{"id": "moshi", "position": [-2, 1.2, -1.2]}])
    assert updated["tick"] == before["tick"]
    link = next(link for link in updated["links"] if link["source"] == "persona" and link["target"] == "moshi")
    assert link["distance"] == 1.0
    moved.step({})
    reference.step({})
    for agent in AGENT_IDS:
        np.testing.assert_array_equal(moved.last_inputs[agent], reference.last_inputs[agent])


def test_scene_reset_clears_counters_and_muting_preserves_listener():
    arena = ArenaEngine()
    for _ in range(20):
        arena.step()
    reset = arena.configure(scenario="cocktail")
    assert reset["time"] == 0 and reset["events"] == []
    assert reset["room"]["obstacles"]
    arena.configure(noise_db=-120, reverb=0, agents=[{"id": "moshi", "muted": True}])
    block = np.full(arena.frame_size, 0.1, dtype=np.float32)
    arena.step({"persona": block, "moshi": block})
    assert np.count_nonzero(arena.last_outputs["moshi"]) == 0
    assert np.max(arena.last_inputs["moshi"]) > 0.001
