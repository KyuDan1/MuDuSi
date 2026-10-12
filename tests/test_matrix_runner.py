"""Plan and orchestration checks; no speech models, GPUs, or server processes."""
import asyncio
from collections import Counter
from contextlib import nullcontext
import itertools
import json
from pathlib import Path

import pytest

from duplex_arena import evaluation
from workers import run_matrix


@pytest.fixture
def configuration():
    return json.loads(Path("configs/matrix.example.json").read_text())


def test_ordered_pairs_have_independent_endpoints_and_balanced_devices(configuration, tmp_path):
    plan = run_matrix.make_plan(configuration, tmp_path)
    assert [(r["left"], r["right"]) for r in plan["rooms"]] == list(
        itertools.product(configuration["families"], repeat=2))
    workers = plan["workers"]
    assert len({w["id"] for w in workers}) == len({w["port"] for w in workers}) == 18
    for gpu in range(6):
        assert Counter(w["family"] for w in workers if w["gpu"] == gpu) == {
            "raon": 1, "moshi": 1, "personaplex": 1}
    for room in plan["rooms"]:
        assert [a["id"] for a in room["agents"]] == ["persona", "moshi"]
    assert configuration["room"]["agents"][2]["muted"] is True


@pytest.mark.parametrize("change", [{"families": ["raon"] * 3}, {"port_start": 65520}, {"seconds": float("nan")}])
def test_invalid_plan_is_rejected_before_launch(configuration, tmp_path, change):
    configuration.update(change)
    with pytest.raises(ValueError):
        run_matrix.make_plan(configuration, tmp_path)


def test_raon_attention_bridge_is_explicit_and_speaker_roles_are_distinct(monkeypatch, configuration):
    monkeypatch.setenv("RAON_SOURCE", "/example/source")
    monkeypatch.setenv("RAON_BUNDLE", "/example/bundle")
    worker = {"family": "raon", "gpu": 0, "port": 18000, "side": "left"}
    assert "--mimi-sdpa-window" not in run_matrix.worker_command(worker)
    left = run_matrix.worker_command(worker, configuration)
    right = run_matrix.worker_command({**worker, "side": "right"}, configuration)
    assert "--mimi-sdpa-window" in left
    assert "--speak-first" in left and "--no-speak-first" in right
    assert left[left.index("--mem-fraction-static") + 1] == "0.30"


@pytest.mark.parametrize("fail_one_room", [False, True])
def test_barrier_waits_for_all_surviving_rooms_and_failure_is_isolated(
        monkeypatch, configuration, tmp_path, fail_one_room):
    processes, ready, started = [], [], []

    class Process:
        def __init__(self, *args, **kwargs):
            self.pid = 100000 + len(processes)
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    async def monitor(output, stop, interval):
        await stop.wait()

    async def evaluate(models, output, *, start_barrier, **kwargs):
        name = Path(output).name
        if fail_one_room and name == "raon-raon":
            raise RuntimeError("Deliberate isolated readiness failure")
        await asyncio.sleep(.001 * len(ready))
        ready.append(name)
        await start_barrier()
        assert len(ready) == (8 if fail_one_room else 9)
        started.append(name)
        await asyncio.sleep(.001)
        return {"frames": 1}

    for key in ("SLURM_JOB_ID", "RAON_SOURCE", "RAON_BUNDLE", "RAON_PYTHON",
                "MOSHI_PYTHON", "PERSONAPLEX_PYTHON"):
        monkeypatch.setenv(key, "test")
    monkeypatch.setattr(run_matrix.subprocess, "Popen", Process)
    monkeypatch.setattr(run_matrix.subprocess, "check_output", lambda *a, **k: "test-commit\n")
    monkeypatch.setattr(run_matrix.socket, "create_connection", lambda *a, **k: nullcontext())
    monkeypatch.setattr(run_matrix, "signal_group", lambda p, sig: setattr(p, "returncode", 0))
    monkeypatch.setattr(run_matrix, "monitor", monitor)
    monkeypatch.setattr(evaluation, "evaluate_live", evaluate)
    result = asyncio.run(run_matrix.experiment(configuration, tmp_path))
    matrix = json.loads((tmp_path / "matrix.json").read_text())
    expected = 8 if fail_one_room else 9
    assert result == (1 if fail_one_room else 0)
    assert matrix["completed"] is (not fail_one_room)
    assert matrix["completed_rooms"] == matrix["simultaneous_room_count"] == len(started) == expected
    assert len(matrix["rooms"]) == 9
    assert matrix["rooms"][0]["directory"] == "moshi-moshi"
    assert matrix["start_spread_ms"] is not None
    assert len(processes) == 18 and all(p.returncode == 0 for p in processes)
