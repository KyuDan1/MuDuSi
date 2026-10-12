"""Artifact-level checks for matrix routing and concurrency verification."""
import itertools
import json
import wave

import numpy as np
import pytest

from workers.verify_matrix import FRAME, RATE, verify_matrix


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_wav(path, samples):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        writer.writeframes(np.asarray(samples, dtype="<i2").tobytes())


@pytest.fixture
def recorded_matrix(tmp_path):
    families = ["moshi", "personaplex", "raon"]
    rooms, metadata, workers = [], [], []
    for index, (left, right) in enumerate(itertools.product(families, repeat=2)):
        room_id = f"{left}-{right}"
        directory = tmp_path / room_id
        directory.mkdir()
        agents = []
        for side_index, (key, family, side) in enumerate(
            [("persona", left, "left"), ("moshi", right, "right")]
        ):
            port = 18000 + index * 2 + side_index
            agents.append({"id": key, "url": f"ws://127.0.0.1:{port}/api/chat"})
            workers.append({"id": f"{room_id}-{side}", "pid": 100 + index * 2 + side_index,
                            "room": room_id, "family": family, "side": side,
                            "gpu": index % 6, "port": port})
        rooms.append({"id": room_id, "directory": room_id, "status": "completed"})
        metadata.append({"id": room_id, "left": left, "right": right, "agents": agents})
        emitted = np.full(FRAME * 2, 100, dtype="<i2")
        received = np.full(FRAME * 2, 50, dtype="<i2")
        inputs = np.concatenate([np.zeros(FRAME, dtype="<i2"), received[:FRAME]])
        states = [{"id": key, "muted": key == "custom", "worker": {}, "transcript": ""}
                  for key in ("persona", "moshi", "custom")]
        for key in ("persona", "moshi"):
            for kind, samples in (("emitted", emitted), ("received", received), ("input", inputs)):
                write_wav(directory / f"{key}-{kind}.wav", samples)
        write_wav(directory / "custom-emitted.wav", np.zeros(FRAME * 2, dtype="<i2"))
        write_json(directory / "summary.json", {
            "completed": True, "sample_rate": RATE, "frames": 2,
            "wall_start_utc": 100 + index * .001, "wall_end_utc": 100.17 + index * .001,
            "seconds": .16, "wall_seconds": .17,
            "final_state": {"agents": states, "performance": {}}})
        trace = [{"tick": tick, "time": tick * .08, "agents": states,
                  "links": [{"source": "persona", "target": "moshi"},
                            {"source": "moshi", "target": "persona"}],
                  "evaluation": {"deadline_lateness_ms": 1}, "performance": {"step_ms": 2}}
                 for tick in (1, 2)]
        (directory / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    write_json(tmp_path / "matrix.json", {"completed": True, "rooms": rooms})
    write_json(tmp_path / "manifest.json", {"family_order": families, "rooms": metadata,
                                           "workers": workers, "configuration": {"gpu_count": 6}})
    return tmp_path


def test_aligned_nine_room_recording_passes(recorded_matrix):
    result = verify_matrix(recorded_matrix)
    assert result["structural_pass"], result["errors"]
    assert result["audio_health_pass"]
    assert result["all_eighteen_agents_emitted_audio"]
    assert result["worker_isolation"]["distinct_pid_count"] == 18
    assert result["worker_isolation"]["distinct_endpoint_count"] == 18
    assert result["simultaneity"]["common_wall_interval_seconds"] == pytest.approx(.162)
    assert result["simultaneity"]["start_spread_ms"] == pytest.approx(8)
    assert json.loads((recorded_matrix / "verification.json").read_text()) == result


def test_one_corrupted_input_sample_fails_routing(recorded_matrix):
    path = recorded_matrix / "moshi-moshi" / "persona-input.wav"
    with wave.open(str(path), "rb") as reader:
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").copy()
    samples[FRAME] += 1
    write_wav(path, samples)
    result = verify_matrix(recorded_matrix)
    assert not result["structural_pass"]
    assert result["errors"] == ["moshi-moshi: persona input/previous-receiver alignment failed"]
    assert not result["rooms"]["moshi-moshi"]["agents"]["persona"]["input_equals_previous_received"]
