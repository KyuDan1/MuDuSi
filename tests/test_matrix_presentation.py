import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from duplex_arena.matrix import create_matrix_app, load_matrix


def make_matrix(root):
    rooms = []
    for i in range(9):
        key = f"room-{i}"
        folder = root / key
        folder.mkdir()
        agents = [dict(id=k, worker={"ready": True, "worker_name": "test-model"})
                  for k in ("persona", "moshi")]
        (folder / "trace.jsonl").write_text(json.dumps({"agents": agents}) + "\n")
        (folder / "summary.json").write_text('{"completed": true}')
        for key_agent in ("persona", "moshi"):
            with wave.open(str(folder / f"{key_agent}-emitted.wav"), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(np.full(1920, (i + 1) * 1000, dtype="<i2").tobytes())
        rooms.append(dict(id=key, left="Test left", right="Test right"))
    manifest = dict(rooms=rooms, completed=True, simultaneous_room_count=9, start_spread_ms=2.5)
    (root / "matrix.json").write_text(json.dumps(manifest))
    return manifest


def test_matrix_audio_stays_in_requested_room(tmp_path):
    make_matrix(tmp_path)
    with TestClient(create_matrix_app(tmp_path)) as client:
        data = client.get("/api/matrix").json()
        assert data["concurrency"]["simultaneous_room_count"] == 9
        for i in range(9):
            response = client.get(f"/api/matrix/audio/room-{i}.wav")
            with wave.open(io.BytesIO(response.content)) as wav:
                samples = np.frombuffer(wav.readframes(1920), dtype="<i2")
            np.testing.assert_allclose(samples, (i + 1) * 2000, atol=1)
        assert client.get("/api/matrix/audio/missing.wav").status_code == 404


@pytest.mark.parametrize("invalid", ["outside", "duplicate", "missing"])
def test_matrix_rejects_unsafe_or_incomplete_room_sets(tmp_path, invalid):
    manifest = make_matrix(tmp_path)
    if invalid == "outside":
        manifest["rooms"][0]["directory"] = "../outside"
    elif invalid == "duplicate":
        manifest["rooms"][1]["id"] = manifest["rooms"][0]["id"]
    else:
        manifest["rooms"].pop()
    (tmp_path / "matrix.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_matrix(tmp_path)
