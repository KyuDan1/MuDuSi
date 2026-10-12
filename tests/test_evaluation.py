import asyncio
import json
import wave

import numpy as np
import pytest
from websockets.asyncio.server import serve

from duplex_arena.evaluation import evaluate_live
from workers.echo_worker import echo_session


def read_pcm(path):
    with wave.open(str(path)) as f:
        assert f.getframerate() == 24000
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")


def test_live_export_records_exact_previous_receiver_input(tmp_path):
    async def run():
        async with serve(echo_session, "127.0.0.1", 0) as server:
            config = tmp_path / "workers.json"
            config.write_text(json.dumps({"agents": [{"id": "moshi", "backend": "pcm",
                "url": f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"}]}))
            output = tmp_path / "run"
            summary = await evaluate_live(config, output, seconds=.4, scenario="cocktail")
            assert summary["completed"] and summary["frames"] == 5
            received = read_pcm(output / "moshi-received.wav")
            inputs = read_pcm(output / "moshi-input.wav")
            np.testing.assert_array_equal(inputs[:1920], 0)
            np.testing.assert_array_equal(inputs[1920:], received[:-1920])
            assert np.any(read_pcm(output / "moshi-emitted.wav"))
            frames = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
            assert len(frames) == 5 and frames[-1]["mode"] == "live"
            assert summary["wall_seconds"] >= .4
            (output / "asr.json").write_text(json.dumps({"model": "test-observer", "files": {
                "moshi-received.wav": {"segments": [{"id": 0, "start": 0, "end": .16, "text": "heard speech"}]},
                "moshi-emitted.wav": {"segments": [{"id": 0, "start": 0, "end": .16, "text": "source speech"}]}}}))
            from duplex_arena.replay import ReplaySession
            replay = ReplaySession(output)
            assert replay.snapshot()["replay"]["frame"] == 0
            await replay.step()
            np.testing.assert_allclose(replay.recordings["moshi"][0], received[:1920] / 32767, atol=1e-7)
            assert replay.snapshot()["agents"][1]["asr_transcript"] == ""
            await replay.step()
            assert replay.snapshot()["agents"][1]["asr_transcript"] == "heard speech"
            assert replay.snapshot()["agents"][1]["transcript"] != "heard speech"
            await replay.reset()
            assert replay.cursor == 0 and not replay.recordings["moshi"]
    asyncio.run(run())


def test_replay_rejects_acoustic_edits(tmp_path):
    from fastapi.testclient import TestClient
    from duplex_arena.engine import ArenaEngine
    from duplex_arena.server import create_app
    engine = ArenaEngine(horizon=1)
    (tmp_path / "trace.jsonl").write_text(json.dumps(engine.step()) + "\n")
    for key in engine.agents:
        with wave.open(str(tmp_path / f"{key}-received.wav"), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(1920 * 2))
    with TestClient(create_app(replay=tmp_path)) as client:
        assert client.get("/api/health").json()["mode"] == "replay"
        assert client.post("/api/control", json={"action": "configure", "noise_db": -30}).status_code == 400
        assert client.post("/api/control", json={"action": "step"}).status_code == 200
        assert client.get("/api/audio/moshi.wav").status_code == 200


def test_live_startup_failure_is_saved_and_raised(tmp_path):
    config = tmp_path / "workers.json"
    config.write_text(json.dumps({"agents": [{"id": "moshi", "backend": "pcm",
        "url": "ws://127.0.0.1:1/stream", "startup_timeout": 1}]}))
    with pytest.raises(Exception):
        asyncio.run(evaluate_live(config, tmp_path / "run", seconds=.08))
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert not summary["completed"] and summary["error"] and summary["frames"] == 0


def test_nine_live_rooms_share_start_barrier_and_keep_receiver_history(tmp_path):
    async def run():
        gate = asyncio.Event()
        ready = []

        async def barrier():
            ready.append(True)
            if len(ready) == 9:
                gate.set()
            await gate.wait()

        async with serve(echo_session, "127.0.0.1", 0) as server:
            config = tmp_path / "workers.json"
            config.write_text(json.dumps({"agents": [{"id": "moshi", "backend": "pcm",
                "url": f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"}]}))
            results = await asyncio.gather(*(evaluate_live(config, tmp_path / str(i),
                seconds=.16, seed=i, scenario="cocktail", start_barrier=barrier) for i in range(9)))
        starts = [r["wall_start_utc"] for r in results]
        assert max(starts) - min(starts) < .08
        assert all(r["completed"] and r["frames"] == 2 for r in results)
        received = []
        for i in range(9):
            pcm = read_pcm(tmp_path / str(i) / "moshi-received.wav")
            inputs = read_pcm(tmp_path / str(i) / "moshi-input.wav")
            np.testing.assert_array_equal(inputs[:1920], 0)
            np.testing.assert_array_equal(inputs[1920:], pcm[:-1920])
            received.append(pcm)
        assert all(not np.array_equal(received[0], pcm) for pcm in received[1:])
    asyncio.run(run())
