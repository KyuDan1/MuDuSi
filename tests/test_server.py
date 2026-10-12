import io
import threading
import time
import wave

import numpy as np
from fastapi.testclient import TestClient
from duplex_arena.asr import ASRObserver
from duplex_arena.engine import ArenaEngine
from duplex_arena.server import create_app


def test_shared_clock_controls_export_audio():
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/state").json()["tick"] == 0
        assert client.get("/api/audio/moshi.wav").status_code == 409
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["tick"] == 0
            for _ in range(8):
                state = client.post("/api/control", json={"action": "step"}).json()
            assert state["tick"] == 8
            assert state["running"] is False
            assert state["mode"] == "demo"
        audio = client.get("/api/audio/moshi.wav")
        assert audio.status_code == 200
        with wave.open(io.BytesIO(audio.content), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnframes() == 8 * 1920
        trace = client.get("/api/export").json()
        assert len(trace["frames"]) == 8
        assert trace["frames"][-1]["time"] == .64
        reset = client.post("/api/control", json={"action": "reset", "scenario": "cocktail", "seed": 9}).json()
        assert reset["tick"] == 0 and reset["seed"] == 9
        assert reset["room"]["obstacles"]
        assert client.get("/api/export").json()["frames"] == []


def test_invalid_controls_do_not_advance_clock():
    with TestClient(create_app()) as client:
        for payload in [{"action": "explode"}, {"action": "configure", "noise_db": 99},
                        {"action": "configure", "agents": [{"id": "unknown"}]}]:
            assert client.post("/api/control", json=payload).status_code == 422
        assert client.post("/api/control", json={"action": "configure", "agents": [{"id": "moshi", "position": [30, 1, 0]}]}).status_code == 400
        assert client.get("/api/state").json()["tick"] == 0
        assert client.get("/api/audio/missing.wav").status_code == 404


def test_configuration_and_pcm_websocket():
    with TestClient(create_app()) as client:
        result = client.post("/api/control", json={"action": "configure", "noise_db": -40,
                             "reverb": .3, "agents": [{"id": "persona", "muted": True}]}).json()
        assert result["noise_db"] == -40
        assert result["agents"][0]["muted"]
        with client.websocket_connect("/ws/audio/persona") as ws:
            client.post("/api/control", json={"action": "step"})
            assert len(ws.receive_bytes()) == 1920 * 4


def test_no_silent_fallback_for_failed_live_config(tmp_path):
    config = tmp_path / "models.json"
    config.write_text('{"agents":[{"id":"persona","backend":"pcm","url":"ws://127.0.0.1:1"}]}')
    with TestClient(create_app(models=str(config))) as client:
        state = client.get("/api/state").json()
        assert state["mode"] == "live"
        assert state["error"]
        assert state["tick"] == 0
        assert client.post("/api/control", json={"action": "play"}).status_code == 400


def _wait_for_asr_state(client, expected_text, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/api/state").json()
        if any(event["text"] == expected_text for event in state["asr_events"]):
            return state
        time.sleep(0.01)
    raise AssertionError(f"ASR result did not reach the viewer: {expected_text}")


def test_receiver_asr_uses_real_mixture_and_exports_late_paused_result():
    received = []

    def transcribe(pcm):
        received.append(pcm.copy())
        return "Recognized receiver audio"

    observer = ASRObserver(agent_id="moshi", transcriber=transcribe)
    reference = ArenaEngine(seed=23)
    expected = []
    with TestClient(create_app(seed=23, asr=observer)) as client:
        for _ in range(63):
            reference.step()
            expected.append(reference.last_inputs["moshi"].copy())
            assert client.post("/api/control", json={"action": "step"}).status_code == 200
        state = _wait_for_asr_state(client, "Recognized receiver audio")
        assert state["tick"] == 63 and not state["running"]
        assert state["asr"]["ready"]
        np.testing.assert_array_equal(received[0], np.concatenate(expected)[:120_000])
        moshi = next(agent for agent in state["agents"] if agent["id"] == "moshi")
        assert moshi["asr_transcript"] == "Recognized receiver audio"
        assert moshi["transcript_kind"] == "scripted"
        event = state["asr_events"][0]
        assert (event["start"], event["end"], event["agent_id"], event["kind"]) == (0.0, 5.0, "moshi", "asr")
        # Inference can finish after the last frame: export must retain it
        # without requiring an additional acoustic simulation step.
        exported = client.get("/api/export").json()
        assert exported["asr_events"] == state["asr_events"]
        assert len(exported["frames"]) == 63


def test_receiver_asr_reset_clears_ui_export_and_ignores_inflight_old_episode():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def transcribe(pcm):
        calls.append(pcm.copy())
        if len(calls) == 1:
            entered.set()
            release.wait(timeout=3)
            return "Obsolete episode"
        return "New episode"

    observer = ASRObserver(agent_id="moshi", transcriber=transcribe)
    with TestClient(create_app(asr=observer)) as client:
        try:
            for _ in range(63):
                assert client.post("/api/control", json={"action": "step"}).status_code == 200
            assert entered.wait(timeout=2)
            reset = client.post("/api/control", json={"action": "reset", "seed": 91}).json()
            assert reset["asr_events"] == []
            assert all(agent["asr_transcript"] == "" for agent in reset["agents"])
            assert client.get("/api/export").json()["asr_events"] == []
            release.set()
            for _ in range(63):
                assert client.post("/api/control", json={"action": "step"}).status_code == 200
            state = _wait_for_asr_state(client, "New episode")
            assert [event["text"] for event in state["asr_events"]] == ["New episode"]
            assert state["asr"]["processed_windows"] == 1
            exported = client.get("/api/export").json()
            assert [event["text"] for event in exported["asr_events"]] == ["New episode"]
        finally:
            release.set()


def test_optional_asr_startup_error_is_visible_without_crashing_server():
    observer = ASRObserver()

    def fail_load():
        raise RuntimeError("Optional ASR model could not load")

    observer._load_transcriber = fail_load
    with TestClient(create_app(asr=observer)) as client:
        state = client.get("/api/state").json()
        assert state["asr"]["enabled"] and not state["asr"]["ready"]
        assert "could not load" in state["asr"]["error"]
