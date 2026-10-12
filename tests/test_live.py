"""Whole live simulation path with real CPU workers; no model weights."""
import asyncio
import json
import wave

import numpy as np
import pytest
from websockets.asyncio.server import serve

from duplex_arena.adapters import FRAME_SIZE, PCMWebSocketAdapter
from duplex_arena.server import Session
from workers.echo_worker import echo_session
from workers.wav_worker import load_stimulus, stimulus_session


async def wait_for_samples(adapter, count):
    for _ in range(200):
        if adapter.stats["received_samples"] >= count:
            return
        if adapter.error:
            raise AssertionError(adapter.error)
        await asyncio.sleep(0.005)
    raise AssertionError("Worker audio did not arrive")


def test_live_session_routes_three_microphone_streams(tmp_path):
    async def scenario():
        async with serve(echo_session, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"
            path = tmp_path / "live.json"
            path.write_text(json.dumps({"agents": [
                {"id": key, "backend": "pcm", "url": url} for key in ("persona", "moshi", "custom")]}))
            session = Session(seed=42, scenario="cocktail", models=str(path))
            try:
                await session.connect_models()
                for tick in range(5):
                    await session.step()
                    await asyncio.gather(*(wait_for_samples(a, (tick + 1) * FRAME_SIZE)
                                           for a in session.adapters.values()))
                state = session.snapshot()
                assert state["mode"] == "live"
                assert len(state["agents"]) == 3
                for agent in state["agents"]:
                    assert agent["worker"]["worker_name"] == "cpu-echo"
                    assert agent["worker"]["input_frames"] == 5
                    assert "no speech model" in agent["transcript"]
                    assert agent["transcript_kind"] == "model"
                    assert np.any(np.abs(session.engine.last_outputs[agent["id"]]) > 1e-7)
                assert state["events"] and all(event["kind"] == "model" for event in state["events"])
                old = list(session.adapters.values())
                await session.reset()
                assert all(not adapter.ready for adapter in old)
                assert all(adapter.stats["input_frames"] == 0 for adapter in session.adapters.values())
            finally:
                await session.close_models()
    asyncio.run(scenario())


def test_wav_stimulus_plays_then_silence_and_restarts(tmp_path):
    wav = tmp_path / "seed.wav"
    source = (np.arange(FRAME_SIZE + 100, dtype=np.int16) - 800)
    with wave.open(str(wav), "wb") as target:
        target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        target.writeframes(source.astype("<i2").tobytes())
    audio = load_stimulus(wav)

    async def scenario():
        async def handler(ws):
            await stimulus_session(ws, audio)
        async with serve(handler, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"
            adapter = PCMWebSocketAdapter(url)
            try:
                await adapter.start()
                assert adapter.stats["worker_name"] == "wav-stimulus"
                blocks = []
                for tick in range(3):
                    await adapter.push_audio(np.zeros(FRAME_SIZE, dtype=np.float32))
                    await wait_for_samples(adapter, (tick + 1) * FRAME_SIZE)
                    blocks.append(adapter.pull_audio())
                emitted = np.concatenate(blocks)
                np.testing.assert_array_equal(emitted[:audio.size], audio)
                np.testing.assert_array_equal(emitted[audio.size:], 0)
                await adapter.close()
                await adapter.start()
                await adapter.push_audio(np.zeros(FRAME_SIZE, dtype=np.float32))
                await wait_for_samples(adapter, FRAME_SIZE)
                np.testing.assert_array_equal(adapter.pull_audio(), audio[:FRAME_SIZE])
            finally:
                await adapter.close()
    asyncio.run(scenario())


def test_stimulus_rejects_wrong_sample_rate(tmp_path):
    wav = tmp_path / "wrong.wav"
    with wave.open(str(wav), "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(b"\x00\x00" * 10)
    with pytest.raises(ValueError, match="24 kHz"):
        load_stimulus(wav)
