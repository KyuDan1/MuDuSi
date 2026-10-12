import asyncio
import json
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from duplex_arena.adapters import (
    AdapterBackpressureError, AdapterError, FRAME_SIZE, MoshiWebSocketAdapter,
    PCMWebSocketAdapter, PersonaPlexWebSocketAdapter, build_adapters, validate_pcm,
)
from workers.echo_worker import echo_session


async def eventually(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("Asynchronous condition did not become true")


def worker_url(server):
    return f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"


def test_echo_is_bidirectional_and_cleans_up():
    async def scenario():
        async with serve(echo_session, "127.0.0.1", 0) as server:
            adapter = PCMWebSocketAdapter(worker_url(server))
            await adapter.start()
            assert adapter.ready
            np.testing.assert_array_equal(adapter.pull_audio(), np.zeros(FRAME_SIZE))
            assert adapter.stats["underrun_frames"] == 1
            pcm = np.linspace(-0.2, 0.2, FRAME_SIZE, dtype=np.float32)
            await adapter.push_audio(pcm)
            pcm[:] = 0  # caller mutation must not alter queued audio
            await eventually(lambda: adapter.stats["received_samples"] >= FRAME_SIZE)
            expected = np.linspace(-0.1, 0.1, FRAME_SIZE, dtype=np.float32)
            np.testing.assert_allclose(adapter.pull_audio(), expected, atol=1e-7)
            assert "no speech model" in "".join(adapter.drain_text())
            assert adapter.drain_text() == []
            await adapter.close()
            await adapter.close()
            assert not adapter.ready and not adapter._tasks
            with pytest.raises(AdapterError):
                adapter.pull_audio()
            await adapter.start()  # reset must create a clean worker session
            assert adapter.stats["received_samples"] == 0
            await adapter.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("pcm", [np.zeros((1, FRAME_SIZE)), np.zeros(20),
                                 np.zeros(FRAME_SIZE, dtype=np.int16),
                                 np.full(FRAME_SIZE, np.nan),
                                 np.full(FRAME_SIZE, np.inf), np.full(FRAME_SIZE, 1.1)])
def test_audio_validation_rejects_bad_shapes_and_values(pcm):
    with pytest.raises(ValueError):
        validate_pcm(pcm)


def test_input_overflow_is_visible_error():
    async def scenario():
        async with serve(echo_session, "127.0.0.1", 0) as server:
            adapter = PCMWebSocketAdapter(worker_url(server), max_input_frames=1)
            await adapter.start()
            await adapter.push_audio(np.zeros(FRAME_SIZE, dtype=np.float32))
            with pytest.raises(AdapterBackpressureError):
                await adapter.push_audio(np.zeros(FRAME_SIZE, dtype=np.float32))
            assert adapter.stats["input_overflow_frames"] == 1
            assert not adapter.ready
            await adapter.close()
    asyncio.run(scenario())


def test_output_overflow_keeps_recent_audio_and_counts_loss():
    adapter = PCMWebSocketAdapter("ws://localhost:1", max_output_frames=1)
    adapter._ready = True
    adapter._append_audio(np.full(FRAME_SIZE, 0.1, dtype=np.float32))
    adapter._append_audio(np.full(FRAME_SIZE, 0.2, dtype=np.float32))
    assert adapter.stats["output_dropped_samples"] == FRAME_SIZE
    np.testing.assert_allclose(adapter.pull_audio(), 0.2)


@pytest.mark.parametrize("payload", [b"bad", np.full(FRAME_SIZE, np.nan, dtype="<f4").tobytes(),
                                   '{"type":"error","message":"GPU out of memory"}',
                                   '{"type":"unknown"}', '[]'])
def test_worker_failure_is_not_silent_audio(payload):
    async def scenario():
        async def bad_worker(ws):
            await ws.recv()
            await ws.send('{"type":"status","state":"ready"}')
            await ws.recv()
            await ws.send(payload)
            await ws.wait_closed()
        async with serve(bad_worker, "127.0.0.1", 0) as server:
            adapter = PCMWebSocketAdapter(worker_url(server))
            await adapter.start()
            await adapter.push_audio(np.zeros(FRAME_SIZE, dtype=np.float32))
            await eventually(lambda: adapter.error is not None)
            with pytest.raises(AdapterError):
                adapter.pull_audio()
            await adapter.close()
    asyncio.run(scenario())


def test_echo_worker_rejects_malformed_input():
    async def scenario():
        async with serve(echo_session, "127.0.0.1", 0) as server:
            async with connect(worker_url(server)) as ws:
                await ws.send(json.dumps({"type": "init", "protocol": "duplex-arena.pcm.v1",
                                           "sample_rate": 24000, "channels": 1,
                                           "frame_size": FRAME_SIZE, "format": "f32le"}))
                assert json.loads(await ws.recv())["state"] == "ready"
                await ws.send(np.full(FRAME_SIZE, np.nan, dtype="<f4").tobytes())
                error = json.loads(await ws.recv())
                assert error["type"] == "error" and "NaN" in error["message"]
    asyncio.run(scenario())


@pytest.mark.parametrize("legacy_codec", [False, True])
def test_native_wire_protocol_with_both_sphn_api_shapes(monkeypatch, legacy_codec):
    # Transport contract test. A reversible fake codec isolates the websocket
    # protocol; this test does not claim to run model weights or real Opus.
    class FakeWriter:
        def __init__(self, sample_rate):
            assert sample_rate == 24000
        def append_pcm(self, pcm):
            self.data = pcm.astype("<f4").tobytes()
            return None if legacy_codec else self.data
        def read_bytes(self):
            return self.data

    class FakeReader:
        def __init__(self, sample_rate):
            assert sample_rate == 24000
        def append_bytes(self, data):
            self.data = np.frombuffer(data, dtype="<f4")
            return None if legacy_codec else self.data
        def read_pcm(self):
            return self.data

    monkeypatch.setitem(sys.modules, "sphn", SimpleNamespace(OpusStreamWriter=FakeWriter, OpusStreamReader=FakeReader))

    async def scenario():
        async def native_worker(ws):
            await ws.send(b"\x00")
            audio = await ws.recv()
            assert audio[0] == 1
            await ws.send(audio)
            await ws.send(b"\x02hello")
            await ws.wait_closed()
        async with serve(native_worker, "127.0.0.1", 0) as server:
            adapter = MoshiWebSocketAdapter(worker_url(server))
            await adapter.start()
            await adapter.push_audio(np.full(FRAME_SIZE, 0.25, dtype=np.float32))
            await eventually(lambda: adapter.stats["received_samples"] == FRAME_SIZE)
            np.testing.assert_allclose(adapter.pull_audio(), 0.25)
            assert adapter.drain_text() == ["hello"]
            await adapter.close()
    asyncio.run(scenario())


def test_personaplex_exact_prompt_query_and_path_validation():
    adapter = PersonaPlexWebSocketAdapter("ws://localhost:8999/api/chat?existing=yes",
                                          voice_prompt="NATF2.pt", text_prompt="Discuss A & B?")
    query = parse_qs(urlsplit(adapter.url).query)
    assert query == {"existing": ["yes"], "voice_prompt": ["NATF2.pt"], "text_prompt": ["Discuss A & B?"]}
    with pytest.raises(ValueError):
        PersonaPlexWebSocketAdapter("ws://localhost:8999", voice_prompt="../voice.pt")


def test_real_opus_roundtrip_through_native_websocket():
    sphn = pytest.importorskip("sphn", reason="Install [models] to test real CPU Opus")

    async def scenario():
        async def codec_worker(ws):
            reader = sphn.OpusStreamReader(24000)
            writer = sphn.OpusStreamWriter(24000)
            await ws.send(b"\x00")
            async for packet in ws:
                assert packet[0] == 1
                decoded = reader.append_bytes(packet[1:])
                if decoded.size:
                    encoded = writer.append_pcm(decoded)
                    if encoded:
                        await ws.send(b"\x01" + encoded)

        async with serve(codec_worker, "127.0.0.1", 0) as server:
            adapter = MoshiWebSocketAdapter(worker_url(server))
            await adapter.start()
            source = (0.2 * np.sin(2 * np.pi * 440 * np.arange(FRAME_SIZE * 6) / 24000)).astype(np.float32)
            for frame in np.split(source, 6):
                await adapter.push_audio(frame)
            await eventually(lambda: adapter.stats["received_samples"] >= source.size)
            decoded = adapter.pull_audio(source.size)
            assert np.all(np.isfinite(decoded))
            assert 0.08 < np.sqrt(np.mean(decoded ** 2)) < 0.2
            dominant_hz = np.argmax(np.abs(np.fft.rfft(decoded))) * 24000 / decoded.size
            assert abs(dominant_hz - 440) < 4
            assert adapter.stats["output_dropped_samples"] == 0
            await adapter.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("agents", [[], [{"id": "typo", "backend": "pcm", "url": "ws://localhost:1"}],
                                    [{"id": "persona", "backend": "unknown", "url": "ws://localhost:1"}],
                                    [{"id": "persona", "backend": "pcm", "url": "http://localhost:1"}],
                                    [{"id": "persona", "backend": "pcm", "url": "ws://localhost:1"}] * 2])
def test_config_rejects_mistakes(tmp_path, agents):
    path = tmp_path / "workers.json"
    path.write_text(json.dumps({"agents": agents}))
    with pytest.raises(ValueError):
        build_adapters(path)


def test_config_load_does_not_import_model_dependencies(tmp_path):
    path = tmp_path / "workers.json"
    path.write_text(json.dumps({"agents": [
        {"id": "persona", "backend": "personaplex", "url": "ws://localhost:8999/api/chat"},
        {"id": "moshi", "backend": "moshi", "url": "ws://localhost:8998/api/chat"},
        {"id": "custom", "backend": "pcm", "url": "ws://localhost:9001/stream"}]}))
    adapters = build_adapters(path)
    assert set(adapters) == {"persona", "moshi", "custom"}
    assert all(not adapter.ready for adapter in adapters.values())
