import asyncio
import threading

import numpy as np
import pytest

from duplex_arena.asr import ASRObserver


async def wait_until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def test_asr_receives_exact_seeded_receiver_windows_without_optional_dependencies():
    async def run():
        seen = []

        def transcriber(audio):
            seen.append(audio.copy())
            return "  A receiver transcription.  "

        observer = ASRObserver(agent_id="moshi", transcriber=transcriber)
        await observer.start()
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 0.05, 240_000).astype(np.float32)
        for index in range(0, len(expected), 1920):
            block = expected[index:index + 1920]
            observer.feed(block, (index + len(block)) / 24_000)
        await wait_until(lambda: observer.stats["processed_windows"] == 2)
        np.testing.assert_array_equal(seen[0], expected[:120_000])
        np.testing.assert_array_equal(seen[1], expected[120_000:])
        events = observer.drain_events()
        assert [(event["start"], event["end"]) for event in events] == [(0.0, 5.0), (5.0, 10.0)]
        assert all(event["kind"] == "asr" and event["agent_id"] == "moshi" for event in events)
        assert events[0]["text"] == "A receiver transcription."
        assert observer.drain_events() == []
        await observer.close()

    asyncio.run(run())


def test_reset_discards_running_and_queued_results_without_concurrent_model_calls():
    async def run():
        entered, release = threading.Event(), threading.Event()
        calls = []

        def transcriber(audio):
            calls.append(float(audio[0]))
            if len(calls) == 1:
                entered.set()
                release.wait(timeout=2)
            return f"window {audio[0]}"

        observer = ASRObserver(transcriber=transcriber)
        await observer.start()
        try:
            observer.feed(np.ones(120_000, dtype=np.float32), 5)
            await wait_until(entered.is_set)
            observer.feed(np.full(120_000, 2, dtype=np.float32), 10)
            await observer.reset()
            observer.feed(np.full(120_000, 3, dtype=np.float32), 5)
            release.set()
            await wait_until(lambda: observer.stats["processed_windows"] == 1)
            events = observer.drain_events()
            assert len(events) == 1 and events[0]["text"] == "window 3.0"
            assert calls == [1.0, 3.0]
        finally:
            release.set()
            await observer.close()

    asyncio.run(run())


def test_backpressure_drops_oldest_waiting_window_and_records_count():
    async def run():
        observer = ASRObserver(transcriber=lambda _: "recognized")
        await observer.start()
        # No await: the worker cannot consume until after all four are offered.
        for end in (5, 10, 15, 20):
            observer.feed(np.zeros(120_000, dtype=np.float32), end)
        assert observer.stats["pending_windows"] == 2
        assert observer.stats["dropped_windows"] == 2
        await wait_until(lambda: observer.stats["processed_windows"] == 2)
        assert [event["end"] for event in observer.drain_events()] == [15.0, 20.0]
        await observer.close()

    asyncio.run(run())


def test_inference_errors_are_visible_and_worker_recovers():
    async def run():
        calls = 0

        def transcriber(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("inference unavailable")
            return ""

        observer = ASRObserver(transcriber=transcriber)
        await observer.start()
        observer.feed(np.zeros(120_000, dtype=np.float32), 5)
        await wait_until(lambda: observer.stats["error"] is not None)
        assert "inference unavailable" in observer.stats["error"]
        observer.feed(np.zeros(120_000, dtype=np.float32), 10)
        await wait_until(lambda: observer.stats["processed_windows"] == 1)
        assert observer.stats["error"] is None
        assert observer.drain_events() == []
        await observer.close()

    asyncio.run(run())


def test_discontinuities_do_not_concatenate_unrelated_audio_and_input_is_copied():
    async def run():
        seen = []
        observer = ASRObserver(transcriber=lambda pcm: seen.append(pcm.copy()) or "")
        await observer.start()
        observer.feed(np.zeros(60_000, dtype=np.float32), 2.5)
        audio = np.ones(120_000, dtype=np.float32)
        observer.feed(audio, 10)
        audio[:] = 0
        await wait_until(lambda: observer.stats["processed_windows"] == 1)
        assert observer.stats["discontinuities"] == 1
        np.testing.assert_array_equal(seen[0], np.ones(120_000, dtype=np.float32))
        await observer.close()

    asyncio.run(run())


def test_close_cancels_worker_and_suppresses_late_result():
    async def run():
        entered, release = threading.Event(), threading.Event()

        def transcriber(_):
            entered.set()
            release.wait(timeout=2)
            return "obsolete"

        observer = ASRObserver(transcriber=transcriber)
        await observer.start()
        observer.feed(np.zeros(120_000, dtype=np.float32), 5)
        await wait_until(entered.is_set)
        await observer.close()
        release.set()
        await asyncio.sleep(0.01)
        assert not observer.stats["ready"]
        assert observer.drain_events() == []
        with pytest.raises(RuntimeError, match="closed"):
            await observer.start()

    asyncio.run(run())


def test_load_failure_is_exposed_and_no_worker_is_started():
    async def run():
        observer = ASRObserver()

        def fail_load():
            raise RuntimeError("install optional ASR dependencies")

        observer._load_transcriber = fail_load
        with pytest.raises(RuntimeError, match="optional"):
            await observer.start()
        assert not observer.stats["ready"]
        assert not observer.stats["loading"]
        assert "optional" in observer.stats["error"]
        await observer.close()

    asyncio.run(run())


@pytest.mark.parametrize("audio,timestamp", [(np.array([np.nan]), 1), (np.zeros((2, 2)), 1), (np.zeros(2), np.inf)])
def test_invalid_inputs_are_rejected(audio, timestamp):
    observer = ASRObserver(transcriber=lambda _: "")
    with pytest.raises(ValueError):
        observer.feed(audio, timestamp)
