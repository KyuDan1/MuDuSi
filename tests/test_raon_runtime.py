"""Verify the mask bridge reaches a fresh spawn process without loading Torch."""
import multiprocessing
import os
import sys
import types

from workers.raon_runtime import install_decoder_spawn_bridge, mimi_window_audio_decoder_worker


def _native_worker(*, audio_tokenizer, output_queue, **kwargs):
    # The injected mask starts disabled inside the child; only the production
    # wrapper can enable it before entering this native-worker stand-in.
    output_queue.put({"pid": os.getpid(), "mask_enabled": audio_tokenizer.mask_enabled,
                      "token": kwargs["test_token"]})


class _DecoderFixture:
    def __getstate__(self):
        return {}

    def __setstate__(self, _):
        # Spawn unpickles arguments before invoking the worker target. Install a
        # fake native module here so the actual production target is exercised.
        self.mask_enabled = False
        server = types.ModuleType("workers.raon_server")
        server.install_mimi_sdpa_window_bridge = lambda: setattr(self, "mask_enabled", True)
        sys.modules["workers.raon_server"] = server
        raon = types.ModuleType("raon")
        modules = types.ModuleType("raon.modules")
        decoder = types.ModuleType("raon.modules.concurrent_audio_decoder")
        decoder.audio_decoder_worker = _native_worker
        modules.concurrent_audio_decoder = decoder
        raon.modules = modules
        sys.modules.update({"raon": raon, "raon.modules": modules,
                            "raon.modules.concurrent_audio_decoder": decoder})


def test_spawned_decoder_installs_window_before_native_loop():
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=mimi_window_audio_decoder_worker,
                              kwargs={"audio_tokenizer": _DecoderFixture(),
                                      "output_queue": output, "test_token": "passed-through"})
    process.start()
    try:
        result = output.get(timeout=15)
        process.join(timeout=15)
        assert process.exitcode == 0
        assert result == {"pid": process.pid, "mask_enabled": True, "token": "passed-through"}
        assert result["pid"] != os.getpid()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        output.close()
        output.join_thread()


def test_spawn_target_installation_is_idempotent(monkeypatch):
    modules = types.ModuleType("raon.modules")
    decoder = types.SimpleNamespace(audio_decoder_worker=_native_worker)
    modules.concurrent_audio_decoder = decoder
    monkeypatch.setitem(sys.modules, "raon", types.ModuleType("raon"))
    monkeypatch.setitem(sys.modules, "raon.modules", modules)
    install_decoder_spawn_bridge()
    assert decoder.audio_decoder_worker is mimi_window_audio_decoder_worker
    install_decoder_spawn_bridge()
    assert decoder.audio_decoder_worker is mimi_window_audio_decoder_worker
