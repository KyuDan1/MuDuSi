"""Process-local compatibility hooks for Raon's spawned audio decoder.

This module deliberately has no Torch imports, so its worker target can be
imported by multiprocessing.spawn before the decoder model is unpickled.
"""


def mimi_window_audio_decoder_worker(**kwargs):
    """Install the Mimi mask in the spawned process, then run the native loop."""
    # A spawn child imports fresh modules; the parent process's patched mask
    # factory is not inherited. The server import must remain inside this call.
    from workers.raon_server import install_mimi_sdpa_window_bridge
    install_mimi_sdpa_window_bridge()
    import os
    print(f"MuDuSi: Mimi window mask installed in decoder child PID {os.getpid()}", flush=True)
    from raon.modules import concurrent_audio_decoder
    native_worker = concurrent_audio_decoder.audio_decoder_worker
    if native_worker is mimi_window_audio_decoder_worker:
        raise RuntimeError("Raon decoder bridge requires a fresh spawn child, not an in-process call")
    native_worker(**kwargs)


def install_decoder_spawn_bridge():
    """Redirect future native spawn targets; call before get_runtime/preload.

    Requires the configured Raon source/environment to be importable. It does
    not affect existing decoder processes or edit the upstream installation.
    """
    from raon.modules import concurrent_audio_decoder
    if concurrent_audio_decoder.audio_decoder_worker is mimi_window_audio_decoder_worker:
        return
    concurrent_audio_decoder.audio_decoder_worker = mimi_window_audio_decoder_worker
    print("MuDuSi: future Raon decoder spawn targets will install the Mimi window mask in-child", flush=True)
