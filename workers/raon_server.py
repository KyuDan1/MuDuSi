"""Launch one official Raon SGLang session in an independent worker process."""
import argparse
import functools
import math
import os
from pathlib import Path
import sys


def memory_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("memory fraction must be finite and in (0, 1)")
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def install_speak_first_bridge(session_module):
    """Carry the session option into Raon's decoding config before first audio.

    The validated upstream LocalRealtimeSession copies prompt/sampling but omits
    speak_first when constructing SessionConfig. Its create_session only creates
    prompt state; decoding state is initialized on the first audio frame. Keep
    this compatibility fix local to our process, without editing the checkout.
    """
    factory = session_module.create_session

    @functools.wraps(factory)
    def create_session(**kwargs):
        instance = factory(**kwargs)
        payload = kwargs.get("session") or {}
        if "speak_first" in payload:
            speak_first = payload["speak_first"]
            if not isinstance(speak_first, bool):
                raise ValueError("session.speak_first must be a boolean")
            state = instance._inner.state
            if state.decoding_state is not None:
                raise RuntimeError("Raon initialized decoding before speak_first could be applied")
            state.config.speak_first = speak_first
            # Preserve the effective option in upstream session artifacts too.
            instance._session_params["speak_first"] = speak_first
        return instance

    session_module.create_session = create_session


def install_mimi_sdpa_window_bridge():
    """Use the native HF window mask for Mimi's SDPA/eager attention only.

    Transformers 4.57.1 Mimi calls create_causal_mask even with sliding_window
    configured. A sliding KV cache alone is insufficient for multi-token chunks:
    early queries in a chunk can still see keys outside their own window. Keep
    this opt-in patch process-local and preserve other attention backends.
    """
    from transformers.models.mimi import modeling_mimi
    from transformers.masking_utils import create_sliding_window_causal_mask

    if getattr(modeling_mimi, "_mudusi_window_bridge", False):
        return
    original = modeling_mimi.create_causal_mask

    @functools.wraps(original)
    def create_mask(*args, **kwargs):
        config = kwargs.get("config") if "config" in kwargs else args[0]
        if (getattr(config, "_attn_implementation", None) in ("sdpa", "eager")
                and getattr(config, "sliding_window", None) is not None):
            return create_sliding_window_causal_mask(*args, **kwargs)
        return original(*args, **kwargs)

    modeling_mimi.create_causal_mask = create_mask
    modeling_mimi._mudusi_window_bridge = True
    print("MuDuSi: Mimi SDPA/eager sliding-window mask bridge ENABLED "
          "(process-local; GPU speed and audio quality require validation)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--speak-first", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable Raon's native first-speaker decoding state (no WAV stimulus)")
    parser.add_argument("--prompt", help="System prompt text or an upstream prompt key")
    parser.add_argument("--mem-fraction-static", type=memory_fraction, default=.45,
                        help="SGLang static GPU-memory budget fraction; includes its model/cache")
    parser.add_argument("--max-total-tokens", type=positive_int,
                        help="Optional SGLang KV token-capacity cap; must cover the full episode")
    parser.add_argument("--compile-audio-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--mimi-sdpa-window", action="store_true",
                        help="Apply a process-local Mimi sliding-window mask fix for SDPA/eager")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    import uvicorn
    from demo.realtime.api.app import create_fastapi_app
    from demo.realtime.runtime import session as session_module

    if args.mimi_sdpa_window:
        install_mimi_sdpa_window_bridge()
        # Raon's Mimi decoder runs in a fresh spawn process. Install the hook
        # before preload creates that child; parent module patches do not carry.
        from workers.raon_runtime import install_decoder_spawn_bridge
        install_decoder_spawn_bridge()
    install_speak_first_bridge(session_module)
    runtime = {"compile_audio_modules": args.compile_audio_modules,
               "disable_cuda_graph": args.disable_cuda_graph,
               "mem_fraction_static": args.mem_fraction_static}
    if args.max_total_tokens is not None:
        runtime["max_total_tokens"] = args.max_total_tokens
    # Canonical upstream prompts preserve the intended listen/speak mode. A
    # custom prompt is separate conditioning, not a script for emitted speech.
    prompt = args.prompt or f"eng:full_duplex:{'speak-first' if args.speak_first else 'listen-first'}"
    app = create_fastapi_app(model_path=args.model, ws_path="/realtime/ws",
        session_kwargs={"result_root": str(Path(os.environ.get("ARENA_OUTPUT_DIR", "runs/raon-runtime")) / "native"),
                        "runtime": runtime,
                        "session": {"prompt": prompt, "speak_first": args.speak_first}})
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws_ping_interval=None, ws_ping_timeout=None)


if __name__ == "__main__":
    main()
