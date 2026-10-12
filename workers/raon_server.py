"""Launch the official Raon SGLang runtime from an existing source/bundle."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8998)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    import uvicorn
    from demo.realtime.api.app import create_fastapi_app
    app = create_fastapi_app(model_path=args.model, ws_path="/realtime/ws",
        session_kwargs={"result_root": str(Path(os.environ.get("ARENA_OUTPUT_DIR", "runs/raon-runtime")) / "native"),
                        "runtime": {"compile_audio_modules": True,
                        "disable_cuda_graph": False, "mem_fraction_static": .45},
                        "session": {"prompt": "You are engaging in real-time conversation. Answer in English.",
                                    "speak_first": False}})
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws_ping_interval=None, ws_ping_timeout=None)


if __name__ == "__main__":
    main()
