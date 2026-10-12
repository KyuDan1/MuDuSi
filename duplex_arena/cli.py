"""Console entry points for a viewer server and reproducible CPU evaluations."""
import argparse
import json
import math
from pathlib import Path
import wave

import numpy as np


def evaluate(seed=42, scenario="roundtable", seconds=18.0, output="runs/evaluation"):
    from .engine import ArenaEngine
    horizon = max(1, math.ceil(seconds / 0.08))
    engine = ArenaEngine(seed=seed, scenario=scenario, horizon=horizon)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for key in engine.last_inputs:
            writer = wave.open(str(root / f"{key}-received.wav"), "wb")
            writer.setparams((1, 2, engine.sample_rate, 0, "NONE", "not compressed"))
            writers[key] = writer
        with (root / "trace.jsonl").open("w") as trace:
            for _ in range(horizon):
                state = engine.step()
                trace.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
                for key, pcm in engine.last_inputs.items():
                    writers[key].writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        summary = {"schema_version": 1, "mode": "synthetic_demo", "seed": seed, "scenario": scenario,
                   "seconds": engine.time, "sample_rate": engine.sample_rate, "frames": horizon,
                   "metrics": state["metrics"], "reward_kind": "acoustic_heuristic",
                   "captions": "scripted; not recognized speech", "configuration": engine.snapshot()["room"]}
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        for writer in writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(prog="duplex-arena", description="Full-duplex speech simulation lab")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local API and built 3D viewer")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.add_argument("--seed", default=42, type=int)
    serve.add_argument("--scenario", default="roundtable", choices=["roundtable", "interruption", "cocktail", "quiet"])
    serve.add_argument("--models", help="JSON worker configuration; omit for a GPU-free synthetic demo")
    serve.add_argument("--replay", help="Replay a completed evaluation directory without contacting GPU workers")
    serve.add_argument("--asr-model", help="Optional faster-whisper model, e.g. tiny; downloads weights on first use")
    serve.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    serve.add_argument("--asr-agent", default="moshi", choices=["persona", "moshi", "custom"])
    evaluate_parser = commands.add_parser("evaluate", help="Write seeded demo trace and per-receiver WAV files")
    evaluate_parser.add_argument("--seed", default=42, type=int)
    evaluate_parser.add_argument("--scenario", default="roundtable", choices=["roundtable", "interruption", "cocktail", "quiet"])
    evaluate_parser.add_argument("--seconds", default=18.0, type=float)
    evaluate_parser.add_argument("--output", default="runs/evaluation")
    evaluate_parser.add_argument("--models", help="Evaluate real workers at wall-clock 1x instead of the synthetic demo")
    evaluate_parser.add_argument("--configuration", help="JSON acoustic controls (noise_db, reverb, agents)")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        observer = None
        if args.asr_model:
            from .asr import ASRObserver
            observer = ASRObserver(model_name=args.asr_model, device=args.asr_device, agent_id=args.asr_agent)
        uvicorn.run(create_app(seed=args.seed, scenario=args.scenario, models=args.models, asr=observer, replay=args.replay), host=args.host, port=args.port)
    else:
        if not math.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 3600:
            parser.error("--seconds must be in (0, 3600]")
        if args.models:
            import asyncio
            from .evaluation import evaluate_live
            configuration = json.loads(Path(args.configuration).read_text()) if args.configuration else None
            result = asyncio.run(evaluate_live(args.models, args.output, args.seconds, args.seed, args.scenario, configuration))
        else:
            if args.configuration:
                parser.error("--configuration requires --models")
            result = evaluate(args.seed, args.scenario, args.seconds, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
