"""Wall-clock model evaluation using the same Session as the viewer."""
import asyncio
import json
import math
from pathlib import Path
import time
import wave

import numpy as np

from .server import Session, IDS


async def evaluate_live(models, output, seconds=60, seed=42, scenario="quiet", configuration=None, *, start_barrier=None):
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("seconds must be in (0, 3600]")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    session = Session(seed=seed, scenario=scenario, models=str(models))
    session.engine.horizon = math.ceil(seconds / .08)
    if configuration:
        session.engine.configure(**configuration)
    writers = {}
    started = None
    wall_start_utc = None
    pre_barrier_worker_stats = {}
    startup_worker_stats = {}
    frames = 0
    lateness = []
    failure = None
    try:
        await session.connect_models()
        for key in IDS:
            for kind in ("received", "emitted", "input"):
                writer = wave.open(str(root / f"{key}-{kind}.wav"), "wb")
                writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                writers[key, kind] = writer
        pre_barrier_worker_stats = {key: dict(adapter.stats) for key, adapter in session.adapters.items()}
        if start_barrier is not None:
            await start_barrier()
        startup_worker_stats = {key: dict(adapter.stats) for key, adapter in session.adapters.items()}
        wall_start_utc = time.time()
        started = time.perf_counter()
        with (root / "trace.jsonl").open("w") as trace:
            for tick in range(session.engine.horizon):
                deadline = started + tick * .08
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                late = max(0, time.perf_counter() - deadline)
                lateness.append(late)
                if late > .08:
                    session.overruns += 1
                # This is exactly the previous receiver block fed to the adapter.
                inputs = {k: v.copy() for k, v in session.engine.last_inputs.items()}
                state = await session.step()
                state["evaluation"] = {"wall_seconds": time.perf_counter() - started,
                                       "deadline_lateness_ms": late * 1000}
                trace.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
                for kind, blocks in (("input", inputs), ("received", session.engine.last_inputs),
                                     ("emitted", session.engine.last_outputs)):
                    for key, pcm in blocks.items():
                        writers[key, kind].writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
                frames += 1
            await asyncio.sleep(max(0, started + frames * .08 - time.perf_counter()))
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started if started else 0
        state = session.snapshot()
        summary = {"schema_version": 1, "mode": "live", "completed": failure is None,
                   "error": failure, "wall_start_utc": wall_start_utc, "wall_end_utc": time.time(),
                   "pre_barrier_worker_stats": pre_barrier_worker_stats,
                   "startup_worker_stats": startup_worker_stats, "frames": frames, "seconds": frames * .08,
                   "wall_seconds": elapsed, "wall_per_audio_second": elapsed / (frames * .08) if frames else None,
                   "deadline_lateness_p95_ms": float(np.percentile(lateness, 95) * 1000) if lateness else None,
                   "seed": seed, "scenario": scenario, "sample_rate": 24000,
                   "transcripts": "model generated text; not receiver ASR",
                   "input_alignment": "input[t] = clipped received[t-1]; initial input is silence",
                   "final_state": state}
        (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        for writer in writers.values():
            writer.close()
        await session.close_models()
    return summary
