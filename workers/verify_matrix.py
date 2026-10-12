"""Verify a completed ordered-pair matrix from its saved artifacts (CPU only).

Exit status reflects structural/routing validity. Audio health and generation
activity are reported separately; neither ASR nor silence proves comprehension.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
import wave

import numpy as np

FRAME = 1920
RATE = 24000
AGENTS = ("persona", "moshi")
COUNTERS = ("input_overflow_frames", "output_dropped_samples", "underrun_frames",
            "underrun_samples", "text_dropped_messages", "decoded_clipped_samples")


def read_json(path):
    return json.loads(path.read_text())


def read_pcm(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError(f"Expected mono PCM16 at {RATE} Hz: {path.name}")
        count = wav.getnframes()
        pcm = np.frombuffer(wav.readframes(count), dtype="<i2")
        if len(pcm) != count:
            raise ValueError(f"Truncated WAV: {path.name}")
        return pcm


def audio_stats(pcm):
    values = pcm.astype(np.float64) / 32767
    rms = float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.
    usable = len(values) // FRAME * FRAME
    frame_rms = np.sqrt(np.mean(values[:usable].reshape(-1, FRAME) ** 2, axis=1)) if usable else np.array([])
    active = frame_rms > 10 ** (-65 / 20)
    return {"samples": len(pcm), "nonzero_samples": int(np.count_nonzero(pcm)),
            "peak": float(np.max(np.abs(values))) if len(values) else 0., "rms": rms,
            "rms_dbfs": 20 * math.log10(rms) if rms else None,
            "active_fraction": float(np.mean(active)) if len(active) else 0.,
            "active_seconds": float(np.count_nonzero(active) * .08)}, active


def check_room(directory, record, errors):
    prefix = record["id"]

    def require(condition, message):
        if not condition:
            errors.append(f"{prefix}: {message}")

    summary = read_json(directory / "summary.json")
    require(summary.get("completed") is True, "summary is not completed")
    require(summary.get("sample_rate") == RATE, "summary sample rate mismatch")
    frames = summary.get("frames", 0)
    require(isinstance(frames, int) and frames > 0, "invalid frame count")
    if not isinstance(frames, int) or frames <= 0:
        return {"error": "invalid frame count"}
    state = summary["final_state"]
    final_agents = {a["id"]: a for a in state["agents"]}
    require(set(final_agents) == {"persona", "moshi", "custom"}, "unexpected scene agent IDs")
    result = {"completed": summary.get("completed"), "frames": frames,
              "seconds": summary.get("seconds"), "wall_seconds": summary.get("wall_seconds"),
              "wall_per_audio_second": summary.get("wall_per_audio_second"),
              "wall_start_utc": summary.get("wall_start_utc"), "wall_end_utc": summary.get("wall_end_utc"),
              "summary_deadline_lateness_p95_ms": summary.get("deadline_lateness_p95_ms"),
              "performance": state.get("performance", {}), "agents": {}}
    trace = []
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            trace.append(json.loads(line))
    require(len(trace) == frames, "trace frame count mismatch")
    require([t.get("tick") for t in trace] == list(range(1, frames + 1)), "trace ticks are not contiguous")
    require(all({a["id"] for a in t["agents"]} == {"persona", "moshi", "custom"} for t in trace),
            "unexpected trace agent IDs")
    late = [t["evaluation"]["deadline_lateness_ms"] for t in trace]
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in late), "non-finite lateness")
    late = [x for x in late if isinstance(x, (int, float)) and math.isfinite(x)]
    result["trace_deadline_lateness_p95_ms"] = float(np.percentile(late, 95)) if late else None
    result["trace_step_p95_ms"] = float(np.percentile([t["performance"]["step_ms"] for t in trace], 95)) if trace else None
    result["trace_deadline_overruns"] = sum(x > 80 for x in late)
    expected_links = {("persona", "moshi"), ("moshi", "persona")}
    links_present = all(expected_links <= {(x["source"], x["target"]) for x in t["links"]} for t in trace)
    require(links_present, "a peer acoustic path is missing from engine history")
    result["peer_paths_present_every_tick"] = links_present
    require(all(x["source"] != x["target"] for t in trace for x in t["links"]), "unexpected own-source path")
    require(all(a.get("muted") for t in trace for a in t["agents"] if a["id"] == "custom"),
            "custom source was not muted throughout")
    custom = read_pcm(directory / "custom-emitted.wav")
    result["custom_emission_silent"] = not bool(np.any(custom))
    require(len(custom) == frames * FRAME and result["custom_emission_silent"], "custom emission is nonzero or wrong length")
    active_masks = []
    for key in AGENTS:
        emitted = read_pcm(directory / f"{key}-emitted.wav")
        received = read_pcm(directory / f"{key}-received.wav")
        inputs = read_pcm(directory / f"{key}-input.wav")
        correct_lengths = all(len(x) == frames * FRAME for x in (emitted, received, inputs))
        require(correct_lengths, f"{key} WAV length mismatch")
        silence = len(inputs) >= FRAME and not bool(np.any(inputs[:FRAME]))
        aligned = correct_lengths and np.array_equal(inputs[FRAME:], received[:-FRAME])
        require(silence and aligned, f"{key} input/previous-receiver alignment failed")
        source_stats, active = audio_stats(emitted)
        receiver_stats, _ = audio_stats(received)
        active_masks.append(active)
        worker = final_agents[key].get("worker", {})
        before = summary.get("pre_barrier_worker_stats", {}).get(key, {})
        startup = summary.get("startup_worker_stats", {}).get(key, {})
        events = []
        previous = int(startup.get("underrun_frames", 0))
        for t in trace:
            row = next(a for a in t["agents"] if a["id"] == key)
            total = int(row.get("worker", {}).get("underrun_frames", 0))
            if total > previous:
                events.append({"tick": t["tick"], "seconds": t["time"], "frames": total - previous})
            previous = total
        result["agents"][key] = {
            "initial_input_silent": silence, "input_equals_previous_received": aligned,
            "emitted": source_stats, "received": receiver_stats, "worker_final": worker,
            "pre_barrier_worker_stats": before, "startup_worker_stats": startup,
            "counter_delta_since_start": {k: worker.get(k, 0) - startup.get(k, 0) for k in COUNTERS},
            "underrun_events": events,
            "underruns_within_first_second": sum(e["frames"] for e in events if e["seconds"] <= 1),
            "underruns_after_first_second": sum(e["frames"] for e in events if e["seconds"] > 1),
            "native_text": final_agents[key].get("transcript", ""),
            "native_text_kind": final_agents[key].get("transcript_kind")}
    if len(active_masks[0]) == len(active_masks[1]):
        result["both_emissions_active_seconds"] = float(np.count_nonzero(active_masks[0] & active_masks[1]) * .08)
    result["both_agents_emitted_nonzero_audio"] = all(result["agents"][k]["emitted"]["nonzero_samples"] > 0 for k in AGENTS)
    asr_path = directory / "asr.json"
    if asr_path.is_file():
        asr = read_json(asr_path)
        result["asr"] = {"kind": asr.get("kind"), "model": asr.get("model"),
                         "files": {name: {"text": item.get("text", ""), "segment_count": len(item.get("segments", []))}
                                   for name, item in asr.get("files", {}).items()}}
    else:
        result["asr"] = None
    result["audio_health_pass"] = all(
        not result["agents"][k]["worker_final"].get(c, 0)
        for k in AGENTS for c in ("input_overflow_frames", "output_dropped_samples", "decoded_clipped_samples"))
    result["audio_health_pass"] &= not bool(result["performance"].get("overruns", 0))
    result["audio_health_pass"] &= not any(result["performance"].get("adc_clipped_samples", {}).values())
    return result


def verify_matrix(root):
    root = Path(root).resolve()
    errors = []
    result = {"schema_version": 1, "kind": "matrix_artifact_verification", "rooms": {},
              "interpretation": "Structural/routing checks and energy activity only; no automatic conversation-quality or causal-comprehension claim.",
              "errors": errors}
    try:
        matrix = read_json(root / "matrix.json")
        manifest = read_json(root / "manifest.json")
        rooms = matrix["rooms"]
        workers = manifest["workers"]
        if len(rooms) != 9 or len({r["id"] for r in rooms}) != 9:
            errors.append("Expected nine distinct matrix rooms")
        if {r["id"] for r in rooms} != {r["id"] for r in manifest["rooms"]}:
            errors.append("Matrix and manifest room IDs disagree")
        families = manifest["family_order"]
        pair_set = {(r["left"], r["right"]) for r in manifest["rooms"]}
        if len(families) != 3 or pair_set != {(a, b) for a in families for b in families}:
            errors.append("Manifest does not contain all nine ordered model pairs")
        if matrix.get("completed") is not True or any(r.get("status") != "completed" for r in rooms):
            errors.append("Not all nine rooms completed")
        if len(workers) != 18 or len({w["id"] for w in workers}) != 18:
            errors.append("Expected eighteen distinct worker records")
        pids = [w.get("pid") for w in workers]
        if any(not isinstance(pid, int) or pid <= 0 for pid in pids) or len(set(pids)) != 18:
            errors.append("Worker PIDs are missing or not distinct")
        endpoints = [a["url"] for r in manifest["rooms"] for a in r["agents"]]
        if len(endpoints) != 18 or len(set(endpoints)) != 18:
            errors.append("Expected eighteen distinct worker endpoints")
        gpu_count = manifest["configuration"]["gpu_count"]
        by_gpu = {}
        for w in workers:
            gpu = w.get("gpu")
            if not isinstance(gpu, int) or not 0 <= gpu < gpu_count:
                errors.append(f"Invalid GPU assignment for {w['id']}")
            by_gpu.setdefault(str(gpu), []).append({k: w.get(k) for k in ("id", "room", "family", "side", "pid", "port")})
        for room in manifest["rooms"]:
            members = [w for w in workers if w["room"] == room["id"]]
            if len(members) != 2 or {w["side"] for w in members} != {"left", "right"}:
                errors.append(f"{room['id']}: expected two independent worker sides")
            if {w["port"] for w in members} != {urlsplit(a["url"]).port for a in room["agents"]}:
                errors.append(f"{room['id']}: endpoint/worker port mismatch")
        result["worker_isolation"] = {"worker_count": len(workers), "distinct_pid_count": len(set(pids)),
                                       "distinct_endpoint_count": len(set(endpoints)), "gpu_collocation": by_gpu,
                                       "scope": "Recorded process/endpoint metadata, not OS-level isolation proof"}
        for room in rooms:
            directory = (root / room["directory"]).resolve()
            if not directory.is_relative_to(root):
                errors.append(f"{room['id']}: directory escapes matrix root")
                continue
            try:
                result["rooms"][room["id"]] = check_room(directory, room, errors)
            except (OSError, ValueError, KeyError, TypeError, StopIteration, wave.Error) as exc:
                errors.append(f"{room['id']}: {type(exc).__name__}: {exc}")
                result["rooms"][room["id"]] = {"error": str(exc)}
        starts = [r.get("wall_start_utc") for r in result["rooms"].values()]
        ends = [r.get("wall_end_utc") for r in result["rooms"].values()]
        valid = len(starts) == 9 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in starts + ends)
        if valid:
            overlap = min(ends) - max(starts)
            spread_ms = (max(starts) - min(starts)) * 1000
            result["simultaneity"] = {"start_spread_ms": spread_ms, "common_wall_interval_seconds": max(0, overlap),
                                      "common_start_epoch": max(starts), "common_end_epoch": min(ends),
                                      "starts_within_one_80ms_tick": spread_ms <= 80}
            if overlap <= 0 or any(b <= a for a, b in zip(starts, ends)):
                errors.append("Nine recording wall intervals do not overlap")
        else:
            errors.append("Missing finite wall-clock epochs for all nine rooms")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Matrix metadata: {type(exc).__name__}: {exc}")
    result["structural_pass"] = not errors
    result["audio_health_pass"] = len(result["rooms"]) == 9 and all(r.get("audio_health_pass", False) for r in result["rooms"].values())
    result["all_eighteen_agents_emitted_audio"] = len(result["rooms"]) == 9 and all(r.get("both_agents_emitted_nonzero_audio", False) for r in result["rooms"].values())
    output = root / "verification.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    result = verify_matrix(args.matrix)
    print(json.dumps({k: result[k] for k in ("structural_pass", "audio_health_pass", "all_eighteen_agents_emitted_audio", "errors")}, indent=2))
    raise SystemExit(0 if result["structural_pass"] else 1)


if __name__ == "__main__":
    main()
