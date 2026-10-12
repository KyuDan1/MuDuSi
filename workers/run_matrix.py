"""One closed batch experiment: nine isolated ordered pairs, concurrently.

Every participant owns a process, endpoint, adapter and model state. Rooms share
neither audio nor a Session. Readiness synchronization is an in-process Event;
NAS files are output artifacts only, never a control interface.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

FAMILIES = ("moshi", "personaplex", "raon")
LABELS = {"moshi": "Moshi", "personaplex": "PersonaPlex", "raon": "Raon"}


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "epoch_seconds": time.time(),
            "monotonic_seconds": time.perf_counter()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_plan(config, output):
    """Pure construction/validation, also used by --plan without loading models."""
    families = config.get("families", list(FAMILIES))
    if len(families) != 3 or set(families) != set(FAMILIES):
        raise ValueError("families must contain moshi, personaplex and raon exactly once")
    seconds = config.get("seconds", 60)
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("seconds must be finite and in (0, 3600]")
    port = config.get("port_start", 18000)
    if not isinstance(port, int) or not 1024 <= port <= 65518:
        raise ValueError("port_start must leave room for 18 unprivileged ports")
    batch_size = config.get("startup_batch_size", 2)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 4:
        raise ValueError("startup_batch_size must be in [1, 4]")
    assignment = config["gpu_assignment"]
    gpu_count = config.get("gpu_count", 6)
    if not isinstance(gpu_count, int) or not 1 <= gpu_count <= 8:
        raise ValueError("gpu_count must be in [1, 8]")
    if set(assignment) != set(FAMILIES) or any(len(assignment[k]) != 6 for k in FAMILIES):
        raise ValueError("Expected six GPU assignments for each model family")
    if any(not isinstance(g, int) or not 0 <= g < gpu_count for values in assignment.values() for g in values):
        raise ValueError("GPU assignments exceed the configured job-local gpu_count")
    session_timeout = config.get("session_startup_timeout_seconds", 600)
    if not 0 < session_timeout <= 600:
        raise ValueError("session_startup_timeout_seconds must be in (0, 600]")
    for key, default in (("startup_timeout_seconds", 1200), ("monitor_interval_seconds", 5)):
        value = config.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    gpu_iter = {key: iter(values) for key, values in assignment.items()}
    rooms, workers = [], []
    for index, (left, right) in enumerate(itertools.product(families, repeat=2)):
        room_id = f"{left}-{right}"
        agents, room_workers = [], []
        for side, slot, family in (("left", "persona", left), ("right", "moshi", right)):
            endpoint = "/realtime/ws" if family == "raon" else "/api/chat"
            agent = {"id": slot, "backend": family, "label": f"{LABELS[family]} {side.title()}",
                     "url": f"ws://127.0.0.1:{port}{endpoint}", "startup_timeout": session_timeout}
            if family == "personaplex":
                agent.update(config.get("personaplex", {}).get(side, {}))
            worker = {"id": f"{room_id}-{side}", "room": room_id, "side": side,
                      "family": family, "slot": slot, "port": port,
                      "gpu": next(gpu_iter[family]),
                      "status": "planned"}
            workers.append(worker)
            room_workers.append(worker["id"])
            agents.append(agent)
            port += 1
        rooms.append({"id": room_id, "left": left, "right": right,
                      "output": str(Path(output) / room_id), "agents": agents,
                      "workers": room_workers, "status": "planned"})
    return {"schema_version": 1, "experiment": "simultaneous_ordered_pair_matrix",
            "family_order": families, "seconds": seconds, "room_count": 9,
            "participant_count": 18, "workers": workers, "rooms": rooms,
            "configuration": config, "stimulus": "none; only model emissions and microphone noise",
            "isolation": "distinct process, endpoint and session per participant; distinct acoustic engine per room"}


def worker_command(worker, config=None):
    args = ["bash", f"workers/launch_{worker['family']}.sh", str(worker["gpu"]), str(worker["port"])]
    if worker["family"] == "raon":
        args += [os.environ["RAON_SOURCE"], os.environ["RAON_BUNDLE"],
                 "--mem-fraction-static", "0.30",
                 "--speak-first" if worker["side"] == "left" else "--no-speak-first"]
        if (config or {}).get("raon", {}).get("mimi_sdpa_window", False):
            args.append("--mimi-sdpa-window")
    return args


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


async def monitor(output, stop, interval):
    """Bounded by the experiment's lifetime; subprocesses have short timeouts."""
    with output.open("w") as stream:
        while not stop.is_set():
            for kind, query in (("gpu", "index,uuid,memory.used,memory.total,utilization.gpu"),
                                ("compute-apps", "gpu_uuid,pid,used_memory")):
                process = None
                try:
                    process = await asyncio.create_subprocess_exec(
                        "nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    out, err = await asyncio.wait_for(process.communicate(), timeout=8)
                    row = {**stamp(), "kind": kind, "returncode": process.returncode,
                           "csv": out.decode().splitlines(), "stderr": err.decode().strip()}
                except (OSError, asyncio.TimeoutError) as exc:
                    if process is not None and process.returncode is None:
                        process.kill()
                        await process.wait()
                    row = {**stamp(), "kind": kind, "error": str(exc)}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


async def experiment(config, root):
    from duplex_arena.evaluation import evaluate_live

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model matrix runs require a closed Slurm batch job; use --plan for inspection")
    for name in ("RAON_SOURCE", "RAON_BUNDLE", "RAON_PYTHON", "MOSHI_PYTHON", "PERSONAPLEX_PYTHON"):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} in the batch environment or MUDUSI_ENV_FILE")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory must be new or empty; never overwrite a recorded experiment")
    root.mkdir(parents=True, exist_ok=True)
    manifest = make_plan(config, root)
    manifest.update(job_id=os.environ["SLURM_JOB_ID"], started=stamp(), status="starting")
    manifest["arena_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    manifest_path = root / "manifest.json"
    processes, logs = {}, []
    stop_monitor = asyncio.Event()
    monitor_task = asyncio.create_task(monitor(root / "gpu-telemetry.jsonl", stop_monitor,
                                               config.get("monitor_interval_seconds", 5)))
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, main_task.cancel)
    write_json(manifest_path, manifest)
    try:
        # Preflight every port before launching any model; no unrelated process is signalled.
        for worker in manifest["workers"]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", worker["port"]))
        for room in manifest["rooms"]:
            directory = Path(room["output"])
            directory.mkdir()
            write_json(directory / "workers.json", {"agents": room["agents"]})
            write_json(directory / "configuration.json", config["room"])
            write_json(directory / "manifest.json", {"matrix_job": manifest["job_id"],
                "matrix_room": room["id"], "agents": room["agents"], "stimulus": manifest["stimulus"],
                "arena_commit": manifest["arena_commit"], "workers": [w for w in manifest["workers"] if w["room"] == room["id"]]})
        # Initialize Raon first, in small batches; native models follow, also batched.
        boot_order = sorted(manifest["workers"], key=lambda w: w["family"] != "raon")
        manifest["boot_order"] = [w["id"] for w in boot_order]
        batch_size = config.get("startup_batch_size", 2)
        for offset in range(0, len(boot_order), batch_size):
            batch = boot_order[offset:offset + batch_size]
            for worker in batch:
                log = (root / worker["room"] / f"{worker['slot']}.log").open("w")
                logs.append(log)
                try:
                    process = subprocess.Popen(worker_command(worker, config), stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    processes[worker["id"]] = process
                    worker.update(pid=process.pid, status="loading", launch=stamp(), boot_batch=offset // batch_size)
                except OSError as exc:
                    worker.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            deadline = time.monotonic() + config.get("startup_timeout_seconds", 1200)
            while any(w["status"] == "loading" for w in batch):
                for worker in batch:
                    if worker["status"] != "loading":
                        continue
                    process = processes[worker["id"]]
                    if process.poll() is not None:
                        worker.update(status="failed", error=f"Process exited {process.returncode}", failed=stamp())
                    elif time.monotonic() >= deadline:
                        worker.update(status="failed", error="Startup port timeout", failed=stamp())
                        signal_group(process, signal.SIGTERM)
                    else:
                        try:
                            with socket.create_connection(("127.0.0.1", worker["port"]), timeout=.1):
                                worker.update(status="port_ready", port_ready=stamp())
                        except OSError:
                            pass
                write_json(manifest_path, manifest)
                if any(w["status"] == "loading" for w in batch):
                    await asyncio.sleep(1)
            print(json.dumps({"boot_batch": offset // batch_size,
                              "workers": [{"id": w["id"], "status": w["status"]} for w in batch]}), flush=True)
        gate = asyncio.Event()
        room_tasks = []

        async def run_room(room):
            workers = [w for w in manifest["workers"] if w["room"] == room["id"]]
            failed = [w["id"] for w in workers if w["status"] != "port_ready" or processes[w["id"]].poll() is not None]
            if failed:
                room.update(status="failed", error=f"Workers unavailable: {failed}", ended=stamp())
                write_json(Path(room["output"]) / "startup-failure.json", room)
                return
            room.update(status="connecting", connection_started=stamp())

            async def ready():
                room.update(status="ready", adapters_ready=stamp())
                await gate.wait()
                room.update(status="running", evaluation_started=stamp())

            try:
                result = await evaluate_live(Path(room["output"]) / "workers.json", room["output"],
                    seconds=manifest["seconds"], seed=config.get("seed", 42),
                    scenario=config.get("scenario", "quiet"), configuration=config["room"],
                    start_barrier=ready)
                room.update(status="completed", frames=result["frames"], summary="summary.json")
            except Exception as exc:
                room.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                room["ended"] = stamp()
                write_json(manifest_path, manifest)

        for room in manifest["rooms"]:
            room_tasks.append(asyncio.create_task(run_room(room), name=room["id"]))
        # A failed room counts as settled. Its error cannot strand the other eight.
        ready_deadline = time.monotonic() + config.get("session_startup_timeout_seconds", 600) + 30
        while any(room["status"] not in ("ready", "failed") for room in manifest["rooms"]):
            if time.monotonic() >= ready_deadline:
                for room, task in zip(manifest["rooms"], room_tasks):
                    if room["status"] not in ("ready", "failed"):
                        room.update(status="failed", error="Adapter readiness deadline", ended=stamp())
                        task.cancel()
                break
            await asyncio.sleep(.05)
        manifest.update(status="running", barrier_release=stamp(),
                        simultaneous_room_count=sum(r["status"] == "ready" for r in manifest["rooms"]))
        write_json(manifest_path, manifest)
        gate.set()
        await asyncio.gather(*room_tasks, return_exceptions=True)
        starts = [r["evaluation_started"]["epoch_seconds"] for r in manifest["rooms"] if "evaluation_started" in r]
        manifest["start_spread_ms"] = (max(starts) - min(starts)) * 1000 if starts else None
        manifest["completed_rooms"] = sum(r["status"] == "completed" for r in manifest["rooms"])
        manifest["status"] = "completed" if manifest["completed_rooms"] == 9 else "partial_failure"
        return 0 if manifest["status"] == "completed" else 1
    except BaseException as exc:
        manifest.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for task in locals().get("room_tasks", []):
            if not task.done():
                task.cancel()
        await asyncio.gather(*locals().get("room_tasks", []), return_exceptions=True)
        for process in processes.values():
            signal_group(process, signal.SIGTERM)
        deadline = time.monotonic() + 25
        while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
            await asyncio.sleep(.2)
        for process in processes.values():
            signal_group(process, signal.SIGKILL)
            if process.poll() is None:
                process.wait(timeout=5)
        stop_monitor.set()
        await monitor_task
        for log in logs:
            log.close()
        manifest["ended"] = stamp()
        write_json(manifest_path, manifest)
        write_json(root / "matrix.json", {
            "schema_version": 1,
            "model_order": [LABELS[k] for k in manifest["family_order"]],
            "rooms": [{"id": r["id"], "left": LABELS[r["left"]], "right": LABELS[r["right"]],
                       "directory": r["id"], "status": r["status"], "error": r.get("error")}
                      for r in manifest["rooms"]],
            "job_id": manifest["job_id"],
            "wall_start_utc": manifest.get("barrier_release", {}).get("utc"),
            "wall_end_utc": manifest["ended"]["utc"],
            "start_spread_ms": manifest.get("start_spread_ms"),
            "completed": manifest["status"] == "completed",
            "completed_rooms": manifest.get("completed_rooms", 0),
            "simultaneous_room_count": manifest.get("simultaneous_room_count", 0),
        })
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/matrix.example.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, help="Override the configured duration for all rooms")
    parser.add_argument("--gpus", type=int, help="Override the expected allocation size; assignments stay explicit")
    parser.add_argument("--plan", action="store_true", help="Print the validated plan without launching processes")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.seconds is not None:
        config["seconds"] = args.seconds
    if args.gpus is not None:
        config["gpu_count"] = args.gpus
    if args.plan:
        print(json.dumps(make_plan(config, args.output), indent=2))
        return
    raise SystemExit(asyncio.run(experiment(config, Path(args.output))))


if __name__ == "__main__":
    main()
