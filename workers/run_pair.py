"""Supervise a bounded, co-located model conversation inside a batch allocation."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stimulus", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--scenario", default="quiet")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--raon-model", help="Replace Moshi with Raon; config must use PCM for the moshi slot")
    parser.add_argument("--raon-source", help="Use official SGLang runtime from this checkout; config backend must be raon")
    args = parser.parse_args()
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ARENA_OUTPUT_DIR"] = str(root.resolve())
    manifest = {"job_id": os.environ.get("SLURM_JOB_ID"), "scenario": args.scenario,
                "seconds": args.seconds, "gpu_assignment": {"0": "raon" if args.raon_model else "moshi", "1": "personaplex"},
                "arena_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "arena_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    children, logs = [], []
    try:
        first = ["bash", "workers/launch_moshi.sh", "0", "8998"]
        if args.raon_model:
            first = ["env", "CUDA_VISIBLE_DEVICES=0", os.environ["RAON_PYTHON"],
                     "-m", "workers.raon_worker", "--model", args.raon_model, "--port", "8998"]
            if args.raon_source:
                first = ["bash", "workers/launch_raon.sh", "0", "8998", args.raon_source, args.raon_model]
        for name, cmd, port in [
            ("raon" if args.raon_model else "moshi", first, 8998),
            ("persona", ["bash", "workers/launch_personaplex.sh", "1", "8999"], 8999),
            ("stimulus", [sys.executable, "-m", "workers.wav_worker", "--wav", args.stimulus,
                          "--port", "9001", "--delay-seconds", "0.8"], 9001),
        ]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            log = (root / f"{name}.log").open("w")
            logs.append(log)
            children.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), port))
        deadline = time.monotonic() + 1200
        pending = {port for _, port in children}
        while pending:
            for child, port in children:
                if child.poll() is not None:
                    raise RuntimeError(f"Worker on {port} exited ({child.returncode}); inspect logs")
                if port not in pending:
                    continue
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        pending.discard(port)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Workers not listening: {pending}")
            time.sleep(1)
        command = [sys.executable, "-m", "duplex_arena", "evaluate", "--models", args.models,
                   "--seconds", str(args.seconds), "--scenario", args.scenario, "--output", str(root)]
        if args.configuration:
            command += ["--configuration", args.configuration]
        subprocess.run(command, check=True)
    finally:
        for child, _ in children:
            if child.poll() is None:
                child.terminate()
        for child, _ in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
