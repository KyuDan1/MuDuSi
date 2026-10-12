"""Offline receiver ASR, kept separate from native model-generated text.

Requires openai-whisper, soundfile and scipy in an ASR environment. --matrix
reads DIRECTORY/matrix.json and reuses one Whisper model for completed rooms.
"""
import argparse
import json
from pathlib import Path


def recording_directories(root, matrix=False):
    """Select finished matrix rooms without recognizing failed/partial audio."""
    root = Path(root)
    if not matrix:
        return [root]
    manifest = json.loads((root / "matrix.json").read_text())
    directories = []
    seen = set()
    for room in manifest["rooms"]:
        if room.get("status") != "completed":
            print(f"Skipping room {room['id']}: {room.get('status', 'unknown')}", flush=True)
            continue
        directory = (root / room["directory"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Matrix recording directories must stay within the matrix root")
        if directory in seen:
            raise ValueError(f"Duplicate matrix recording: {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        if not summary.get("completed"):
            raise ValueError(f"Matrix marks {room['id']} completed but its summary does not")
        seen.add(directory)
        directories.append(directory)
    if not directories:
        raise ValueError("No completed matrix rooms to transcribe")
    return directories


def save_results(root, results):
    temporary = root / "asr.json.tmp"
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(root / "asr.json")


def transcribe_recording(root, model, model_name, device, agent_ids):
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf

    results = {"kind": "offline_asr", "model": model_name, "files": {}}
    for key in agent_ids:
        for kind in ("received", "emitted"):
            path = root / f"{key}-{kind}.wav"
            audio, sr = sf.read(path, dtype="float32")
            if sr != 24000 or audio.ndim != 1:
                raise ValueError(f"Expected mono 24 kHz recording: {path}")
            if not np.isfinite(audio).all():
                raise ValueError(f"Non-finite recording: {path}")
            if not np.any(np.abs(audio) > 1e-5):
                # Whisper can invent words for silence. Never call it for an
                # empty/zero waveform; leave the ordinary caption schema intact.
                results["files"][path.name] = {"text": "", "segments": []}
            else:
                audio = resample_poly(audio, 2, 3).astype(np.float32)
                result = model.transcribe(audio, language="en", fp16=device != "cpu", temperature=0,
                                          condition_on_previous_text=False, word_timestamps=True)
                results["files"][path.name] = {"text": result["text"], "segments": result["segments"]}
                print(f"{root.name}/{path.name}", result["text"], flush=True)
            save_results(root, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--matrix", action="store_true",
                        help="Transcribe completed rooms in DIRECTORY/matrix.json with one model load")
    args = parser.parse_args()
    directories = recording_directories(args.directory, args.matrix)
    import torch
    import whisper
    torch.set_num_threads(8)
    local_checkpoint = Path("work/asr") / f"{args.model}.pt"
    model = whisper.load_model(str(local_checkpoint) if local_checkpoint.is_file() else args.model, device=args.device)
    # Matrix rooms have exactly two participants; custom is a muted empty slot.
    agent_ids = ("persona", "moshi") if args.matrix else ("persona", "moshi", "custom")
    for root in directories:
        print(f"Transcribing {root}", flush=True)
        transcribe_recording(root, model, args.model, args.device, agent_ids)


if __name__ == "__main__":
    main()
