"""Offline receiver ASR, kept separate from native model-generated text.

Requires openai-whisper, soundfile and scipy in an ASR environment.
"""
import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import whisper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(args.directory)
    local_checkpoint = Path("work/asr") / f"{args.model}.pt"
    model = whisper.load_model(str(local_checkpoint) if local_checkpoint.is_file() else args.model, device=args.device)
    results = {"kind": "offline_asr", "model": args.model, "files": {}}
    paths = [root / f"{key}-{kind}.wav" for key in ("persona", "moshi", "custom")
             for kind in ("received", "emitted")]
    for path in paths:
        audio, sr = sf.read(path, dtype="float32")
        if sr != 24000 or audio.ndim != 1:
            raise ValueError("Expected mono 24 kHz recording")
        audio = resample_poly(audio, 2, 3).astype(np.float32)
        result = model.transcribe(audio, language="en", fp16=args.device != "cpu", temperature=0,
                                  condition_on_previous_text=False)
        results["files"][path.name] = {"text": result["text"], "segments": result["segments"]}
        print(path.name, result["text"], flush=True)
        (root / "asr.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
