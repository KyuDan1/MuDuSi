"""Read-only presentation data and an explicitly labelled speaker audio mix."""
import io
import json
from pathlib import Path
import wave

import numpy as np


def load_presentation(directory):
    root = Path(directory)
    frames = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['completed']:
        raise ValueError('Presentation requires a completed recording')
    # Only connected model workers are participants. WAV stimuli are not models.
    participants = [a['id'] for a in frames[-1]['agents']
                    if a.get('worker', {}).get('ready')
                    and a.get('worker', {}).get('worker_name') not in ('wav-stimulus', 'cpu-echo')
                    and not a.get('muted')]
    asr_path = root / 'asr.json'
    asr = json.loads(asr_path.read_text()) if asr_path.exists() else {'files': {}}
    words = {}
    tracks = []
    for key in participants:
        words[key] = [word for segment in asr['files'].get(f'{key}-emitted.wav', {}).get('segments', [])
                      for word in segment.get('words', [])]
        with wave.open(str(root / f'{key}-emitted.wav')) as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2):
                raise ValueError('Expected 24 kHz mono PCM16 emitted audio')
            track = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
        if len(track) != len(frames) * 1920:
            raise ValueError('Presentation audio/trace length mismatch')
        tracks.append(track)
    if not tracks:
        raise ValueError('Recording has no model participants')
    mixed = np.sum(tracks, axis=0)
    gain = min(1.0, .92 / max(float(np.max(np.abs(mixed))), 1e-9))
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        wav.writeframes((mixed * gain * 32767).astype('<i2').tobytes())
    return {'frames': frames, 'participants': participants, 'words': words,
            'duration': len(frames) * .08, 'audio_kind': 'emitted speaker mix',
            'caption_kind': 'offline emitted-audio ASR, revealed at word timestamps',
            'mix_gain': gain, 'summary': {k: summary.get(k) for k in (
                'completed', 'wall_per_audio_second', 'deadline_lateness_p95_ms')}}, output.getvalue()
