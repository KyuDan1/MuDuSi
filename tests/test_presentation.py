import io
import json
import wave

import numpy as np
import pytest

from duplex_arena.presentation import load_presentation


def recording(tmp_path):
    agents = [dict(id='persona', worker={'ready': True}),
              dict(id='moshi', worker={'ready': True}),
              dict(id='custom', worker={'ready': True, 'worker_name': 'wav-stimulus'})]
    (tmp_path / 'trace.jsonl').write_text(json.dumps({'agents': agents}) + '\n')
    (tmp_path / 'summary.json').write_text(json.dumps({'completed': True}))
    for key, value in [('persona', 10000), ('moshi', 5000), ('custom', 30000)]:
        with wave.open(str(tmp_path / f'{key}-emitted.wav'), 'wb') as f:
            f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            f.writeframes(np.full(1920, value, dtype='<i2').tobytes())
    return tmp_path


@pytest.mark.parametrize("non_model", ["wav-stimulus", "cpu-echo"])
def test_presentation_uses_real_participants_source_asr_and_aligned_mix(tmp_path, non_model):
    root = recording(tmp_path)
    trace = root / 'trace.jsonl'
    trace.write_text(trace.read_text().replace('wav-stimulus', non_model))
    word = {'start': .02, 'end': .06, 'word': 'Hello'}
    (root / 'asr.json').write_text(json.dumps({'files': {
        'persona-emitted.wav': {'segments': [{'words': [word]}]},
        'persona-received.wav': {'segments': [{'words': [{'word': 'Wrong source'}]}]}}}))
    data, audio = load_presentation(root)
    assert data['participants'] == ['persona', 'moshi']
    assert data['words']['persona'] == [word]
    assert data['duration'] == .08
    with wave.open(io.BytesIO(audio)) as f:
        assert f.getnframes() == 1920 and f.getframerate() == 24000
        samples = np.frombuffer(f.readframes(1920), dtype='<i2')
    # WAV stimulus was excluded, and no time-shifting or per-track normalization occurred.
    np.testing.assert_allclose(samples, 15000, atol=1)


def test_presentation_rejects_failed_recording(tmp_path):
    root = recording(tmp_path)
    (root / 'summary.json').write_text('{"completed": false}')
    with pytest.raises(ValueError, match='completed'):
        load_presentation(root)
