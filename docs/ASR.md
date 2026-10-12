# Receiver ASR

The optional ASR observer transcribes the **actual microphone mixture at one selected receiver**. It hears propagation delay, attenuation, divider loss, echo, other agents, and microphone noise. It never reads scripted captions or clean speaker channels.

Install the optional backend and run the server with an observer:

```sh
pip install -e '.[asr]'
mudusi serve --asr-model tiny --asr-device cpu --asr-agent moshi
```

Model startup may download weights. CPU uses INT8; CUDA uses FP16. GPU ASR needs a compatible faster-whisper/CTranslate2 CUDA environment and additional GPU memory. The ASR option is off by default. The base simulator does not import its optional dependencies.

The observer uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Its segment generator is consumed in the worker thread so inference does not block the simulation event loop. Actual recognition and model loading have not been benchmarked on the target GPU server; tests inject a backend without downloading weights.

## What the display means

`kind: "scripted"` labels are synthetic demo captions. `kind: "asr"` labels are results returned by the optional recognition backend. The demo produces harmonic speech-like tones, **not intelligible speech**: its script is not expected to appear in actual ASR. Use real model adapters or recorded spoken PCM to exercise recognition. Silence and synthetic audio can still provoke recognition errors or hallucinations; a transcript is an estimate.

ASR is a delayed external observer. It does not show a model's internal understanding, prove that an agent comprehended a phrase, or affect the agents' audio observations or rewards. A receiver mixture may include multiple overlapping speakers. This observer performs no diarization and provides no speaker attribution: `agent_id` identifies the **listening microphone**, not the person who spoke.

## Timing and load

- Input is mono float32 PCM at 24 kHz, copied from `engine.last_inputs[agent_id]`.
- Non-overlapping 5-second windows are resampled to 16 kHz with SciPy's polyphase resampler before recognition.
- Results arrive only after a full window and inference time. `start` and `end` are simulation-time window boundaries; `time` equals `end`, not wall-clock completion time. Words can straddle window boundaries; there is no streaming partial transcript.
- One inference worker runs at a time. Two additional windows can wait. Overload drops the oldest waiting window and increments `dropped_windows` instead of slowing the simulation.
- `stats` exposes readiness, model loading, queued windows, completed windows, buffered duration, discontinuities, and the most recent inference error. A successful later window clears the error.
- Reset clears pending windows and events. An already-running inference is allowed to finish but its generation token prevents the old episode's result from appearing. Close cancels the async worker; Python cannot forcibly stop an already-running native inference thread.

## Python interface

```python
from duplex_arena.asr import ASRObserver

observer = ASRObserver(model_name="tiny", device="cpu", agent_id="moshi")
await observer.start()  # Model loading runs in a worker thread.

# After each engine step, on the same asyncio event loop:
observer.feed(engine.last_inputs[observer.agent_id], engine.time)
for event in observer.drain_events():
    publish(event)
status = observer.stats

await observer.reset()  # Also call this when resetting the simulation.
await observer.close()
```

Events contain `id`, `time`, `start`, `end`, `agent_id`, `text`, `kind: "asr"`, and `type: "asr"`. Empty recognition results produce no event. `feed()` and `drain_events()` are synchronous, quick operations intended for the observer's own event loop.

For tests, inject `transcriber=lambda pcm_24k: "recognized text"`. The callable receives a copied 120,000-sample window and replaces the entire production backend, including resampling. It runs in a thread and must return a string. This permits deterministic tests of receiver audio routing, backpressure, reset races, and error handling without installing SciPy or faster-whisper.
