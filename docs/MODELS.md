# Streaming model integration

The arena runs the acoustic world on CPU. Each model runs in its own process and
receives its own rendered microphone signal. No physical microphone or speaker
is needed. A single simulation step is 1,920 float32 mono samples at 24 kHz
(80 ms). The previous step's received sound becomes the next model input; models
can emit audio while receiving audio.

## Available transports

| Backend | Worker endpoint | Audio transport | Text meaning |
| --- | --- | --- | --- |
| `moshi` | Official PyTorch `moshi.server`, `/api/chat` | Streaming Opus | Generated model text |
| `personaplex` | NVIDIA PersonaPlex `moshi.server`, `/api/chat` | Streaming Opus | Generated model text |
| `pcm` | Your worker, any configured path | Mono float32 little-endian | Your worker's generated text |
| `raon` | Raon official SGLang runtime, `/realtime/ws` | Kind byte + float32 PCM | Generated model text |

Native bridges follow the upstream [Moshi server](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/server.py)
and [PersonaPlex server](https://github.com/NVIDIA/personaplex/blob/main/moshi/moshi/server.py):
the server acknowledges a session with binary byte `0`; audio packets are byte
`1` followed by Opus bytes; text packets are byte `2` followed by UTF-8. Codec
state persists for the whole connection. The arena's optional `sphn` dependency
encodes and decodes Opus on CPU. No Torch dependency is required in the arena.

PersonaPlex receives URL-encoded `voice_prompt` (for example `NATF2.pt`) and
`text_prompt` query parameters. Voice files must exist on its worker. The adapter
does not send unsupported generation parameters. Read [the upstream prompting
guide](https://github.com/NVIDIA/personaplex#prompting-guide) for model-specific roles.

Model text is **not recognition of received audio**. These streams must not be
reported as listener ASR. Enable the [optional ASR observer](ASR.md) on rendered
receiver audio to evaluate recognition. The CPU demo uses synthetic sounds and scripted
captions; the echo worker only repeats audio. Neither is a speech model.

To initiate an otherwise quiet model-to-model conversation, a controlled WAV
source can occupy the `custom` agent. Supply a spoken question as an uncompressed
24 kHz mono PCM16 WAV; the source plays once, then becomes silent:

```bash
python -m workers.wav_worker --wav /path/to/question.wav --port 9001 --delay-seconds 0.8
```

Keep the `custom` endpoint in `configs/workers.example.json` pointing at port
9001. The sound passes through the same geometry, propagation delay, noise and
mixing as any agent's emitted audio. Reset replays the stimulus from the beginning.
`--loop` repeats the file (including the configured initial silence). A stimulus
is prerecorded audio, never represented as generated speech or ASR; its worker
name is `wav-stimulus` in telemetry. The echo worker identifies as `cpu-echo`.

## Configure workers

Copy `configs/workers.example.json` to a local configuration and change endpoints
as needed. Agent IDs are `persona`, `moshi`, and `custom`. Duplicate/unknown IDs,
backends and invalid URLs fail configuration. Missing agents are explicitly
silent in live mode. `custom` is intentionally generic because a third model was
not identified precisely.

```bash
mudusi serve --models configs/workers.example.json
```

The arena connects to all workers concurrently. Startup waits for native model
readiness, including persona conditioning. Set `startup_timeout` per agent
(seconds, default 120, maximum 600). Native reference servers serialize sessions;
allocate a distinct worker process per concurrent native agent. The sample CPU
echo worker can accept several independent connections on one port.

An optional `label` (up to 80 characters) changes the displayed agent name without
changing its stable slot ID. For example, `{"id":"moshi", "backend":"raon",
"label":"Raon", "url":"ws://127.0.0.1:8998/realtime/ws"}` replaces the second
model while retaining the third slot for a WAV stimulus. The official Raon
protocol is not Opus: byte 0 is ready, 1 is PCM, 2 is text, 5/6 are error/close,
and 8 is pong. Use `workers/launch_raon.sh` with an existing source checkout and
exported SGLang bundle. The alternative `workers.raon_worker` directly exercises
stateful PyTorch decoding, but failed real-time pacing on B200; it raises on
queue overflow instead of discarding observations. See the server verification
report before choosing a Raon backend.

`mudusi evaluate --models ...` records all frames at wall-clock 1×, including
each source's emitted WAV, spatial receiver WAV, and the previous receiver block
actually submitted to the adapter (`*-input.wav`, before lossy Opus encoding).
Failures produce a partial summary with `completed: false` and a nonzero exit.
`serve --replay RUN` reads those artifacts, never connects model workers, and
rejects changes to recorded acoustics. Optional `asr.json` from
`workers.transcribe_recording` adds offline receiver ASR separately from generated
text. Stop at episode end; Reset starts playback again.

## Adapter API

```python
from duplex_arena.adapters import build_adapters

adapters = build_adapters("configs/workers.example.json")
agent = adapters["moshi"]
await agent.start()
await agent.push_audio(received_audio)  # (1920,), float32, normalized [-1, 1]
emitted_audio = agent.pull_audio()      # synchronous, never waits for inference
tokens = agent.drain_text()            # synchronous generated text fragments
stats = agent.stats                    # ready, error, counters and queue lengths
await agent.close()
```

All sessions must be closed in a `finally` block in production code. Failures
raise `AdapterError`; a broken worker does not turn into a synthetic model.
`start()` after `close()` creates a fresh model session and resets local buffers.

Send and receive tasks run independently. Missing output samples are zero-filled
and counted in `underrun_samples`/`underrun_frames`. The default output buffer
holds at most 64 blocks (5.12 seconds); overflow discards oldest output and counts
`output_dropped_samples`. The input queue holds 32 blocks; overflow raises
`AdapterBackpressureError` because discarding observations would invalidate
timing. Set `max_input_frames` and `max_output_frames` to tune these limits.
Opus playback clipping is counted as `decoded_clipped_samples`.
`worker_name` contains the custom worker's optional initialization metadata.

Live playback uses wall-clock 1× pacing. Pausing freezes the simulation but does
not roll back a remote model: already-sent input can finish generating output.
Use Reset to start a fresh conversation; pausing is not a reproducible policy
checkpoint. Record queue statistics and overruns alongside every evaluation.

## Custom worker protocol v1

Each WebSocket connection is one independent model session. Client initialization:

```json
{"type":"init","protocol":"duplex-arena.pcm.v1","sample_rate":24000,"channels":1,"frame_size":1920,"format":"f32le"}
```

Server acknowledgment:

```json
{"type":"status","state":"ready"}
```

After acknowledgment, each client binary message contains exactly 7,680 bytes,
representing one 80 ms block of 1,920 normalized little-endian float32 samples.
Server binary messages may contain any positive whole number of float32 samples;
the adapter buffers and splits them into simulation frames. Empty packets, NaNs,
infinities, malformed JSON and unknown events fail the connection.

Optional server events:

```json
{"type":"text","text":"A generated text fragment"}
{"type":"status","state":"listening"}
{"type":"error","message":"Model inference failed"}
```

Allowed status states are `ready`, `listening`, and `speaking`. `text` fragments
are appended without inserting spaces; your tokenizer is responsible for them.
The worker must resample its native output to 24 kHz, maintain inference state
across frames, and run input reception independently from output generation.
Adapt `workers/echo_worker.py` as a protocol reference, replacing its audio gain
operation with the model's streaming inference API.

```bash
python -m workers.echo_worker --port 9001
# In another terminal:
mudusi serve --models configs/echo.example.json
```

## Validation boundary and training

CPU tests cover real PCM WebSocket roundtrips, transport errors, malformed input,
queue overflow, cleanup/restart, and native protocol framing. With `[models]`
installed, a real Opus decode/re-encode roundtrip also checks sample count,
frequency, and amplitude through a local native-protocol test worker. These tests
do not load Moshi or PersonaPlex weights; GPU inference remains a server-side
acceptance test.
An integration test connects three independent CPU echo sessions to the live
simulation and verifies rendered noisy inputs return through the worker path.
Stimulus tests check WAV validation, end padding, silence and session restart.

These adapters provide **inference**, not a speech-model optimizer. Training a
speech model with PPO or another on-policy algorithm additionally needs sampled
token/code actions, log probabilities, policy versions, masks, recurrent/KV
state boundaries and a gradient-capable learner. Native inference WebSockets
do not expose that contract. The arena can supply acoustic observations and
record emitted actions; the learner/rollout transport must be extended for
model-parameter training. Acoustic rewards alone are not semantic conversation
quality or proof that a listener understood speech.

## Model ownership

Weights are downloaded separately by their workers. Moshi's upstream repository
lists CC-BY 4.0 weights and MIT Python code; PersonaPlex lists NVIDIA Open Model
License weights and MIT code. Follow each selected checkpoint's own license,
access rules and attribution requirements. Sources: [Moshi licenses](https://github.com/kyutai-labs/moshi#license),
[PersonaPlex licenses](https://github.com/NVIDIA/personaplex#license).
