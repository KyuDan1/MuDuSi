# Nine isolated conversations

The matrix experiment runs every ordered pair of Moshi, PersonaPlex, and Raon concurrently. Rows identify the left model and columns identify the right model, in that order. Moshi–PersonaPlex and PersonaPlex–Moshi are distinct rooms with separate model instances and left/right conditioning.

| Left / right | Moshi | PersonaPlex | Raon |
|---|---|---|---|
| Moshi | `moshi-moshi` | `moshi-personaplex` | `moshi-raon` |
| PersonaPlex | `personaplex-moshi` | `personaplex-personaplex` | `personaplex-raon` |
| Raon | `raon-moshi` | `raon-personaplex` | `raon-raon` |

There are **nine acoustic engines and eighteen model processes**, with distinct endpoints, adapters, and model state. Every room has two active model slots (`persona` = left, `moshi` = right); `custom` is muted. These slot names do not determine the backend. No room sends audio to another room, and no prerecorded stimulus is inserted.

Use the **measured four-GPU configuration** below to reproduce the verified collection setup. All nine rooms completed with eighteen emitting agents, correct receiver-to-input routing, and no sustained underruns at **0.25× pacing**. See [measured results and limitations](MATRIX_RESULTS.md). This is **not a real-time throughput benchmark** or a conversation-quality score.

## Prepare and submit

Install the arena and independent model environments using [GPU setup](GPU_SERVER.md) and [model protocols](MODELS.md). The Raon source checkout and exported SGLang bundle must already exist. On the shared cluster, installations belong on login and sustained inference belongs in a closed batch job.

Prepare an ignored environment file such as `work/server.env`, exporting:

```bash
export MOSHI_PYTHON=/path/to/moshi/environment/bin/python
export PERSONAPLEX_PYTHON=/path/to/personaplex/environment/bin/python
export RAON_PYTHON=/path/to/raon/environment/bin/python
export RAON_SOURCE=/path/to/Raon-Speech
export RAON_BUNDLE=/path/to/Raon-SpeechChat-9B-sglang
```

## Measured four-GPU configuration

`configs/matrix.four-gpu.json` runs all nine ordered rooms and eighteen independent model processes on four job-local GPUs. GPUs 0–3 hold **5, 5, 4, and 4 processes**, respectively. The family totals are six Moshi, six PersonaPlex, and six Raon instances. Every participant retains its own endpoint and model state.

Inspect the plan without loading models:

```bash
.venv/bin/python -m workers.run_matrix \
  --config configs/matrix.four-gpu.json --output runs/matrix-run --plan
```

For the shared cluster's `sub` account, override the batch defaults to use the protected four-GPU `own` quota:

```bash
mkdir -p runs
export MUDUSI_ENV_FILE="$PWD/work/server.env"
sbatch --qos=own --gres=gpu:4 --cpus-per-gpu=12 --mem=512G \
  workers/matrix.sbatch --config configs/matrix.four-gpu.json \
  --output runs/matrix-run
```

This requests **4 GPUs, 48 CPUs, 512 GiB RAM, and 35 minutes**. The configuration uses ports 18000–18017, startup batches of four, and **0.25× pacing**: 60 simulation/audio seconds take approximately **240 wall-clock seconds** after readiness. The measured run is job 190030; its GPU/timing/buffer results are documented [separately](MATRIX_RESULTS.md). Availability within the user's own quota still determines when a new submission starts.

Raon uses `max_total_tokens=8192`, `max_running_requests=4`, and `max_allocated_req_pool_indices=4`, with `mem_fraction_static=0.85`. The fraction participates in the available-memory budget calculation; the explicit token cap means each process does **not** allocate 85% of the GPU. For a bounded 60-second rollout, the expected generated-token budget is at most 2,250 positions plus conditioning. The estimated capped KV cache is approximately **1.125 GiB per Raon process**, distinct from weights, codecs, and workspaces. Actual combined memory was measured in the [results](MATRIX_RESULTS.md#four-gpu-memory-and-utilization); do not use the KV estimate as total VRAM demand.

Left Raon instances use speak-first conditioning; right instances use listen-first conditioning. `raon.mimi_sdpa_window: true` explicitly enables the Mimi SDPA window bridge in the parent and spawned decoder. Check each decoder-child startup message; the flag alone is not evidence of successful propagation.

The runner checks ports, waits for surviving rooms' adapters, and releases one in-process start barrier. It records pre/post-barrier counters so startup output is visible. A failed room does not strand other rooms; overall success requires all nine to finish. Startup and cleanup are bounded, and cancellation terminates the runner's owned worker process groups. NAS files are output artifacts, not a live control channel.

## Other allocation and pacing choices

`configs/matrix.example.json` retains the **six-GPU, three-models-per-GPU, 0.5×** option. It targets 120 wall-clock seconds for 60 simulation seconds and uses Raon's static memory fraction 0.30. Submit with the unmodified batch defaults:

```bash
sbatch workers/matrix.sbatch \
  --config configs/matrix.example.json --output runs/matrix-six-gpu
```

This requests **6 GPUs, 48 CPUs, 512 GiB RAM, and 35 minutes** with preemptible `extra` QOS. Eligible `own` jobs can preempt it. A pending submission cancelled to change allocation is a scheduling event, not a failed inference run. This six-GPU 0.5× configuration has no completed measured run in the results; the earlier six-GPU baseline ran at **1×** and had sustained Raon underruns.

Always use a new output directory. `--seconds` changes simulation/audio duration; `--playback-rate` changes input pacing. GPU assignments remain explicit in the configuration: `--gpus` only changes the expected allocation size and does not redistribute workers. Revised settings require fresh validation.

## Simulation time versus wall-clock pacing

`playback_rate` is simulation seconds per wall-clock second and must be in `(0, 1]`. At the measured rate of 0.25, a 60-second, 750-frame conversation runs for about 240 wall-clock seconds, excluding startup and cleanup. At 0.5 it takes about 120 seconds. All nine rooms run concurrently behind the same readiness barrier; pacing does not merge model sessions or acoustic rooms.

The audio format stays **24 kHz mono**, with **1,920 samples per 80 ms simulation tick**. Slower input pacing does not resample audio, stretch voices, or add a silence block between recorded frames. Replay and exported videos use the normal audio/simulation timeline. A 60-second video therefore presents 60 seconds of generated audio, while the measured inference job took about 240 seconds to collect it. Keep that distinction visible in published timing claims.

The manifest and `matrix.json` record `playback_rate`, `simulation_seconds`/`seconds`, and `target_wall_seconds`; room summaries record actual wall time. At 0.5×, a wall/audio ratio near 2 is intentional. The four-GPU configuration instead targets a wall/audio ratio near 4. Compare startup and sustained buffer counters separately and review the resulting audio: relaxed pacing alone does not establish good dialogue or remove every underrun.

## Verify the experiment

The root `manifest.json` preserves configuration, model/room order, GPU assignments, process IDs, startup batches, per-room readiness/start/end timestamps, and the measured start spread. `gpu-telemetry.jsonl` records time-stamped GPU and process memory/utilization samples throughout the job. `matrix.json` is the viewer's room manifest and includes completion/concurrency metadata.

Each room contains its worker configuration, logs, `trace.jsonl`, `summary.json`, and emitted/received/input WAVs. Keep these together. Configuration and manifests can include prompts or private paths, so review them before sharing.

Before reporting success, check:

- All nine summaries completed and `simultaneous_room_count` is nine. Compare actual wall-clock start/end times; a synchronized visualization alone does not prove concurrent inference.
- Each receiver's previous PCM16 block matches the next recorded model-input block, with initial silence. Confirm every claimed speaker actually emitted intelligible audio.
- Wall time/audio duration, tick lateness, overruns, startup versus sustained underruns, dropped output, input overflow, and clipping are acceptable for the experiment.
- GPU telemetry and worker logs show no memory failure or unintended process sharing. Sharing a GPU among three processes is intentional; reusing a model session is not.

Run the artifact verifier to write `verification.json` with structural checks,
per-agent audio activity, buffer counters, and timing summaries:

```bash
.venv/bin/python -m workers.verify_matrix --matrix runs/matrix-run
```

Its exit status covers recording structure and routing integrity. Audio-health
flags and intelligibility need separate review; a passing exit status is not a
conversation-quality score.

Generated text, received-audio ASR, and emitted-audio captions remain separate evidence streams. A healthy connection, finite waveform, or conversational-looking transcript is not by itself a speech-quality result.

## Add captions after inference

After the GPU matrix has finished, use the [showcase ASR setup](SHOWCASE.md) to generate `asr.json` for each completed room. For example:

```bash
export ASR_PYTHON=/path/to/whisper/environment/bin/python
sbatch --time=00:20:00 workers/asr_gpu.sbatch --matrix runs/matrix-run
```

This loads Whisper once and processes emitted and received audio from all completed rooms. Do not launch competing `own` ASR jobs while relying on a preemptible inference job to finish. Captions are optional for replay and use offline, word-timed ASR; they are not live streaming-recognition measurements.

## Replay and render

Build the viewer, then serve the finished matrix on CPU:

```bash
npm --prefix web run build
mudusi serve --matrix runs/matrix-run --host 127.0.0.1 --port 18766
```

Open `http://127.0.0.1:18766/?matrix`. From a workstation:

```bash
ssh -N -L 18766:127.0.0.1:18766 YOUR_SSH_LOGIN_ALIAS
```

The grid presents nine rooms on a common recorded simulation/audio timeline; its clock is not the elapsed GPU-job wall time. Select a room to hear its audio. Presentation audio mixes the two emissions within the selected room; independent rooms are never mixed together or fed back into one another. Per-agent receiver WAVs remain the evidence of what each model heard.

Install the Chromium/Playwright/FFmpeg dependencies described in [Showcase](SHOWCASE.md), then render a bounded CPU batch:

```bash
export ARENA_NODE=/path/to/node
sbatch workers/render_matrix.sbatch --matrix runs/matrix-run \
  --output output/mudusi-nine-rooms.mp4 --start 0 --seconds 60 --fps 24 \
  --focus-seconds 6
```

The default soundtrack moves its focus among rooms every six seconds while all rooms retain their original recording time. To keep the soundtrack on one room, add `--audio-room moshi-personaplex`. This selects an existing recording; it does not rerun, combine, or reorder the inference experiments. Inspect the resulting video and provenance sidecar before publishing it.

### Audio-focus order

`--focus-order` changes the order of audio spotlighting, not the grid layout, inference order, or recording timestamps. Supply a comma-separated permutation containing **each of the nine room IDs exactly once**. The default is row-major order (Moshi row, PersonaPlex row, then Raon row). For example, put the three same-family pairs first:

```bash
sbatch workers/render_matrix.sbatch --matrix runs/matrix-run \
  --output output/mudusi-nine-rooms.mp4 --start 0 --seconds 60 --fps 24 \
  --focus-seconds 6 \
  --focus-order moshi-moshi,personaplex-personaplex,raon-raon,moshi-personaplex,personaplex-moshi,moshi-raon,raon-moshi,personaplex-raon,raon-personaplex
```

Focus is selected from the **absolute simulation/audio time**: with six-second windows, room 1 plays at 0–6 seconds, room 2 at 6–12 seconds, and so on, cycling after 54 seconds. A nonzero `--start` enters the corresponding existing window rather than restarting the focus order. Every selected room is heard at that same time in its own recording; no conversations are concatenated or time-shifted. Boundary ramps prevent clicks without mixing independent rooms. `--audio-room ROOM_ID` overrides cycling with a fixed room. The output sidecar records the focus settings.

To reproduce the 54-second overview selected for job 190043 from the measured four-GPU recording:

```bash
sbatch workers/render_matrix.sbatch --matrix runs/matrix-nine-four-gpu-20261012 \
  --output output/mudusi-nine-rooms-overview.mp4 --start 0 --seconds 54 --fps 24 \
  --focus-seconds 6 \
  --focus-order moshi-raon,moshi-personaplex,raon-moshi,personaplex-moshi,raon-raon,moshi-moshi,raon-personaplex,personaplex-raon,personaplex-personaplex
```

Each room receives one six-second audio spotlight at its original common time. This presentation includes operationally healthy recordings with differing speech quality; consult the [ASR-based review](MATRIX_RESULTS.md#asr-based-review-and-unresolved-speech-quality) rather than treating the grid as nine dialogue-quality passes. Validate the rendered audio/video before publication.
