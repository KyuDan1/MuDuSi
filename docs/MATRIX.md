# Nine isolated conversations

The matrix experiment runs every ordered pair of Moshi, PersonaPlex, and Raon concurrently. Rows identify the left model and columns identify the right model, in that order. Moshi–PersonaPlex and PersonaPlex–Moshi are distinct rooms with separate model instances and left/right conditioning.

| Left / right | Moshi | PersonaPlex | Raon |
|---|---|---|---|
| Moshi | `moshi-moshi` | `moshi-personaplex` | `moshi-raon` |
| PersonaPlex | `personaplex-moshi` | `personaplex-personaplex` | `personaplex-raon` |
| Raon | `raon-moshi` | `raon-personaplex` | `raon-raon` |

There are **nine acoustic engines and eighteen model processes**, with distinct endpoints, adapters, and model state. Every room has two active model slots (`persona` = left, `moshi` = right); `custom` is muted. These slot names do not determine the backend. No room sends audio to another room, and no prerecorded stimulus is inserted.

This document describes the runner and verification procedure. The example uses **0.5× wall-clock pacing** to give colocated models more compute time. Successful GPU inference and usable conversations must be established from the recorded run; this setting is **not a real-time throughput benchmark**.

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

Review the configuration and inspect the plan without launching models:

```bash
.venv/bin/python -m workers.run_matrix \
  --config configs/matrix.example.json --output runs/matrix-run --plan
```

The example allocates one worker of each family to each of six job-local GPUs. It uses ports 18000–18017, starts models in batches of four, and defaults to 60 simulation seconds per room at `playback_rate: 0.5` (120 wall-clock seconds after readiness). Raon uses a static memory fraction of 0.30, with speak-first conditioning on the left and listen-first conditioning on the right. `raon.mimi_sdpa_window: true` explicitly enables the optional Mimi SDPA window bridge; it is not silently applied to other launchers. Inspect its startup log and validate generated audio when using this compatibility path.

```bash
mkdir -p runs
export MUDUSI_ENV_FILE="$PWD/work/server.env"
sbatch workers/matrix.sbatch \
  --config configs/matrix.example.json --output runs/matrix-run
```

The batch requests **6 GPUs, 48 CPUs, 512 GiB RAM, and 35 minutes**, using `extra` QOS. On this cluster `extra` is preemptible, including by eligible `own` jobs; a cancelled matrix is an incomplete experiment. Use a fresh output directory to rerun. `--seconds 45` can override the simulation/audio duration (90 wall-clock seconds at 0.5×). `--playback-rate 1` selects real-time input pacing when explicitly testing that setting. GPU assignments are explicit in the configuration; `--gpus` only changes the expected allocation size and does not redistribute workers.

The runner checks ports before launching, waits for all surviving rooms' adapter readiness, then releases an in-process start barrier. Startup conditioning may produce audio before release; evaluation summaries preserve the startup counters. A room that fails readiness does not strand the others. The overall exit status is nonzero unless all nine rooms finish. Startup/cleanup are bounded, and cancellation terminates the runner's own worker process groups. There is no worker attach, external control channel, or NAS-based live coordination.

## Simulation time versus wall-clock pacing

`playback_rate` is simulation seconds per wall-clock second and must be in `(0, 1]`. With the example's rate of 0.5, a 60-second, 750-frame conversation runs for about 120 wall-clock seconds, excluding startup and cleanup. All nine rooms still run concurrently behind the same readiness barrier. Model instances, room separation, and the three-models-per-GPU placement remain unchanged.

The audio format stays **24 kHz mono**, with **1,920 samples per 80 ms simulation tick**. Slower input pacing does not resample audio, stretch voices, or add a silence block between recorded frames. Replay and exported videos use the normal audio/simulation timeline. A 60-second video therefore presents 60 seconds of generated audio, while the inference job took about 120 seconds to collect it. Keep that distinction visible in published timing claims.

The manifest and `matrix.json` record `playback_rate`, `simulation_seconds`/`seconds`, and `target_wall_seconds`; room summaries record actual wall time. At 0.5×, a wall/audio ratio near 2 is intentional. Compare startup and sustained buffer counters separately and review the resulting audio: relaxed pacing alone does not establish good dialogue or remove every underrun.

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
