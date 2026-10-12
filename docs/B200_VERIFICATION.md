# B200 execution record — 2026-10-12

Actual model audio passed through the spatial mixer into the other model's next
input. Two 60-second PersonaPlex–Moshi conversations completed, with source,
receiver and adapter-input WAVs, JSONL traces, offline receiver ASR, and Chromium
3D playback checks. These are inference results, not SLM training results.

## Environment and allocation

The login node exposed 8 NVIDIA B200 GPUs, 183,359 MiB each, driver 580.95.05
(driver CUDA ceiling 13.0). Existing users' processes were left untouched.
One bounded PersonaPlex correctness probe used login GPU 2 for 80 audio frames
and then released it. Full conversations and timing measurements used Slurm.

Each conversation requested `sub-own`, 2 GPUs, 16 CPUs, 96 GiB RAM and 30 minutes,
using `nvcr.io/nvidia/pytorch:25.04-py3` for runtime compilation. Model processes
used job-local GPU 0 (Moshi) and GPU 1 (PersonaPlex). The WAV source, mixer and
transport ran on CPU. Native ports 8998/8999 and stimulus port 9001 stayed inside
the isolated job; the launchers bind loopback. Queuing and cold image startup
are separate from the reported inference durations.

| Component | Runtime / source |
| --- | --- |
| Arena | Python 3.13.12, separate `.venv`, sphn 0.2.1 |
| Viewer | Node 24.16.0; TypeScript/Vite production build |
| Moshi | Separate Python 3.11 venv, Torch 2.8.0+cu128; `pip check` clean |
| PersonaPlex | Existing AudioWorld Python 3.11 environment, Torch 2.8.0+cu128 |
| Moshi source | `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` |
| PersonaPlex source | `3428dfd95309a7f3c84fd93259ded0f810d1ff91`, existing `read_pcm() is None` guard |
| Moshi checkpoint | `kyutai/moshiko-pytorch-bf16`, revision `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` |
| PersonaPlex checkpoint | `nvidia/personaplex-7b-v1`, revision `fdaf4090a61cb315c138a1faee287ffd6c716309` |
| Raon | Torch 2.9.1+cu128; source `afec41185e27653aa905d4ad28a7b8497681275e`, existing AudioWorld checkpoint and SGLang bundle |
| Receiver ASR | OpenAI Whisper `base`, offline CPU batch; distinct from generated text |

PersonaPlex's upstream metadata pins Torch below 2.5. The reused environment
also has unrelated vLLM/xformers dependency conflicts; `pip check` is **not**
clean there. This record validates the particular PersonaPlex inference path,
not every package in that environment. Moshi and PersonaPlex never share a
Python environment. Resolved package lists are saved under ignored
`runs/environment/`. Weights and credentials are not tracked.

## Measured conversations

The stimulus is upstream PersonaPlex `assets/test/input_assistant.wav`, converted
to 24 kHz mono PCM16. It contains two actual spoken questions about cooking rice
and adding garnish, with silence between them. The third character emits this
recording, not model-generated speech. Its active phrases occur approximately
at 4.08–8.56 s and 22.64–26.88 s in these traces.

| Measurement | Near / quiet | Distant / cocktail |
| --- | ---: | ---: |
| Slurm job | 189794 | 189798 |
| Artifact directory | `runs/pair-near` | `runs/pair-cocktail` |
| Audio duration / frames | 60 s / 750 | 60 s / 750 |
| Measured wall time | 60.011 s | 60.007 s |
| Tick lateness p95 | 1.45 ms | 2.04 ms |
| Tick deadline overruns | 0 | 0 |
| Input overflow / dropped output samples | 0 / 0 | 0 / 0 |
| PersonaPlex / Moshi underrun frames | 1 / 2 | 1 / 2 |
| Moshi server frame processing mean / p95 | 24.59 / 26.15 ms | 24.19 / 26.20 ms |
| Both actual models speaking | 3.12 s | 6.56 s |
| Engine overlap ratio, includes stimulus | 0.1301 | 0.2692 |
| Model separation | 1.00 m | 5.70 m |
| Direct model-to-model pressure gain | 0 dB | −29.12 dB |
| Divider | absent | 14 dB loss |
| Microphone noise | −72 dBFS | −35 dBFS |

Underruns occurred at startup before generated samples arrived; buffers were not
prefilled. Final queues held one 80 ms block, which lies beyond the fixed episode
boundary. No ADC or Opus-decoder clipping was counted. During the near run,
`sjob --window 1m --json` reported about 17,839 and 19,625 MiB VRAM on the two job
GPUs, about 36.4 GiB host working memory, no OOM events and no CPU throttling.
These are monitoring-window readings, not exact allocator peaks. Moshi's server
timings measure its frame handler, not end-to-end conversation latency.

For every receiver in both runs, `input[t]` exactly matches the PCM16 encoding
of `received[t-1]`, with an initial silent block. Re-rendering recorded source
WAVs through the mixer reproduces received audio within 9.2e-5 normalized
amplitude in the near run and 4.5e-5 in the cocktail run (source/WAV quantization).
The checks are saved as `routing-verification.json`.

Whisper recognizes Moshi's cooking answer in PersonaPlex's received WAV and
PersonaPlex's introduction/backchannels in Moshi's received WAV. Native model
text remains separate. The second stimulus starts while Moshi is speaking;
overlap and a subsequent stop/resume are visible in WAVs and traces. This proves
an interruption stimulus and response activity, not a causal benchmark of
intentional yielding. The noisy receiver ASR contains hallucinated repetitions;
it must not be treated as ground truth. Different acoustic conditions also
produce different stochastic model trajectories, so this is not an isolated
estimate of each acoustic parameter's semantic effect.

## Re-execution

On the validated server, `work/server.env`, `configs/local-workers.json`,
`configs/local-near.json`, `configs/local-raon-sglang.json`, and
`work/stimulus.wav` contain the local paths/configuration. All are ignored.
From the repository root:

```bash
source work/server.env
sbatch workers/pair.sbatch \
  --output runs/new-near --stimulus work/stimulus.wav \
  --models configs/local-workers.json --configuration configs/local-near.json \
  --scenario quiet --seconds 60

# A separate experiment / allocation for changed acoustics:
sbatch workers/pair.sbatch \
  --output runs/new-cocktail --stimulus work/stimulus.wav \
  --models configs/local-workers.json --scenario cocktail --seconds 60

# After a conversation finishes; the local Whisper checkpoint avoids worker downloads:
sbatch workers/asr.sbatch runs/new-near --model work/asr/base.pt
```

For a new server, install the arena as described in `GPU_SERVER.md`, create the
Moshi environment with Torch 2.8.0/CUDA 12.8 and the pinned source above, and set
`MOSHI_PYTHON` and `PERSONAPLEX_PYTHON` to separate working environments. Copy
`configs/workers.example.json` to an ignored local file, increasing native
`startup_timeout` to 600 if needed. Near geometry uses PersonaPlex at
`[-0.5,1.2,0]`, Moshi at `[0.5,1.2,0]`, and stimulus at `[0,1.2,0.8]`, with
`noise_db=-72`, `reverb=0.06`. Download models on login before submission. No new
license approval or authentication was required on the validated server.

Run the viewer in a foreground login terminal, with no GPU allocation:

```bash
.venv/bin/mudusi serve --replay runs/pair-near --host 127.0.0.1 --port 18766
# Separate terminal for comparison:
.venv/bin/mudusi serve --replay runs/pair-cocktail --host 127.0.0.1 --port 18767
```

From the client machine:

```bash
ssh -N -L 18766:127.0.0.1:18766 -L 18767:127.0.0.1:18767 USER@LOGIN_HOST
```

Open `http://127.0.0.1:18766` or `http://127.0.0.1:18767`, press Run, select an
agent, and enable Hear receiver. Replay shows recorded telemetry and actual
receiver audio; it does not launch a new live GPU conversation. Acoustic edits
are rejected in replay. Both scenes were loaded in headless Chromium with a
WebGL canvas and no page exceptions; screenshots are saved in their run folders.

## Raon and remaining boundaries

The direct Raon PyTorch bridge generated real audio/text, but averaged 207 ms per
80 ms frame and filled its 32-frame input queue after 4.24 simulated seconds.
Job 189796 correctly failed with a partial trace instead of dropping inputs.
Peak PyTorch allocation reported by that worker was about 19.4 GiB. This path
is available for diagnostics and is not certified for real-time operation.

The optimized SGLang attempt required CUDA 13 runtime for FlashInfer (installed
in a separate environment overlay, pinned to 13.0.96). Its first startup failed
because SGLang searched Conda `lib64` for `libcudart`, which is actually in `lib`.
`launch_raon.sh` now exports `LIBRARY_PATH` as well as `LD_LIBRARY_PATH` and uses
the environment's CUDA toolkit. After this fix, job **189820** completed a real
60-second Raon–PersonaPlex conversation (`runs/raon-sglang-v2`): 750 frames in
60.012 wall seconds, p95 tick lateness 1.68 ms, zero input overflow/output drops,
and zero clock overruns. Raon generated speech and text about the rice stimulus.
Raon had 6 startup underrun frames (480 ms, within the first seven ticks) and 480 ms of output still queued at episode
end; PersonaPlex had one underrun frame. SGLang's preallocated KV cache makes its
VRAM footprint larger: the inference monitoring window reported approximately
90,085 MiB for Raon and 19,697 MiB for PersonaPlex, with no CPU throttling/OOM.
The native audio-module warmup emitted a mask/embedding count mismatch and
continued via its fallback, so this is not a clean full-compilation result.

```bash
source work/server.env
sbatch workers/pair.sbatch --output runs/new-raon \
  --stimulus work/stimulus.wav --models configs/local-raon-sglang.json \
  --configuration configs/local-near.json --scenario quiet --seconds 60 \
  --raon-model "$RAON_BUNDLE" --raon-source "$RAON_SOURCE"
```

For a fresh overlay, use the existing Raon environment's Python to create
`.venvs/raon` with `--system-site-packages`, then install only
`nvidia-cuda-runtime==13.0.96` from `https://pypi.nvidia.com` into the overlay.
Set `RAON_PYTHON` to that overlay's Python. The existing Raon environment and
checkpoint remain unchanged. The bundle must already contain `text_model` and
`raon_runtime`; this launcher does not export or retrain a checkpoint.

The Raon checkpoint loader warns that its SDPA Mimi fallback lacks the intended
sliding window when FlashAttention is absent; long-audio fidelity remains a
separate issue from transport correctness. Offline ASR of Raon's emitted audio
diverges substantially from generated text later in the episode, including
garbled phrases. This is observed degradation, not a clean intelligibility pass;
the missing attention implementation and failed warmup are candidate causes,
not established causal diagnoses. Correct those upstream paths and rerun a
long-audio reference before using Raon for quality comparisons. No full-weight
SLM RL is claimed.
See [required learner interfaces and follow-up gates](SLM_TRAINING.md).

## Final repository checks

68 Python tests passed, including native Raon framing, live WAV/input alignment,
partial-failure export, receiver-ASR replay timing and rejection of replay edits.
The viewer passed TypeScript/Vite build. A pre-existing Vite bundle-size warning
and a Starlette/httpx deprecation warning remain non-fatal. All GPU experiments
ended and released their allocations. Runtime files, screenshots, environments,
local settings and checkpoints remain untracked.
