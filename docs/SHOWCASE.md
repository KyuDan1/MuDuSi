# Recorded conversation showcase

The presentation view plays actual model emissions from a completed evaluation. It
shows a slowly orbiting 3D scene, speech activity lanes, acoustic propagation delay,
TX/RX levels, cumulative buffer underruns, and the recorded scheduler timing.
Speech bubbles use **ASR of each speaker's emitted WAV**, revealed at word
timestamps. These are offline, word-timed ASR captions, not live ASR latency
measurements or native model text. The audio is a constant-gain sum of participant
emissions; receiver WAVs remain available in the engineering replay.

## Generate two or three real participants

Create an ignored `configs/local-conversation.json` with two or three entries from
`configs/workers.example.json`. Supported launch backends are `personaplex` and
`moshi`. Use the fixed IDs `persona`, `moshi`, and (for three speakers) `custom`.
Give each worker a unique loopback port. Two PersonaPlex processes can use distinct
`voice_prompt`, `text_prompt`, and `label` values. Each entry occupies one GPU, in
configuration order. There is no WAV stimulus in this launcher.

Set `PERSONAPLEX_PYTHON` and `MOSHI_PYTHON` to independent environments, then:

```bash
# For two participants override the default three-GPU allocation.
sbatch --gres=gpu:2 --mem=96G workers/conversation.sbatch \
  --models configs/local-conversation.json \
  --configuration configs/local-acoustics.json \
  --seconds 90 --output runs/conversation

# For three participants omit the GPU override.
# After completion, use an environment with openai-whisper, scipy, soundfile:
export ASR_PYTHON=/path/to/asr/environment/bin/python
sbatch workers/asr.sbatch runs/conversation
```

Choose passages after reading the emitted-audio ASR and checking the audio. Do not
call a WAV stimulus a third model. A completed rollout alone does not establish
conversation quality. Keep the original trace, emitted/received/input WAVs,
summary, manifest and ASR together for provenance.

## Open the presentation

```bash
.venv/bin/python -m duplex_arena serve --host 127.0.0.1 --port 18766 \
  --replay runs/conversation
```

Open `http://localhost:18766/?showcase` and press **Play with audio**. The timeline
is seekable. `http://localhost:18766/` opens the engineering view. This serves
finished artifacts and does not connect to GPU workers.

From a workstation, forwarding to the login host running the replay server:

```bash
ssh -N -L 18766:127.0.0.1:18766 YOUR_SSH_LOGIN_ALIAS
```

## Render an MP4

Install Playwright in an ignored personal directory, download its Chromium
headless shell on the login node, and install `imageio-ffmpeg` into `.venv`.
Browser system libraries must be available in the worker image or in a personal
library directory. Do not install packages into a running worker.

The renderer defaults to `work/browser/node_modules/playwright` and
`work/playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell`.
Override with `ARENA_PLAYWRIGHT`, `ARENA_CHROMIUM`, `ARENA_PYTHON`, and
`ARENA_FFMPEG` as needed. The batch wrapper adds
`work/browser-libs/usr/lib/x86_64-linux-gnu` to the library search path.

```bash
export ARENA_NODE=/path/to/node
sbatch workers/render.sbatch --recording runs/conversation \
  --output output/conversation.mp4 --start 2 --seconds 35 --fps 24
```

Rendering uses CPU SwiftShader, requires no allocated GPU, and stays within one
closed batch job. Browser frames are stepped at explicit audio times, then muxed
with the corresponding WAV samples. It produces H.264/AAC MP4 mastered to −18 LUFS / −1.5 dBTP, a poster PNG, a
clip provenance JSON, and encoder/server logs. `--width 1920 --height 1080` enables
1080p output (default 1280×720). Source recordings and exports are ignored by git.

The small bundled ArenaSans font is a subset of Noto Sans CJK KR under SIL OFL;
its license is in `web/src/fonts/LICENSE-Noto.txt`. It keeps the presentation's
English and Korean labels readable without external font downloads.

## Verified showcase recordings (2026-10-12)

| Recording | Models | GPU job | Duration | Scheduler p95 lateness | Startup underruns |
|---|---|---:|---:|---:|---:|
| `runs/showcase-2` | Alex / PersonaPlex NATF2 + Moshi | 189846 | 90 s | 1.57 ms | 1 + 2 frames |
| `runs/showcase-3` | Alex / PersonaPlex NATF2 + Moshi + Sam / PersonaPlex NATM1 | 189845 | 90 s | 1.26 ms | 1 + 2 + 1 frames |

Each model ran on a separate B200; PersonaPlex used about 19.2 GiB per instance,
Moshi about 17.4 GiB. Both runs had zero input overflow, output drops and clipping.
Every connected model's input WAV exactly matched its previous receiver block.
Model speech overlapped for 10.96 s / 48.08 s over the full two/three-agent runs.
There was no prerecorded speaker in either run.

The selected clips use the opening conversation; later portions contain long
monologues and repeated backchannels. These examples demonstrate acoustic
interaction and simultaneous speech, not consistently good long-form dialogue.
The three participants are two PersonaPlex instances and one Moshi, not three
different model architectures. Raon was excluded because the earlier verified
runtime produced degraded audio in longer passages.

`workers/asr_gpu.sbatch` uses a separately allocated GPU for Whisper-small and
word timestamps. Download `small.pt` to `work/asr/` on login first. This is optional;
the CPU ASR batch script remains available. Font/browser installation and builds
belong on login; model inference, ASR and video rendering run through Slurm.

The selected introduction clips cover 0–26.92 s (two agents) and 0–14.92 s
(three agents). Within these intervals, measured model overlap is 3.76 s and
6.96 s respectively. Rendering samples at 24 fps / 1280×720; the source acoustic
engine remains at 80 ms per tick. Animation phases and camera angle follow the
recording time, and the cinematic canvas renders only when that time changes.

Join finished clips and validate both audio/video streams:

```bash
sbatch workers/package_video.sbatch --output output/duplex-arena-introduction.mp4 \
  output/duplex-arena-two-agents.mp4 output/duplex-arena-three-agents.mp4
```

The package sidecar records the input clips and full-stream decode results.
