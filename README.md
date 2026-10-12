<div align="center">

# Duplex Arena

**Full-duplex speech agents, sharing one acoustic world.**

[![Simulator checks](https://github.com/KyuDan1/duplex-arena/actions/workflows/check.yml/badge.svg)](https://github.com/KyuDan1/duplex-arena/actions/workflows/check.yml)

A 3D simulator for multi-agent conversation, evaluation, and acoustic trajectory collection.

[Technical report](doc/duplex-arena-technical-report.pdf) · [Quick start](#quick-start) · [Simulation modes](#simulation-modes) · [GPU setup](docs/GPU_SERVER.md) · [Architecture](docs/ARCHITECTURE.md)

**24 kHz audio · simultaneous listening and speaking · per-agent microphone recordings**

</div>

### Two friends

<video controls playsinline width="100%" poster="doc/media/two-friends.jpg" src="https://github.com/KyuDan1/duplex-arena/raw/refs/heads/main/doc/media/two-friends.mp4"></video>

[Watch with sound](doc/media/two-friends.mp4) — PersonaPlex and Moshi share a casual conversation. Two independent models hear each other through the simulated room.

### Clerk and customer

<video controls playsinline width="100%" poster="doc/media/clerk-customer.jpg" src="https://github.com/KyuDan1/duplex-arena/raw/refs/heads/main/doc/media/clerk-customer.mp4"></video>

[Watch with sound](doc/media/clerk-customer.mp4) — Two PersonaPlex agents take the roles of a cafe clerk and customer: an iced oat-latte and croissant order, followed by confirmation.

### Three-way conversation

<video controls playsinline width="100%" poster="doc/media/three-agents.jpg" src="https://github.com/KyuDan1/duplex-arena/raw/refs/heads/main/doc/media/three-agents.mp4"></video>

[Watch with sound](doc/media/three-agents.mp4) — Two PersonaPlex voices and one Moshi agent converse in the same room, including simultaneous speech.

These selected clips contain **actual model-generated audio**, with no prerecorded speaker or scripted dialogue. Speech bubbles are offline ASR of each agent's emitted audio, revealed at word timestamps. The soundtrack mixes clean emissions; each agent's spatially mixed receiver audio is saved separately. See [recording provenance and limitations](docs/SHOWCASE.md).

## What the simulator does

Each agent emits audio while receiving everyone else's audio, transformed by distance, propagation delay, divider occlusion, a wall reflection, and independent microphone noise. The same PCM engine powers the 3D viewer, recorded evaluations, and the Python environment interface.

- **Observe:** speaker and listener activity, TX/RX levels, overlap, timing, and buffer diagnostics in 3D.
- **Experiment:** change positions, noise, occlusion, and scenarios; connect PersonaPlex, Moshi, or a custom PCM worker.
- **Record:** preserve source audio, receiver audio, actual model-input blocks, frame traces, and acoustic rewards.
- **Replay:** inspect a finished GPU experiment on a CPU machine, without loading model weights.

## Quick start

Requires Python 3.11+ and Node.js 22+. Model workers use separate environments; Python 3.12 is a suitable starting point for their dependencies.

```bash
git clone https://github.com/KyuDan1/duplex-arena.git
cd duplex-arena
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
npm --prefix web ci
npm --prefix web run build
duplex-arena serve
```

Open **http://127.0.0.1:8765** and press **Run**. This default mode needs no GPU or model download: it uses synthetic tones and scripted captions to demonstrate the simulator. The videos above use real model workers.

For frontend development, run `npm --prefix web run dev` in a second terminal. The viewer on port 5173 connects to the API on port 8765.

## Simulation modes

| Workflow | Use it for | Entry point | Result |
|---|---|---|---|
| **1. Inference and evaluation** | Let speech models converse; inspect acoustics, responses, overlap, and runtime health | `serve --models …` or `evaluate --models …` | Interactive session or recorded experiment |
| **2. Trajectory collection for RL research** | Preserve acoustic observations/actions/rewards, or integrate a policy with the environment | `evaluate --models …` for real-model recordings; `DuplexArenaEnv` for trainer-controlled steps | Acoustic trajectories; model-side token instrumentation is still needed for SLM RL |

These are two workflows over the same simulator, not two separate CLI flags. The existing full-weight speech models run inference; the included trainable example is a small speak/listen gate policy.

### 1. Run conversations and evaluate inference

Install the streaming bridge in the arena environment:

```bash
pip install -e '.[models]'
```

Follow [GPU setup](docs/GPU_SERVER.md) to install and start the native workers. **PersonaPlex and Moshi must use separate Python environments** because both install a package named `moshi`. Use one native worker process and a distinct endpoint per agent; the verified deployment used one GPU per agent.

Prepare an ignored local configuration from the example. Edit worker URLs and prompts; remove the `custom` entry for a two-model experiment unless its endpoint is running.

```bash
cp configs/workers.example.json configs/local-workers.json
# Edit configs/local-workers.json, then start its workers.

# Interactive simulation: open the viewer and press Run.
duplex-arena serve --models configs/local-workers.json

# Headless evaluation: records a complete 60-second conversation at wall-clock 1×.
duplex-arena evaluate --models configs/local-workers.json \
  --scenario quiet --seconds 60 --output runs/conversation

# Optional room controls: add --configuration configs/two-agent-stage.example.json.
# After evaluation completes, replay its audio and 3D state without GPU workers.
duplex-arena serve --replay runs/conversation --port 18766
```

Open `http://127.0.0.1:18766/` for the engineering view or `http://127.0.0.1:18766/?showcase` for presentation playback. For a remote replay server:

```bash
ssh -N -L 18766:127.0.0.1:18766 YOUR_SSH_LOGIN_ALIAS
```

On the shared B200 cluster, run inference and evaluation together inside a Slurm job, then replay the finished artifacts. The provided cafe configuration launches two PersonaPlex agents:

```bash
export PERSONAPLEX_PYTHON=/path/to/personaplex/environment/bin/python
mkdir -p runs
sbatch --gres=gpu:2 --mem=96G workers/conversation.sbatch \
  --models configs/cafe.example.json \
  --configuration configs/two-agent-stage.example.json \
  --seconds 60 --output runs/cafe
```

The batch launcher starts and stops its workers automatically. For PersonaPlex–Moshi conversations, also set `MOSHI_PYTHON`; see [two/three-agent configurations and video export](docs/SHOWCASE.md). For a controlled opening utterance, [use the WAV stimulus worker](docs/GPU_SERVER.md#4-connect-and-open-the-arena). A WAV source is a stimulus, not another model.

### 2. Collect trajectories for RL research

A real-model evaluation already records an **acoustic trajectory**. Use a new output directory for each episode:

```bash
duplex-arena evaluate --models configs/local-workers.json \
  --configuration configs/two-agent-stage.example.json \
  --seed 42 --seconds 60 --output runs/trajectories/episode-0001
```

| Artifact | Contents |
|---|---|
| `trace.jsonl` | One snapshot per 80 ms tick: geometry, activity, model-generated text, acoustic reward terms, and worker/timing diagnostics |
| `<agent>-emitted.wav` | Agent emission at the simulator boundary: the acoustic action |
| `<agent>-received.wav` | Spatially mixed audio at that agent's microphone |
| `<agent>-input.wav` | PCM block supplied to its adapter: clipped previous-tick receiver audio; initially silence |
| `summary.json` | Completion status, duration, timing, final state, and input alignment |

WAVs use 24 kHz mono PCM16. A tick spans 1,920 samples, so tick indices align the audio files with the trace. Live inference is wall-clock paced and is not bitwise reproducible from the acoustic seed alone. Preserve model revisions, sampling settings, prompts, and runtime versions with each experiment; the conversation batch launcher adds a manifest.

For trainer-controlled rollouts, use the Python environment. All agents act simultaneously; each observation contains its own mixed microphone audio, position, and simulation time:

```python
import numpy as np
from duplex_arena.env import DuplexArenaEnv

env = DuplexArenaEnv(seed=42, horizon=750)
observations, infos = env.reset()
while env.agents:
    # Replace silence with each policy's next 1,920-sample float32 PCM emission.
    actions = {
        agent: np.zeros(env.frame_size, dtype=np.float32)
        for agent in env.agents
    }
    next_observations, rewards, terminated, truncated, infos = env.step(actions)
    # Store observations, actions, rewards, next_observations and boundary flags here.
    observations = next_observations
env.close()
```

This PettingZoo-style dictionary interface is not a registered PettingZoo environment and does not automatically attach the native model servers. A working CPU REINFORCE example supplies its own tiny stochastic gate policies, logs behavior log-probabilities and policy versions, and updates them after each frozen-policy rollout:

```bash
python examples/on_policy.py --episodes 12 --output runs/gates
```

**Full-weight SLM RL is not implemented.** Native audio/text WebSockets do not expose sampled token actions, behavior log-probabilities, value estimates, token-to-audio alignment, or learner/actor policy-version barriers. Acoustic recordings alone are insufficient to claim an on-policy SLM update. The required interfaces and staged training plan are in [SLM training](docs/SLM_TRAINING.md).

## Validation and interpretation

The [B200 verification record](docs/B200_VERIFICATION.md) and [showcase report](docs/SHOWCASE.md) document real PersonaPlex/Moshi runs, receiver-to-input checks, memory use, pacing, and buffer counters. The showcased two/three-agent runs had no dropped output, input overflow, or clipping, with small startup underruns recorded explicitly. They demonstrate the interaction path; selected short clips do not establish reliable long-form dialogue or role adherence.

- Levels are **dBFS**, not calibrated sound pressure. Listener activity detects audio energy, not comprehension.
- Acoustic propagation uses 343 m/s; visual wave rings are deliberately slowed. Reflection is a single wall image source, without HRTF, diffraction, or a calibrated room impulse response.
- Rewards combine audibility, overlap, and energy. They are acoustic heuristics, not semantic conversation scores; useful backchannels may receive overlap penalties.
- Model-generated text, emitted-audio captions, and [receiver ASR](docs/ASR.md) have distinct provenance. The presentation's word-timed captions are offline ASR, not a streaming ASR latency measurement.
- The current engine has three named agent slots and excludes self-sidetone/self-echo. Live UI history is bounded; use headless evaluation to retain complete recordings.

Check the CPU engine and worker transport without model weights:

```bash
# Synthetic evaluation and tests.
duplex-arena evaluate --scenario cocktail --seed 42 --seconds 18 --output runs/demo
pytest -q

# In separate terminals: an echo protocol probe, then its viewer (not a language model).
python -m workers.echo_worker --port 9001
duplex-arena serve --models configs/echo.example.json
```

## Documentation

[Technical report (PDF)](doc/duplex-arena-technical-report.pdf) · [LaTeX source](doc/duplex-arena-technical-report.tex) · [Architecture](docs/ARCHITECTURE.md) · [GPU setup](docs/GPU_SERVER.md) · [Model protocols](docs/MODELS.md) · [ASR](docs/ASR.md) · [Showcase recording](docs/SHOWCASE.md) · [SLM training plan](docs/SLM_TRAINING.md)

Built around [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) and [Kyutai Moshi](https://github.com/kyutai-labs/moshi). Model names identify supported integrations; they do not imply affiliation with their providers.
