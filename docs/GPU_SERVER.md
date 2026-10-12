# Run on a multi-GPU Linux server

On the shared B200/k8sl cluster, full evaluations belong in Slurm, with both
workers and the simulator inside the same job. Worker ingress is closed; use
`serve --replay` on completed artifacts for the viewer instead of forwarding into
a worker. See [tested B200 commands and results](B200_VERIFICATION.md).

Use one worker process and one visible GPU per native speech agent, with the
arena and viewer served by a lightweight CPU process. For example, GPU 0 runs
Moshi, GPU 1 runs PersonaPlex, and GPU 2 can run a custom speech model. A custom
worker is an integration point; this repository does not bundle a third model.

GPU count alone is not enough: checkpoint size, precision, context length,
concurrent sessions and inference implementation determine memory/throughput.
Bounded PersonaPlex–Moshi inference runs have been verified on B200 GPUs;
[recorded measurements](B200_VERIFICATION.md) cover memory use, frame timing,
audio routing, and buffer diagnostics. These measurements validate that setup,
not a cross-hardware or conversation-quality benchmark.

## 1. Install the arena

From your clone of Duplex Arena, use Python 3.12 for broad model-package
compatibility and a recent Node LTS to build the viewer:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,models]'
npm --prefix web ci
npm --prefix web run build
.venv/bin/python -m pytest -q
```

The arena's `.venv` holds no Torch or model weights. The Opus library `sphn` is
used only by native streaming bridges. CPU demo mode does not need `[models]`.

## 2. Isolate the two native model environments

Moshi and PersonaPlex both expose a Python package named `moshi`; installing
them together can overwrite modules. They also declare different dependency
versions. Keep them in separate virtual environments and keep their source
checkouts outside this project's tracked files.

The commands below follow the upstream source installations. Pin the source
commit and checkpoint revision used for a reported experiment.

```bash
mkdir -p work/model-sources .venvs
git clone https://github.com/kyutai-labs/moshi.git work/model-sources/moshi
git clone https://github.com/NVIDIA/personaplex.git work/model-sources/personaplex
python3.12 -m venv .venvs/moshi
python3.12 -m venv .venvs/personaplex
.venvs/moshi/bin/pip install ./work/model-sources/moshi/moshi
.venvs/personaplex/bin/pip install ./work/model-sources/personaplex/moshi
```

For PersonaPlex, install your OS's Opus development library (`libopus-dev` on
Ubuntu/Debian), accept the selected model's license on Hugging Face, and
authenticate in the worker environment. Do not place access tokens in the
tracked JSON configuration. Upstream installation details:
[Moshi](https://github.com/kyutai-labs/moshi#pytorch-implementation),
[PersonaPlex](https://github.com/NVIDIA/personaplex#usage).

**For B100/other Blackwell GPUs:** select a PyTorch/CUDA build supporting that
GPU before launching. PersonaPlex's current README points to PyTorch's CUDA 13.0
wheel index, while its package metadata still pins an older Torch range. That
upstream mismatch needs resolution in your chosen source revision/environment;
the arena does not hide it or claim the stock dependency resolution is tested
on B100. Follow [PersonaPlex's Blackwell note](https://github.com/NVIDIA/personaplex#installation)
and verify the actual worker environment, for example:

```bash
nvidia-smi
.venvs/moshi/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
.venvs/personaplex/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
.venvs/moshi/bin/pip check
.venvs/personaplex/bin/pip check
```

Do not interpret a successful CUDA allocation as a complete inference test.
The selected models must still finish initialization and process audio. Store
resolved package versions and any intentional upstream dependency overrides
with the experiment. PersonaPlex offers CPU offload for memory pressure, but
offload latency must be measured for real-time use.

## 3. Start one worker per model

Run these in separate terminals from the project root:

```bash
# Terminal A: GPU 0, Moshi, port 8998
bash workers/launch_moshi.sh 0 8998

# Terminal B: GPU 1, PersonaPlex, port 8999
bash workers/launch_personaplex.sh 1 8999

# Terminal C: CPU protocol probe, until your custom GPU model is integrated
.venv/bin/python -m workers.echo_worker --port 9001
```

The scripts set `CUDA_VISIBLE_DEVICES` before starting each process. Consequently
the selected physical GPU appears as `cuda:0` inside that worker. Override
`MOSHI_PYTHON` or `PERSONAPLEX_PYTHON` if the environments live elsewhere. Optional
`MOSHI_HF_REPO`/`PERSONAPLEX_HF_REPO` select a compatible checkpoint.

Wait for each worker to finish weight download and warmup. Every native agent
needs its own process: the reference servers lock inference per session, so two
agents using one native endpoint will wait for one another rather than conduct
independent conversations. The CPU echo worker is multi-session and is only a
protocol probe. It will be silent when its received signal is silent.

Keep model ports on a private network. **The reviewed PersonaPlex server ignores
`--host` in its final `web.run_app` call**. The launcher now runs
`workers.native_server`, which enforces the requested aiohttp bind address without
editing a shared upstream checkout. Direct `python -m moshi.server` invocations
still have the upstream behavior. See the [upstream server implementation](https://github.com/NVIDIA/personaplex/blob/main/moshi/moshi/server.py).

## 4. Connect and open the arena

```bash
.venv/bin/duplex-arena serve --host 127.0.0.1 --port 8765 \
  --models configs/workers.example.json
```

`configs/workers.example.json` maps each of the three displayed agents to its
worker. Replace the `custom` PCM endpoint when your third model worker exists.
If experimenting with only two models, remove `custom` from the configuration;
the unconnected agent remains explicitly silent.

For a controlled opening utterance, use the WAV source in place of the CPU echo
probe on port 9001. Supply your own 24 kHz mono PCM16 WAV containing a question or
other evaluation stimulus:

```bash
.venv/bin/python -m workers.wav_worker --wav /path/to/question.wav \
  --port 9001 --delay-seconds 0.8
```

The custom character now plays the recording once and then remains silent while
the native models can respond. Reset restarts the source. This avoids relying on
models to spontaneously start talking when every initial observation is silence.

From your laptop, forward the viewer/API port (replace `your-gpu-server`):

```bash
ssh -L 8765:127.0.0.1:8765 your-gpu-server
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Native audio stays within the
GPU server; the browser receives visualization, receiver audio and telemetry.
No physical microphones or speakers participate in the simulated sound field.
The optional browser audio monitor plays the rendered receiver sound.

## Acceptance run

1. Run the CPU echo configuration first to verify worker transport. Its captions
   explicitly identify it as a probe; it is not a conversation-quality test.
2. Start native model workers and the native configuration; verify API health
   has `mode: "live"` with no error, and worker counters increase after Play.
3. Inspect audible model output, model text, input/output levels, underruns,
   dropped output, clipping and clock overruns. A healthy connection with no
   generated speech is not evidence of a useful conversation.
4. Export a trace and each receiver WAV. Record model/source revisions, prompts,
   GPU type, driver, Torch/CUDA versions, seed and room settings.
5. Reset before comparing noise, geometry or interruption scenarios. Live model
   sampling and network scheduling are not guaranteed deterministic by the
   arena's acoustic seed. Demo scenario captions are not prompts sent to native
   models; use persona prompts or the WAV stimulus worker for controlled tasks.

If startup fails, the UI/API reports a model error and stays stopped. Check
worker logs, model access, endpoint ports and GPU memory, then Reset to reconnect.
There is no fallback from a failed live model to demo output.
