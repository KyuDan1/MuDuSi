# Contributing to MuDuSi

MuDuSi is a virtual acoustic testbed for speech-agent interaction. Contributions should make its behavior easier to reproduce, inspect, or extend. Useful areas include model workers, acoustic tests, receiver ASR/diarization observers, evaluation tasks, multi-experiment tooling, and model-side rollout interfaces for RL.

With repository access, use [GitHub issues](https://github.com/KyuDan1/MuDuSi/issues) to describe a bug or discuss a substantial interface change, then submit a focused pull request with the resulting behavior and relevant validation. Small fixes can go directly to a pull request. Do not claim model inference, speech quality, or training behavior that the submitted checks did not exercise.

## Local setup

Requires Python 3.11+ and Node.js 22+. Work in a personal virtual environment:

```bash
git clone https://github.com/KyuDan1/MuDuSi.git
cd MuDuSi
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,models]'
npm --prefix web ci
npm --prefix web run build
mudusi serve
```

Open `http://127.0.0.1:8765` and press **Run**. The default demonstration uses synthetic audio and scripted captions; it needs no model weights or GPU. In another terminal, `npm --prefix web run dev` starts the development viewer on port 5173 with the API still on port 8765.

The public command and distribution name are `mudusi`. The internal Python package remains `duplex_arena`, and `duplex-arena` remains a compatibility CLI alias. The custom worker protocol identifier is still `duplex-arena.pcm.v1`; changing branding does not change a wire protocol.

## Validate the changed behavior

For code changes, run the relevant focused tests first. Before submitting changes that affect the engine, API, or frontend, run the corresponding checks used by CI:

```bash
pytest -q
npm --prefix web run build
mudusi evaluate --seconds 1 --output runs/contribution-smoke
python examples/on_policy.py --episodes 2 --horizon 10 --output runs/contribution-gates
```

The tests exercise CPU acoustics, environment semantics, worker protocols, recording, replay, and observer behavior. They do not load native speech-model weights. For a documentation-only change, check the commands, local links, and factual claims rather than starting GPU workloads.

Test the property your change is meant to preserve: sample arrival and mixing for acoustics; observation/action alignment for rollouts; malformed packets, queue handling, reset, and cleanup for workers. Avoid tests that merely duplicate the implementation. Record failed checks or untested hardware paths in the pull request.

## Add or improve a model worker

Read [model protocols](docs/MODELS.md) and [GPU setup](docs/GPU_SERVER.md). A new model can implement the generic PCM WebSocket protocol without changing the acoustic engine. Use `workers/echo_worker.py` as a transport reference, not as a model implementation.

1. Keep model dependencies in a separate personal environment. PersonaPlex and Moshi both provide a `moshi` package and must not share an environment.
2. Handle the documented initialization and ready acknowledgment. Receive 1,920 mono float32 samples at 24 kHz per input message; resample native model audio to the protocol rate when necessary.
3. Maintain model state for one connection and run input reception independently of output generation. Allocate separate native processes/endpoints when the upstream server serializes sessions.
4. Preserve causal input/output order. Report underruns, overflow, failure, and clipping; do not silently substitute synthetic speech or drop model observations.
5. Close tasks, sockets, codec/model state, and owned subprocesses on normal completion and failure. Reset must begin a fresh session.
6. Add a sanitized configuration example and explain its dependencies, expected GPU memory, checkpoint/source revision, and known limitations.

A CPU transport smoke test is available in separate terminals:

```bash
python -m workers.echo_worker --port 9001
mudusi serve --models configs/echo.example.json
```

Then perform a bounded real-model evaluation when GPU validation is needed. Preserve emitted, received, and input WAVs with `trace.jsonl` and `summary.json`; verify that model output reaches another agent's next input and that queue counters remain healthy. Report model loading separately from steady-state behavior. On a shared cluster, follow local scheduling rules and use batch jobs for sustained inference or benchmarks.

## Add speech analysis or RL integrations

Keep these evidence streams distinct:

- Native model text describes generated output.
- Receiver ASR analyzes the noisy, spatially mixed microphone signal.
- Showcase captions are offline ASR of clean emitted audio.
- Speaker identity from a known source channel is not the output of a diarization model.

The built-in receiver ASR does not provide speaker attribution. An external diarizer can consume saved receiver WAVs, but live integration requires an observer and event schema carrying speaker labels and timing. Make that addition explicit, with tests for overlap, reset, latency, and dropped analysis windows. Document whether an analyzer observes the experiment or changes the agents' inputs or rewards.

For RL work, start with `duplex_arena/env.py`, `examples/on_policy.py`, and the [SLM training contract](docs/SLM_TRAINING.md). The tiny gate example performs actual REINFORCE updates. Native speech workers currently expose audio/text inference, not a complete on-policy learner contract. A token-policy integration must log the sampled actions and behavior probabilities, identify the policy version, align tokens to audio/rewards, handle state resets and truncation, and coordinate actor/learner updates. Do not present black-box acoustic recordings as sufficient evidence of on-policy SLM training.

## Documentation and reproducible evidence

Update the relevant guide when commands, artifact formats, protocol semantics, or prerequisites change. Separate what is implemented, what was measured, and what is proposed. Include a minimal reproducer and validation results in the pull request; for GPU results, include hardware, runtime/model revisions, duration, timing, and buffer diagnostics. A successful connection alone does not establish useful speech generation or dialogue quality.

The README describes supported workflows. `docs/` contains engineering guides; `doc/` contains the technical report and its sources. Rebuild the report when changing its source, using the instructions in `doc/README.md`. Keep project-page claims consistent with the same evidence.

## Recordings, credentials, and large files

Use ignored `configs/local*.json` files for personal endpoints, paths, and prompts. Keep run artifacts under `runs/`, exports under `output/`, and downloaded weights or private tools outside tracked source files. The conversation launcher can write prompts and worker configuration into its manifest, so inspect artifacts before sharing them.

Do not commit access tokens, credentials, model weights, private conversations, or large raw experiment dumps. Share only recordings you are authorized to publish, with appropriate source/model attribution and clear generated-versus-prerecorded provenance. Prefer small, sanitized fixtures for tests. Check `git diff --cached` before committing; ignored paths are a convenience, not a substitute for reviewing what is being published.

## License

Simulator code is licensed under [MIT](LICENSE). Submit code you have the right to contribute under those terms, and preserve required copyright and license notices. Model weights and bundled font assets retain their own licenses; adding an integration does not relicense its dependencies or checkpoints.
