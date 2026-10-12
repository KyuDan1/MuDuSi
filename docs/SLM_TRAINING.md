# Follow-up: training speech-model weights

The live PCM/Opus evaluation path is a black-box inference experiment. It does
not expose behavior probabilities and must not be used as a claimed on-policy
SLM update. The existing NumPy gate example remains a separate toy experiment.

## Required model-side contract

Implement a rollout endpoint alongside the native inference server, with a
matching gradient-capable learner for each model family. Each action record
needs `episode_id`, `agent_id`, `policy_version`, checkpoint revision, sampling
configuration, input sample interval, generated text/audio-code token IDs,
token positions, behavior log-probabilities, valid-action masks, value estimates,
and an explicit mapping from tokens to emitted PCM sample intervals. Record
KV/recurrent-state reset boundaries and truncation versus natural termination.

The observation is the agent's own spatially mixed microphone block. Global
positions, other agents' clean audio and future receiver blocks must not leak
into an actor input. A centralized critic may consume privileged global state,
but its features must be separately tagged and stored.

## Implementation sequence

1. Add an instrumented, frozen single-model rollout worker. Verify logged
   behavior probabilities against a recomputation with the exact same weights,
   token history, masks and sampling distribution (including temperature/top-k).
2. Add version barriers: finish or discard all episodes of version N, publish
   version N+1, reset KV state, and acknowledge the new version before rollout.
   Pausing a WebSocket does not implement this barrier.
3. Implement a learner minibatch with per-token rewards/advantages aligned to
   audio intervals. Distinguish acoustic delay, 80 ms simulation decision delay,
   inference latency and codec delay. Test delayed reward attribution and
   truncation bootstrapping with short deterministic examples.
4. Start with adapters/LoRA and one actor before allocating full optimizer state.
   Add KL/reference-policy control and loss masks for conditioning versus sampled
   actions. Verify finite gradients and checkpoint/resume on a bounded GPU job.
5. Only then expand to full-weight distributed optimization and multiple actors.
   Determine memory from measured parameters/optimizer states; do not reuse
   inference VRAM measurements as a training memory estimate. On this cluster,
   use one Slurm experiment per allocation and checkpoint to NAS.

## Evaluation gates

Retain the current acoustic diagnostics, but add externally labeled task success,
response latency, interruption yield/recovery, intelligibility, and appropriate
backchannel timing. Receiver ASR and generated text are separate evidence streams;
neither alone proves comprehension. Compare a held-out frozen baseline with
paired stimuli, controlled acoustics and multiple model seeds. A lower overlap
penalty is not by itself evidence of better conversation.

Raon exposes stateful token-level PyTorch decoding, while native Moshi/PersonaPlex
WebSockets expose only generated text and audio. Instrument those model internals
instead of reconstructing policy actions from lossy Opus or ASR output.
