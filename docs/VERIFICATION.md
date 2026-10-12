# Verification of v0.1

For actual Linux/B200 inference results, see [B200 verification](B200_VERIFICATION.md).
The original development-machine checks below are retained as historical scope.

Checked locally on macOS/Apple Silicon, Python 3.14 and Node.js 24.
The GitHub workflow separately targets Ubuntu, Python 3.12 and Node.js 22.

## Completed checks

- 64 Python tests pass: impulse propagation across frame boundaries, attenuation,
  independent seeded noise, occlusion, reflection, absence of self audio,
  deterministic reset, per-agent rewards, environment truncation, input validation.
- Native Opus bridge is exercised with the real `sphn` codec and a local test
  server, including an encode/decode roundtrip and received frequency/RMS checks.
- Three independent PCM WebSocket sessions are exercised through the live
  simulator, including reset, queues, error propagation and cleanup.
- WAV stimulus validation, final-frame padding, silence after playback and reset.
- Injected ASR tests verify exact receiver PCM routing, queue bounds, reset races,
  startup failure handling and late recognition included in export while paused.
- Production viewer passes TypeScript checks and Vite build. The Three.js bundle
  currently produces a non-fatal size warning (about 338 kB compressed).
- Browser check: real WebGL scene loads; scenario change adds a partition and
  updates loss; selecting/moving/muting agents changes live state; play/pause,
  frame stepping, 2× demo speed and timeline/RMS telemetry operate. No browser
  console warnings or errors observed. Document has no horizontal overflow at
  the tested 1374 px viewport.
- An 18-second interruption evaluation produces 225 frames, three 24 kHz mono
  WAVs, full trace and summary. Demo overlap ratio for seed 42 is 0.3099.
- Four 50-frame REINFORCE episodes run and save behavior probabilities,
  per-agent reward components and updated toy gate policies.

## Not claimed

- Actual Moshi/PersonaPlex weights, GPU throughput or B100 compatibility were
  not tested on this Mac. Follow `GPU_SERVER.md` for target-server acceptance.
- Actual faster-whisper recognition/model loading was not executed. ASR queue
  and data-routing tests use an injected recognizer; enabling it downloads weights.
- Docker files and CI configuration are included. A local Docker build was not
  run; runtime Web UI location is explicit via `DUPLEX_ARENA_WEB_DIR`.
- The demo is synthetic tones, not speech. Its captions are authored scripts.
- The toy policy is a working environment/training example, not a fine-tuned SLM
  or evidence of improving semantic conversation quality.
- Acoustics are a lightweight direct-path and single-reflection approximation.
