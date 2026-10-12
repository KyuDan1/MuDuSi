# Matrix experiment results — October 12, 2026

## Verified four-GPU collection

Slurm job **190030** completed all nine ordered pairs of Moshi, PersonaPlex, and Raon with eighteen independent model processes. Each room recorded **60 simulation/audio seconds / 750 frames**. Structural routing and buffer-health checks passed, and all eighteen agents emitted nonzero audio. There were **no underruns after the first simulation second**, no input overflow, dropped output/text, clipping, or tick deadline overruns.

This is a **0.25× collection run**, not a real-time throughput result: each 60-second recording took **240.013–240.198 wall-clock seconds**, excluding startup and cleanup. Audio remains 24 kHz mono and replays at normal speed. The nine recording starts spanned **9.636 ms**, and all recording intervals overlapped for **240.004 seconds**. The dashboard's common clock reflects simulation/audio time.

The protected `own` allocation used **four B200 GPUs, 48 CPUs, and 512 GiB host RAM**. Job-local GPUs 0–3 held **5, 5, 4, and 4 model processes**, respectively. There were eighteen distinct recorded process IDs and endpoints, with one acoustic engine per room. Processes shared GPU compute; they did not share model sessions or audio across rooms.

For all eighteen agents, the model-input WAV matched the receiver's previous PCM16 block, initially silence. Every frame retained both peer acoustic paths, and the muted `custom` slot stayed silent. These checks establish the recorded audio routing and runtime behavior; they do not by themselves establish intelligibility, comprehension, or conversational quality.

## Four-GPU timing and audio buffers

The startup column counts underruns within the **first simulation second**—four wall-clock seconds at this pacing. An underrun means insufficient samples for an output frame; it does not necessarily mean that the whole frame was silent. Left/right order follows the room name. All listed later-underrun and deadline-overrun counts are zero.

| Left → right | First-second underruns L / R | Later underruns L / R | Tick lateness p95 (ms) | Deadline overruns |
|---|---:|---:|---:|---:|
| Moshi → Moshi | 2 / 2 | 0 / 0 | 1.77 | 0 |
| Moshi → PersonaPlex | 2 / 1 | 0 / 0 | 15.62 | 0 |
| Moshi → Raon | 2 / 2 | 0 / 0 | 15.84 | 0 |
| PersonaPlex → Moshi | 1 / 2 | 0 / 0 | 15.14 | 0 |
| PersonaPlex → PersonaPlex | 2 / 2 | 0 / 0 | 28.91 | 0 |
| PersonaPlex → Raon | 1 / 2 | 0 / 0 | 25.89 | 0 |
| Raon → Moshi | 2 / 2 | 0 / 0 | 13.82 | 0 |
| Raon → PersonaPlex | 2 / 1 | 0 / 0 | 22.90 | 0 |
| Raon → Raon | 2 / 2 | 0 / 0 | 16.27 | 0 |

Each participant had one or two initial underrun frames. Pre-barrier and post-barrier startup counters recorded no queued input or received samples; no hidden startup overflow/drop was excluded from the result. Final queues retained at most two 80 ms output blocks beyond the recording boundary. The deadline-overrun threshold here is the **320 ms wall tick**, not the 80 ms simulation tick.

## Four-GPU memory and utilization

The common recording interval contains **45 telemetry samples per GPU**. The table summarizes that interval, rather than averaging model loading and idle startup into inference utilization.

| Job-local GPU | Model processes | Mean used memory (GiB) | Maximum sampled memory (GiB) | Mean GPU utilization |
|---|---:|---:|---:|---:|
| 0 | 5 | 98.590 | 98.669 | 52.38% |
| 1 | 5 | 98.589 | 98.669 | 42.56% |
| 2 | 4 | 78.252 | 78.296 | 33.56% |
| 3 | 4 | 78.248 | 78.296 | 35.09% |

These are sampled device-wide readings for all colocated processes, not exact allocator peaks or isolated per-model requirements. Observed total memory is distinct from Raon's configured cache capacity. See [the measured configuration](MATRIX.md#measured-four-gpu-configuration) for explicit cache limits and launch commands.

## Runtime provenance and remaining quality review

The run manifest records commit `29bfd751d2530925f1b146029a9dba9a1b590c52` and `working_tree_dirty: false`. All six Raon worker logs contain the explicit **decoder-child** Mimi window-mask installation message. This provides evidence that the opt-in bridge reached the spawned decoder in this run; the earlier baseline's parent-only message did not.

The run combined several changes: reduced input pacing, a different GPU placement, explicit Raon cache limits, and the spawned-decoder compatibility fix. The improved buffer behavior is **not a controlled ablation** identifying which change caused it. Neither this comparison nor reduced GPU utilization establishes real-time capacity for eighteen models.

All six Raon logs also recorded the upstream `compile_audio_modules failed (continuing without)` warning **twice per worker**. The reported error was a mismatch between masked positions and audio-embedding element counts during compile warmup. The runtime caught it and continued; successful inference therefore does **not** demonstrate that all audio modules compiled successfully. This warning is separate from the verified decoder-child window-mask installation.

## ASR-based review and unresolved speech quality

Dependent job **190032** completed offline Whisper-small transcription of **36 WAVs**: both emitted and both received streams in each of nine rooms. The following observations come from reading those ASR outputs, **not from a human listening test**. ASR can introduce errors or hallucinations; these observations are not a calibrated intelligibility score or proof of comprehension.

| Room | What the emitted/received ASR supports | Limitations |
|---|---|---|
| Moshi → Moshi | Connected discussion of Dungeons & Dragons and its fifth edition | No factual-accuracy or conversational-quality benchmark |
| Moshi → PersonaPlex | Related exchanges about hobbies, games, painting, and parks | Some awkward phrases remain |
| PersonaPlex → Moshi | Related movie/weekend planning | Repetition and a tangent about coats/blankets |
| PersonaPlex → PersonaPlex | Related coffee, museum, and weekend plans | Odd phrasing and repeated planning questions |
| Raon → PersonaPlex | Recognizable coffee-shop discussion | Confused details and topic shifts |
| PersonaPlex → Raon | Recognizable weekend, music, and skydiving discussion | Odd details and repeated goodbyes |
| Moshi → Raon; Raon → Moshi | Some recognizable phrases, but substantial phonetic-looking or nonsensical text in the Raon stream | The problems appear in both Raon's clean emitted-audio ASR and its partner's received-audio ASR; **neither pairing is a speech-quality pass** |
| Raon → Raon | Loose discussion of work, family, and ethics | Broken phrases and weak topical continuity |

The receiver transcriptions preserve recognizable content from the other agent, consistent with the independent sample-level routing check. However, the Moshi↔Raon cases show why zero sustained underruns cannot serve as a speech-quality criterion. Garbled ASR on both clean emissions and their received versions is not explained solely by a receiver-only transcription view; attribution among generated speech, codec/runtime behavior, and recognition errors still needs targeted listening and controlled follow-up. No blanket dialogue-quality pass is claimed for all nine rooms.

### Targeted follow-up

Integration inspection found the expected Raon transport: kind byte `0x01`, float32 little-endian mono PCM at 24 kHz, delivered in 1,920-sample simulation blocks. It did not identify a format mismatch, low input level, or clipping as the cause of the garbled cases. That narrows the integration checks without diagnosing the remaining speech problem.

The next controlled test should feed Raon a fixed input, retain native text, generated acoustic codes, and PCM, then decode **the same acoustic codes** through offline and streaming Mimi paths. This would distinguish an acoustic-generation issue from a codec/streaming discrepancy. The upstream call signature also filters the separate `code_temperature`/`code_top_k` configuration, leaving the first code under the joint sampler; this is an **untested potential factor**, not an established cause. No such follow-up run is claimed in these results.

Recognizable text is not a factual-correctness pass either: the coffee transcript includes a claim that cold brew is caffeine-free. Model truthfulness and task accuracy have not been validated.

## Presentation capture

Render job **190043** completed the common **0–54 second simulation interval**, spotlighting each room for six seconds in this order: Moshi→Raon, Moshi→PersonaPlex, Raon→Moshi, PersonaPlex→Moshi, Raon→Raon, Moshi→Moshi, Raon→PersonaPlex, PersonaPlex→Raon, PersonaPlex→PersonaPlex. This preserves the common recording clock; the soundtrack selects one room at a time rather than mixing independent rooms. Packaging job **190044** produced the web video at **1920 × 1080, 24 fps, 54 seconds**. All **1,296 video frames** and the audio stream decoded without errors, and the browser renderer reported no JavaScript errors. The final poster was visually checked for all nine rooms, matching audio focus, captions, and the 0.25× collection label. The [project-page demo](https://kyudan1.github.io/MuDuSi/#nine-rooms) includes [machine-readable provenance](https://kyudan1.github.io/MuDuSi/assets/nine-rooms.json) with the video SHA-256, inference/render job IDs, source revision, focus order, and verification flags. These media checks do not change the speech-quality limitations above.

## Earlier six-GPU baseline

Slurm job **189994** recorded all nine ordered model pairs concurrently on October 12, 2026. Every room completed **60 seconds / 750 frames** at **1× input pacing**, with 24 kHz mono audio. The eighteen recorded workers have distinct process IDs and endpoints, and each of six B200 GPUs hosted one Moshi, one PersonaPlex, and one Raon worker. Model states and acoustic rooms were separate; GPU compute was shared.

All eighteen agents emitted nonzero audio. Every recorded model-input WAV matched its receiver's preceding PCM16 block, starting with silence; peer acoustic paths were present throughout and the muted third slot emitted silence. Structural/routing verification passed. **Audio-health verification failed:** Raon workers repeatedly lacked enough output samples to fill an 80 ms frame at this input pace. Completion and correct routing do not make these recordings a throughput or conversation-quality success.

The start barrier released at **17:52:17.795 UTC**. The manifest reports **8.581 ms** between the earliest and latest room start callbacks; recorded evaluator timestamps give **8.567 ms**, with **60.003 seconds** common to all nine recording intervals. Each room's recording wall duration was 60.011–60.183 seconds. The root manifest's final end time includes process cleanup and should not be used as the audio-recording duration.

### Baseline timing and buffers

Each underrun entry counts a simulation frame with insufficient output samples, not necessarily a whole silent frame. The table divides counts into the **first simulation second** and **after that first second**; this explicit cutoff is a descriptive startup comparison, not a claim that every early event is harmless. Left/right order follows the room name.

| Left → right | First-second underruns L / R | Later underruns L / R | Total underruns L / R | Tick lateness p95 (ms) | Deadline overruns |
|---|---:|---:|---:|---:|---:|
| Moshi → Moshi | 5 / 5 | 0 / 0 | 5 / 5 | 1.20 | 5 |
| Moshi → PersonaPlex | 5 / 5 | 0 / 0 | 5 / 5 | 17.41 | 6 |
| Moshi → Raon | 5 / 7 | 0 / 216 | 5 / 223 | 9.26 | 5 |
| PersonaPlex → Moshi | 5 / 5 | 0 / 0 | 5 / 5 | 16.42 | 6 |
| PersonaPlex → PersonaPlex | 5 / 5 | 1 / 1 | 6 / 6 | 18.56 | 5 |
| PersonaPlex → Raon | 5 / 7 | 0 / 215 | 5 / 222 | 14.69 | 6 |
| Raon → Moshi | 7 / 5 | 245 / 0 | 252 / 5 | 9.38 | 5 |
| Raon → PersonaPlex | 6 / 5 | 220 / 1 | 226 / 6 | 9.50 | 6 |
| Raon → Raon | 6 / 6 | 257 / 256 | 263 / 262 | 8.08 | 4 |

Across all eighteen workers, **input overflow, dropped output samples, dropped text messages, decoded clipping, and ADC clipping were zero**. Before the barrier and immediately after its release, the recorded input/output/underrun/drop counters were zero. The later counts therefore describe activity during the recorded experiment, not overflow hidden during initialization.

Raon accumulated **222–263 affected frames per 750-frame run**, including **215–257 after the first second**. Moshi underruns were confined to the first second. Three PersonaPlex instances had one later underrun each. All rooms also recorded deadline overruns, even where the p95 was low; percentile timing alone should not conceal those events.

### Baseline GPU measurements

`gpu-telemetry.jsonl` contains twelve GPU samples within the common recording interval, collected at approximately five-second spacing. Means over that interval were:

| Job-local GPU | Mean used memory (GiB) | Mean GPU utilization |
|---|---:|---:|
| 0 | 98.47 | 95.17% |
| 1 | 98.47 | 95.50% |
| 2 | 98.46 | 93.33% |
| 3 | 98.25 | 95.50% |
| 4 | 98.25 | 93.33% |
| 5 | 98.25 | 93.75% |

These are sampled device readings for all colocated processes, not allocator peaks or per-model memory estimates. High utilization and sustained Raon underruns are consistent with insufficient compute headroom at this placement; this experiment does not isolate GPU contention from each runtime's own decoding and buffering behavior.

### Baseline provenance

The baseline enabled the opt-in Mimi SDPA sliding-window bridge in the Raon **parent process only**. Its decoder was created in a fresh spawned process, where that process-local change was not propagated. Consequently this baseline is **not validation of the corrected spawned-decoder path**. A startup message saying the parent bridge was enabled does not establish that the audio decoder used it. The subsequent implementation propagates the opt-in correction to the spawned decoder; its GPU/audio behavior must be evaluated in a new run.

The run manifest records repository commit `2022c03a0ffcd722fcfc8677e3417122ed5c780c` with `working_tree_dirty: true`. Matrix implementation work was present as uncommitted changes when the run began; later commit `eeca256` records matrix code, and `4c0b862` includes subsequent pacing/spawned-decoder work. **Neither later commit should be described as the exact clean source revision executed by this baseline.** The retained manifests, logs, configuration, and recording artifacts are the evidence for this run.

Offline Whisper ASR is present in the verification artifact, but it is not a ground-truth transcript, intelligibility score, or conversation-quality pass. This report makes no ASR-based quality acceptance claim and no claim that every agent understood its partner.

## Scheduling and evidence records

The six-GPU 0.5× submission **190025** was cancelled while pending because of capacity; it never ran and is not an inference failure. The verified four-GPU run above is **190030**, using `configs/matrix.four-gpu.json`. The six-GPU `configs/matrix.example.json` at 0.5× is a separate configuration and must not inherit the measured results of either completed run.

Evidence remains in ignored local recording directories:

| Run | Directory | Structural/routing result | Buffer-health result |
|---|---|---|---|
| 190030, four GPUs, 0.25× | `runs/matrix-nine-four-gpu-20261012/` | Pass | Pass, allowing recorded initial underruns |
| 189994, six GPUs, 1× | `runs/matrix-nine-20261012/` | Pass | Fail |

Each directory retains `manifest.json`, `matrix.json`, `verification.json`, `gpu-telemetry.jsonl`, and the per-room summaries, traces, WAVs, and worker logs. The verifier checks completion, chronology, participant activity, routing, and buffer conditions separately; a successful process exit is not the entire acceptance criterion. Recording manifests may include prompts or private configuration and are not automatically public artifacts.

See [Matrix workflow](MATRIX.md) for reproduction, verification, captions, and presentation rendering. Video selection and audio-focus editing do not change which inference rooms ran concurrently.
