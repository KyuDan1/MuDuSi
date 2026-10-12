"""Raon-SpeechChat stateful PyTorch PCM bridge (one session per process).

Runs inference on one dedicated thread while the socket receives independently.
Queue overflow fails explicitly rather than dropping microphone observations.
Install Raon in its own environment; no Raon/Torch dependency is added to Arena.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import time

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class RaonInference:
    def __init__(self, model_path):
        import torch
        from raon import RaonPipeline
        self.torch = torch
        self.pipe = RaonPipeline(model_path, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
        if self.pipe.processor.sampling_rate != 24000 or self.pipe.processor.frame_rate != 12.5:
            raise ValueError("Raon checkpoint must use 24 kHz / 12.5 fps")
        self.state = None
        self.times = []

    def start(self):
        torch = self.torch
        with torch.inference_mode():
            tokens = self.pipe.processor(
                [{"role": "system", "content": "You are engaging in real-time conversation. Answer in English."}],
                add_generation_prompt=False, device="cuda", dtype=torch.bfloat16)["input_ids"]
            self.state = self.pipe.model.init_duplex_decoding_state(
                sequences=tokens, attention_mask=torch.ones_like(tokens), do_sample=True,
                temperature=.9, top_p=.95, top_k=66, speak_first=False)
        self.times = []

    def step(self, audio):
        torch = self.torch
        started = time.perf_counter()
        with torch.inference_mode():
            before = self.state.sequences.shape[1]
            pcm = torch.from_numpy(audio.copy()).to(device="cuda", dtype=torch.bfloat16)[None]
            self.state, output = self.pipe.model.duplex_decoding_step(state=self.state, audio_input=pcm)
            tokens = self.state.sequences[0, before:].tolist()
            tokens = [t for t in tokens if t < self.pipe.model.text_vocab_size]
            text = self.pipe.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            output = output.float().cpu().numpy().reshape(-1)
        self.times.append(time.perf_counter() - started)
        return output, text

    def close(self):
        if self.state is not None:
            self.pipe.model.free_duplex_decoding_state(self.state)
            self.state = None
        print(json.dumps({"frames": len(self.times), "step_ms_mean": float(np.mean(self.times)*1000) if self.times else None,
                          "peak_allocated_mib": self.torch.cuda.max_memory_allocated()/2**20}), flush=True)


async def run(args):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        async def call(fn, *values):
            return await loop.run_in_executor(executor, functools.partial(fn, *values))
        model = await call(RaonInference, args.model)
        lock = asyncio.Lock()

        async def handler(ws):
            if lock.locked():
                await ws.close(code=1013, reason="One session per Raon worker")
                return
            async with lock:
                tasks = []
                try:
                    init = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    expected = {"type":"init", "protocol":"duplex-arena.pcm.v1", "sample_rate":24000,
                                "channels":1, "frame_size":1920, "format":"f32le"}
                    if any(init.get(k) != v for k,v in expected.items()):
                        raise ValueError("Invalid PCM initialization")
                    await call(model.start)
                    await ws.send(json.dumps({"type":"status", "state":"ready", "worker":"Raon-SpeechChat-9B (PyTorch)"}))
                    queue = asyncio.Queue(maxsize=32)

                    async def receive():
                        async for message in ws:
                            if not isinstance(message, bytes) or len(message) != 7680:
                                raise ValueError("Expected 1920 float32 samples")
                            pcm = np.frombuffer(message, dtype="<f4")
                            if not np.all(np.isfinite(pcm)):
                                raise ValueError("Non-finite audio")
                            try:
                                queue.put_nowait(pcm)
                            except asyncio.QueueFull:
                                raise RuntimeError("Raon input overflow: inference slower than real time")

                    async def generate():
                        while True:
                            pcm = await queue.get()
                            output, text = await call(model.step, pcm)
                            await ws.send(output.astype("<f4").tobytes())
                            if text:
                                await ws.send(json.dumps({"type":"text", "text":text}))

                    tasks = [asyncio.create_task(receive()), asyncio.create_task(generate())]
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                except ConnectionClosed:
                    pass
                except Exception as exc:
                    print(f"Raon session failed: {exc}", flush=True)
                    try:
                        await ws.send(json.dumps({"type":"error", "message":str(exc)}))
                    except ConnectionClosed:
                        pass
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    # The executor serializes cleanup after any in-flight CUDA step.
                    await call(model.close)

        async with serve(handler, "127.0.0.1", args.port, max_size=7680):
            print(f"Raon ready on {args.port}", flush=True)
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8998)
    asyncio.run(run(parser.parse_args()))
