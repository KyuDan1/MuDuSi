import asyncio

import numpy as np
from websockets.asyncio.server import serve

from duplex_arena.adapters import RaonWebSocketAdapter, FRAME_SIZE


def test_raon_native_float_audio_text_and_failure():
    async def run():
        async def handler(ws):
            await ws.send(b"\x00")
            incoming = await ws.recv()
            assert incoming[0] == 1 and len(incoming) == 1 + FRAME_SIZE * 4
            # Preserve float samples, unlike the Opus protocol on Moshi.
            await ws.send(incoming)
            await ws.send(b"\x02Hello")
            await ws.send(b"\x08pong")
            await ws.recv()
            await ws.send(b"\x05runtime failed")
            await ws.wait_closed()
        async with serve(handler, "127.0.0.1", 0) as server:
            adapter = RaonWebSocketAdapter(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/realtime/ws")
            try:
                await adapter.start()
                pcm = np.linspace(-.4, .4, FRAME_SIZE, dtype=np.float32)
                await adapter.push_audio(pcm)
                for _ in range(100):
                    if adapter.stats["received_samples"] >= FRAME_SIZE:
                        break
                    await asyncio.sleep(.01)
                np.testing.assert_array_equal(adapter.pull_audio(), pcm)
                assert adapter.drain_text() == ["Hello"]
                await adapter.push_audio(pcm)
                for _ in range(100):
                    if adapter.error:
                        break
                    await asyncio.sleep(.01)
                assert "runtime failed" in adapter.error
            finally:
                await adapter.close()
    asyncio.run(run())
