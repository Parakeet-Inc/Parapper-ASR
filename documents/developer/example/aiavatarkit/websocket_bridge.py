"""WebSocket input: stream microphone audio to Parapper, then call AIAvatarKit."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid

import httpx
import sounddevice as sd
import websockets

from aiavatar_chat import AIAvatarConversation

PARAPPER_URL = os.getenv("PARAPPER_URL", "ws://127.0.0.1:18082/ws/recognition")
PARAPPER_API_KEY = os.getenv("PARAPPER_API_KEY")

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 512  # 32 ms / 1024 bytes


def put_latest(queue: asyncio.Queue[bytes], pcm: bytes) -> None:
    """Keep microphone latency bounded if the event loop briefly falls behind."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(pcm)


async def run() -> None:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
    recognition_session_id = f"mic-{uuid.uuid4().hex[:12]}"
    headers = (
        {"Authorization": f"Bearer {PARAPPER_API_KEY}"} if PARAPPER_API_KEY else None
    )

    def audio_callback(indata, _frames, _time_info, status) -> None:
        if status:
            print(f"microphone warning: {status}")
        loop.call_soon_threadsafe(put_latest, audio_queue, bytes(indata))

    async with websockets.connect(PARAPPER_URL, extra_headers=headers) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "version": 1,
                    "type": "session.start",
                    "session_id": recognition_session_id,
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE,
                        "channels": 1,
                    },
                }
            )
        )

        ready = json.loads(await websocket.recv())
        if ready.get("type") != "session.ready":
            raise RuntimeError(f"expected session.ready, got {ready}")

        sending_audio = True

        async def send_audio() -> None:
            while sending_audio:
                pcm = await audio_queue.get()
                await websocket.send(pcm)

        sender = asyncio.create_task(send_audio())
        stop_event = asyncio.Event()

        def wait_for_enter() -> None:
            input("Press Enter to stop\n")
            loop.call_soon_threadsafe(stop_event.set)

        threading.Thread(target=wait_for_enter, daemon=True).start()
        stop_request: asyncio.Task[bool] | None = asyncio.create_task(stop_event.wait())
        chat_tasks: list[asyncio.Task[None]] = []
        conversation = AIAvatarConversation()

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=SAMPLES_PER_FRAME,
                callback=audio_callback,
            ):
                async with httpx.AsyncClient() as http_client:
                    while True:
                        receive = asyncio.create_task(websocket.recv())
                        waiters: set[asyncio.Task] = {receive}
                        if stop_request is not None:
                            waiters.add(stop_request)
                        done, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stop_request is not None and stop_request in done:
                            sending_audio = False
                            sender.cancel()
                            await asyncio.gather(sender, return_exceptions=True)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "version": 1,
                                        "type": "session.stop",
                                        "session_id": recognition_session_id,
                                    }
                                )
                            )
                            stop_request = None

                        if receive not in done:
                            receive.cancel()
                            await asyncio.gather(receive, return_exceptions=True)
                            continue

                        message = json.loads(receive.result())
                        kind = message.get("type")
                        if kind == "turn.partial":
                            # Preview only. Never start an LLM request for partial text.
                            print(f"\r[partial] {message['text']}", end="", flush=True)
                        elif kind == "turn.final":
                            text = message["text"].strip()
                            print(f"\n[final] {text}")
                            if text:
                                # Only immutable final text enters AIAvatarKit /chat.
                                task = asyncio.create_task(
                                    conversation.send_final(http_client, text)
                                )
                                chat_tasks.append(task)
                        elif kind == "error":
                            raise RuntimeError(
                                f"Parapper error {message.get('code')}: "
                                f"{message.get('message')}"
                            )
                        elif kind == "session.done":
                            if chat_tasks:
                                await asyncio.gather(*chat_tasks)
                            return
        finally:
            sending_audio = False
            sender.cancel()
            tasks: list[asyncio.Task] = [sender]
            if stop_request is not None:
                stop_request.cancel()
                tasks.append(stop_request)
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(run())
