"""Use AIAvatarKit v0.8.19's native Parapper detector with a microphone."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid

import httpx
import sounddevice as sd
from aiavatar.sts.vad.parapper_stream import ParapperStreamSpeechDetector

from aiavatar_chat import AIAvatarConversation


PARAPPER_URL = os.getenv(
    "PARAPPER_URL",
    "ws://127.0.0.1:18082/ws/recognition",
)
PARAPPER_API_KEY = os.getenv("PARAPPER_API_KEY")

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 512  # 32 ms / 1024 bytes


class ParapperConnectionError(RuntimeError):
    """Raised when the configured Parapper WebSocket cannot be started."""


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
    stop_event = asyncio.Event()
    session_id = f"mic-{uuid.uuid4().hex[:12]}"
    detector = ParapperStreamSpeechDetector(
        url=PARAPPER_URL,
        api_key=PARAPPER_API_KEY,
        sample_rate=SAMPLE_RATE,
        channels=1,
        debug=True,
    )
    conversation = AIAvatarConversation()
    chat_tasks: set[asyncio.Task[None]] = set()
    detector_connected = False

    def report_chat_result(task: asyncio.Task[None]) -> None:
        chat_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            print(f"\nAIAvatarKit request failed: {error}", flush=True)

    def audio_callback(indata, _frames, _time_info, status) -> None:
        if status:
            print(f"microphone warning: {status}")
        loop.call_soon_threadsafe(put_latest, audio_queue, bytes(indata))

    def wait_for_enter() -> None:
        input("Press Enter to stop\n")
        loop.call_soon_threadsafe(stop_event.set)

    async with httpx.AsyncClient() as http_client:

        @detector.on_recording_started
        async def on_recording_started(_session_id: str) -> None:
            print("\n[speech.started]", flush=True)

        @detector.on_speech_detecting
        async def on_partial(text: str, _session: object) -> None:
            # Preview only. Never start an LLM request for partial text.
            print(f"\r[partial] {text}", end="", flush=True)

        @detector.on_speech_detected
        async def on_final(
            _audio: bytes,
            text: str,
            _metadata: dict,
            _duration: float,
            _session_id: str,
        ) -> None:
            final_text = text.strip()
            print(f"\n[final] {final_text}", flush=True)
            if not final_text:
                return
            # Only immutable final text enters AIAvatarKit /chat.
            task = asyncio.create_task(
                conversation.send_final(http_client, final_text)
            )
            chat_tasks.add(task)
            task.add_done_callback(report_chat_result)

        @detector.on_speech_recognition_error
        async def on_recognition_error(
            error: Exception,
            _session_id: str,
        ) -> None:
            print(f"\nParapper recognition failed: {error}", flush=True)

        try:
            # Establish the WebSocket before opening the microphone. This also
            # prevents finalize_session() from awaiting the same failed ready
            # task after an initial connection failure.
            try:
                await detector.process_samples(
                    b"\x00\x00" * SAMPLES_PER_FRAME,
                    session_id,
                )
            except Exception as error:
                raise ParapperConnectionError(
                    f"Parapperに接続できません: {PARAPPER_URL}\n"
                    "Parapperの接続タブで開発者向け接続をONにし、"
                    "接続modeをWebSocket、入力ソースをWebSocketに設定して"
                    "Startを押してください。表示がWaitingForClientになってから"
                    "このexampleを実行し、portも接続先URLと一致させてください。\n"
                    f"詳細: {error}"
                ) from error

            detector_connected = True
            print(f"Connected to Parapper: {PARAPPER_URL}", flush=True)
            threading.Thread(target=wait_for_enter, daemon=True).start()
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=SAMPLES_PER_FRAME,
                callback=audio_callback,
            ):
                while not stop_event.is_set():
                    try:
                        pcm = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.1,
                        )
                    except TimeoutError:
                        continue
                    await detector.process_samples(pcm, session_id)
        finally:
            if detector_connected:
                # session.stop is sent and turn.final/session.done are drained here.
                await detector.finalize_session(session_id)
                if chat_tasks:
                    await asyncio.gather(*chat_tasks)


def main() -> int:
    try:
        asyncio.run(run())
    except ParapperConnectionError as error:
        print(f"\n{error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
