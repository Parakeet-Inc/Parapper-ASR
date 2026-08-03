"""Check AIAvatarKit v0.8.19's native Parapper protocol integration."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version
import json
import os
from pathlib import Path
import uuid
import wave

import websockets
from aiavatar.sts.vad.parapper_stream import ParapperStreamSpeechDetector


EXPECTED_AIAVATAR_VERSION = "0.8.19"
SAMPLE_RATE = 16_000
FRAME_BYTES = 1600  # 50 ms of mono PCM s16le; below Parapper's 3200 byte limit.


@dataclass
class CheckResult:
    recording_started: int = 0
    partials: list[str] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def print_event(event_type: str, **payload: object) -> None:
    print(
        json.dumps(
            {"event": event_type, **payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual_format = (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        )
        expected_format = (SAMPLE_RATE, 1, 2, "NONE")
        if actual_format != expected_format:
            raise ValueError(
                "WAV must be 16kHz mono 16-bit uncompressed PCM: "
                f"got rate={actual_format[0]} channels={actual_format[1]} "
                f"sample_width={actual_format[2]} compression={actual_format[3]}"
            )
        return wav.readframes(wav.getnframes())


async def check_server(
    *,
    url: str,
    api_key: str | None,
    pcm: bytes,
    realtime: bool,
) -> CheckResult:
    installed_version = version("aiavatar")
    if installed_version != EXPECTED_AIAVATAR_VERSION:
        raise RuntimeError(
            "AIAvatarKit version mismatch: "
            f"expected {EXPECTED_AIAVATAR_VERSION}, got {installed_version}"
        )

    detector = ParapperStreamSpeechDetector(
        url=url,
        api_key=api_key,
        connect_timeout=5.0,
        drain_timeout=10.0,
        debug=True,
    )
    result = CheckResult()
    session_id = f"aiavatarkit-check-{uuid.uuid4().hex[:12]}"

    @detector.on_recording_started
    async def on_recording_started(callback_session_id: str) -> None:
        result.recording_started += 1
        print_event("recording.started", session_id=callback_session_id)

    @detector.on_speech_detecting
    async def on_partial(text: str, _session: object) -> None:
        result.partials.append(text)
        print_event("turn.partial", text=text)

    @detector.on_speech_detected
    async def on_final(
        audio: bytes,
        text: str,
        metadata: dict,
        duration: float,
        callback_session_id: str,
    ) -> None:
        result.finals.append(text)
        print_event(
            "turn.final",
            session_id=callback_session_id,
            text=text,
            audio_bytes=len(audio),
            audio_duration_seconds=duration,
            metadata=metadata,
        )

    @detector.on_speech_recognition_error
    async def on_error(error: Exception, callback_session_id: str) -> None:
        result.errors.append(str(error))
        print_event(
            "error",
            session_id=callback_session_id,
            message=str(error),
        )

    print_event(
        "check.start",
        aiavatarkit_version=installed_version,
        url=url,
        audio_bytes=len(pcm),
    )
    detector_connected = False
    try:
        for offset in range(0, len(pcm), FRAME_BYTES):
            frame = pcm[offset : offset + FRAME_BYTES]
            await detector.process_samples(frame, session_id)
            detector_connected = True
            if realtime:
                await asyncio.sleep(len(frame) / (SAMPLE_RATE * 2))
    finally:
        if detector_connected:
            await detector.finalize_session(session_id)

    if result.errors:
        raise RuntimeError("; ".join(result.errors))
    print_event(
        "check.complete",
        recording_started=result.recording_started,
        partial_count=len(result.partials),
        final_count=len(result.finals),
    )
    return result


async def self_test() -> None:
    observed_frames: list[int] = []
    observed_start: dict | None = None
    observed_stop: dict | None = None

    async def handler(socket) -> None:
        nonlocal observed_start, observed_stop
        observed_start = json.loads(await socket.recv())
        protocol_session_id = observed_start["session_id"]
        await socket.send(
            json.dumps(
                {
                    "version": 1,
                    "type": "session.ready",
                    "session_id": protocol_session_id,
                    "capabilities": {
                        "partial": True,
                        "speech_started": True,
                        "cancel": True,
                    },
                }
            )
        )
        speech_started = False
        while True:
            message = await socket.recv()
            if isinstance(message, bytes):
                observed_frames.append(len(message))
                if not speech_started:
                    speech_started = True
                    await socket.send(
                        json.dumps(
                            {
                                "version": 1,
                                "type": "speech.started",
                                "session_id": protocol_session_id,
                            }
                        )
                    )
                    await socket.send(
                        json.dumps(
                            {
                                "version": 1,
                                "type": "turn.partial",
                                "session_id": protocol_session_id,
                                "turn_id": 1,
                                "revision": 1,
                                "text": "疎通",
                            },
                            ensure_ascii=False,
                        )
                    )
                continue
            observed_stop = json.loads(message)
            if observed_stop.get("type") == "session.stop":
                break

        await socket.send(
            json.dumps(
                {
                    "version": 1,
                    "type": "turn.final",
                    "session_id": protocol_session_id,
                    "turn_session_id": 1,
                    "turn_id": 1,
                    "revision": 2,
                    "segment_id": 1,
                    "previous_segment_id": None,
                    "text": "疎通確認に成功しました。",
                    "source_asr_model": "self-test",
                    "source_language": "ja",
                    "detected_language": None,
                    "audio_duration_ms": 100,
                    "elapsed_ms": 10,
                },
                ensure_ascii=False,
            )
        )
        await socket.send(
            json.dumps(
                {
                    "version": 1,
                    "type": "session.done",
                    "session_id": protocol_session_id,
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await check_server(
            url=f"ws://127.0.0.1:{port}/ws/recognition",
            api_key=None,
            pcm=b"\x01\x00" * (SAMPLE_RATE // 10),
            realtime=False,
        )

    assert observed_start is not None
    assert observed_start["version"] == 1
    assert observed_start["type"] == "session.start"
    assert observed_start["audio"] == {
        "encoding": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
    }
    assert observed_stop is not None
    assert observed_stop["type"] == "session.stop"
    assert observed_stop["session_id"] == observed_start["session_id"]
    assert observed_frames and max(observed_frames) <= 3200
    assert result.recording_started == 1
    assert result.partials == ["疎通"]
    assert result.finals == ["疎通確認に成功しました。"]
    print_event("self_test.pass", binary_frame_sizes=observed_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--url",
        default=os.getenv(
            "PARAPPER_URL",
            "ws://127.0.0.1:18082/ws/recognition",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PARAPPER_API_KEY"),
    )
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    if args.self_test:
        await self_test()
        return

    pcm = load_wav(args.wav) if args.wav else b"\x00\x00" * (SAMPLE_RATE // 10)
    result = await check_server(
        url=args.url,
        api_key=args.api_key,
        pcm=pcm,
        realtime=not args.no_realtime,
    )
    if args.wav and not result.finals:
        raise RuntimeError("WAV was sent, but Parapper returned no turn.final")
    print_event(
        "manual_check.pass",
        mode="wav" if args.wav else "handshake",
    )


def main() -> int:
    try:
        asyncio.run(async_main(parse_args()))
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print_event("check.failed", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
