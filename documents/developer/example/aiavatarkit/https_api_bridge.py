"""HTTP(S) input: receive Parapper events and forward final text to AIAvatarKit."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, Field

from aiavatar_chat import AIAVATAR_CHAT_URL, AIAvatarConversation

BRIDGE_HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "15522"))
SSL_CERTFILE = os.getenv("SSL_CERTFILE")
SSL_KEYFILE = os.getenv("SSL_KEYFILE")


class ParapperRecognitionEvent(BaseModel):
    """Version 1 payload sent by Parapper's developer HTTP connection."""

    model_config = ConfigDict(populate_by_name=True)

    version: Literal[1]
    event_type: Literal["turn.partial", "turn.final"] = Field(alias="type")
    id: str
    text: str
    turn_session_id: int
    turn_id: int
    revision: int
    output_sequence: int
    segment_id: int
    previous_segment_id: int | None
    source_asr_model: str
    source_language: str
    detected_language: str | None
    recognized_at_ms: int
    elapsed_ms: int
    audio_duration_ms: int | None


conversation = AIAvatarConversation()
chat_tasks: set[asyncio.Task[None]] = set()


def report_chat_result(task: asyncio.Task[None]) -> None:
    chat_tasks.discard(task)
    if task.cancelled():
        return
    if error := task.exception():
        print(f"AIAvatarKit request failed: {error}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    try:
        yield
    finally:
        if chat_tasks:
            await asyncio.gather(*chat_tasks, return_exceptions=True)
        await app.state.http_client.aclose()


app = FastAPI(title="parapper-aiavatarkit-https-example", lifespan=lifespan)


@app.post("/api/events", status_code=202)
async def receive_recognition_event(
    event: ParapperRecognitionEvent,
) -> Response:
    if event.event_type == "turn.partial":
        # Preview only. Never start an LLM request for partial text.
        print(f"[partial] {event.text}", flush=True)
        return Response(status_code=202)

    text = event.text.strip()
    print(f"[final]   {text}", flush=True)
    if text:
        # Only immutable final text enters AIAvatarKit /chat.
        task = asyncio.create_task(conversation.send_final(app.state.http_client, text))
        chat_tasks.add(task)
        task.add_done_callback(report_chat_result)
    return Response(status_code=202)


def main() -> None:
    if bool(SSL_CERTFILE) != bool(SSL_KEYFILE):
        raise SystemExit("Set both SSL_CERTFILE and SSL_KEYFILE, or neither")
    scheme = "https" if SSL_CERTFILE else "http"
    print(f"Parapper event URL: {scheme}://{BRIDGE_HOST}:{BRIDGE_PORT}/api/events")
    print(f"AIAvatarKit chat:   {AIAVATAR_CHAT_URL}")
    uvicorn.run(
        app,
        host=BRIDGE_HOST,
        port=BRIDGE_PORT,
        ssl_certfile=SSL_CERTFILE,
        ssl_keyfile=SSL_KEYFILE,
    )


if __name__ == "__main__":
    main()
