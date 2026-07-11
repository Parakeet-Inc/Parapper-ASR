"""Shared AIAvatarKit HTTP chat client for both Parapper input examples."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import httpx

AIAVATAR_CHAT_URL = os.getenv("AIAVATAR_CHAT_URL", "http://127.0.0.1:8000/chat")


class AIAvatarConversation:
    def __init__(self) -> None:
        self.session_id = f"parapper-{uuid.uuid4().hex[:12]}"
        self.context_id: str | None = None
        self._lock = asyncio.Lock()

    async def send_final(self, client: httpx.AsyncClient, text: str) -> None:
        """Forward one finalized Turn and consume AIAvatarKit's SSE response."""
        async with self._lock:
            payload = {
                "type": "start",
                "session_id": self.session_id,
                "user_id": "parapper-example",
                "context_id": self.context_id,
                "text": text,
            }
            async with client.stream(
                "POST", AIAVATAR_CHAT_URL, json=payload, timeout=120.0
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line.removeprefix("data:").strip())
                    if event.get("context_id"):
                        self.context_id = event["context_id"]
                    if event.get("type") == "chunk" and event.get("text"):
                        print(event["text"], end="", flush=True)
                    elif event.get("type") == "error":
                        raise RuntimeError(
                            f"AIAvatarKit error: {event.get('metadata')}"
                        )
            print(flush=True)
