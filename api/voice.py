"""Voice edge for AnBan: raw audio in, bounded text out.

The endpoint is deliberately narrow.  It accepts only a small set of audio
content types, never forwards the caller's bearer token or cookies, and talks
to one configured loopback ASR service.  Planning, banking and mandates are not
touched here: the transcript is returned to the client, which decides what to
do with it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 500
MAX_LANGUAGE_CHARS = 32
MODEL_NAME = "Qwen3-ASR-1.7B"
TIMEOUT_SECONDS = 90.0

ALLOWED_AUDIO_TYPES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
    }
)

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
_ASR_PATH = "/transcribe"


def _audio_type(value: str | None) -> str | None:
    """Return the bare media type, ignoring parameters such as ``codecs``."""

    if not value:
        return None
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type if media_type in ALLOWED_AUDIO_TYPES else None


def _endpoint() -> str | None:
    """Resolve ``ANBAN_ASR_URL`` to the one allowed loopback endpoint.

    Only plain HTTP to the literal ``127.0.0.1`` or ``localhost`` hosts is
    accepted; userinfo, queries, fragments and arbitrary paths are refused so
    the configured value can never become an SSRF destination.  The request
    path is always the service's ``/transcribe`` route.
    """

    raw = os.environ.get("ANBAN_ASR_URL", "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
        hostname = parts.hostname
        if parts.port is not None and not 1 <= parts.port <= 65535:
            return None
    except ValueError:
        return None
    if parts.scheme != "http":
        return None
    if hostname not in _ALLOWED_HOSTS:
        return None
    if parts.username is not None or parts.password is not None:
        return None
    if parts.query or parts.fragment:
        return None
    if parts.path not in ("", "/", _ASR_PATH):
        return None
    return parts._replace(path=_ASR_PATH, query="", fragment="").geturl()


async def _read_audio(request: Request) -> bytes:
    """Read at most ``MAX_AUDIO_BYTES`` from the request body."""

    declared = request.headers.get("content-length")
    if declared:
        try:
            length = int(declared)
        except ValueError:
            raise HTTPException(status_code=400, detail="bad_request") from None
        if length > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="payload_too_large")

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="payload_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _translate(response: httpx.Response) -> dict[str, str]:
    """Validate the service reply and shape the public answer.

    Upstream bodies, errors and transcripts are never echoed: only a small,
    validated set of fields leaves this function.
    """

    if response.status_code != 200:
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="speech_busy")
        if response.status_code == 422:
            try:
                code = response.json().get("detail")
            except (ValueError, AttributeError):
                code = None
            public = {"silent_audio": "no_speech", "empty_transcript": "no_speech",
                      "audio_too_long": "audio_too_long", "malformed_audio": "invalid_audio"}
            raise HTTPException(status_code=422, detail=public.get(code, "invalid_audio") if isinstance(code, str) else "invalid_audio")
        raise HTTPException(status_code=503, detail="speech_unavailable")

    try:
        payload: Any = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="speech_malformed") from None

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="speech_malformed")

    text = payload.get("text")
    language = payload.get("language")
    if not isinstance(text, str) or not isinstance(language, str):
        raise HTTPException(status_code=502, detail="speech_malformed")
    if len(language) > MAX_LANGUAGE_CHARS:
        raise HTTPException(status_code=502, detail="speech_malformed")

    text = text.strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail="invalid_transcript")

    return {"text": text, "language": language, "model": MODEL_NAME}


def register_voice(app: FastAPI, current_customer: Callable[..., str]) -> None:
    """Attach ``POST /voice/transcribe`` to an existing app."""

    @app.post("/voice/transcribe")
    async def voice_transcribe(
        request: Request,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, str]:
        content_type = _audio_type(request.headers.get("content-type"))
        if content_type is None:
            raise HTTPException(status_code=415, detail="unsupported_media_type")

        endpoint = _endpoint()
        if endpoint is None:
            raise HTTPException(status_code=503, detail="speech_unavailable")

        audio = await _read_audio(request)
        if not audio:
            raise HTTPException(status_code=400, detail="empty_audio")

        try:
            async with httpx.AsyncClient(
                trust_env=False, timeout=TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                # Raw bytes only: no bearer token and no cookies from the
                # banking session are forwarded to the speech service.
                response = await client.post(
                    endpoint, content=audio, headers={"Content-Type": content_type}
                )
        except httpx.HTTPError:
            raise HTTPException(status_code=503, detail="speech_unavailable") from None

        return _translate(response)
