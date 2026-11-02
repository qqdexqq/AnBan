"""Standalone loopback ASR service for AnBan voice input.

Run on the homeserver next to the API::

    python -m api.asr_service

Heavy ML dependencies (torch, qwen_asr, av, numpy) are imported lazily, so the
banking app and the offline test suite never need them.  The model is loaded on
the first transcription (never on import or health checks) and a single
non-blocking process lock serialises decode + inference; a concurrent request is
answered with 429 rather than queueing behind the GPU.
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 30
SAMPLE_RATE = 16000
MAX_SAMPLES = MAX_SECONDS * SAMPLE_RATE
MAX_TRANSCRIPT_CHARS = 500
MAX_LANGUAGE_CHARS = 32
SILENCE_RMS = 1e-5

ALLOWED_AUDIO_TYPES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
    }
)

_MODEL_ENV = "ANBAN_ASR_MODEL"

app = FastAPI(title="AnBan ASR", version="0.1")

_inference_lock = threading.Lock()
_model: Any = None


class AudioError(Exception):
    """A bounded, user-safe decoding or inference failure."""

    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


class BusyError(Exception):
    """The single inference slot is already busy."""


def _audio_type(value: str | None) -> str | None:
    if not value:
        return None
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type if media_type in ALLOWED_AUDIO_TYPES else None


def _model_dir() -> str | None:
    """Return the configured local model directory, or None.

    The directory must already exist: ``from_pretrained`` is never allowed to
    fetch weights over the network.
    """

    raw = os.environ.get(_MODEL_ENV, "").strip()
    if not raw:
        return None
    path = os.path.abspath(raw)
    return path if os.path.isdir(path) else None


def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model

    model_dir = _model_dir()
    if model_dir is None:
        raise AudioError(503, "model_not_configured")

    try:
        import torch
        import qwen_asr

        loaded = qwen_asr.Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
    except Exception:
        # No paths, tracebacks or upstream messages escape the service.
        raise AudioError(503, "model_unavailable") from None

    _model = loaded
    return loaded


def _resampled(resampler: Any, frame: Any) -> list[Any]:
    result = resampler.resample(frame)
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return [item for item in result if item is not None]
    return [result]


def decode_audio(data: bytes) -> Any:
    """Decode any allowlisted container to mono 16 kHz float32 samples.

    The input is bounded to 30 seconds (480 000 samples), including the
    resampler's final flush.  Silence, non-finite samples, malformed streams
    and missing audio tracks are rejected before the model is called.
    """

    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    try:
        container = av.open(io.BytesIO(data), mode="r", options={
            "protocol_whitelist": "pipe",
            "format_whitelist": "wav,matroska,webm,ogg,mov",
        })
    except Exception:
        raise AudioError(422, "malformed_audio") from None

    chunks: list[Any] = []
    total = 0

    def consume(frame: Any) -> None:
        nonlocal total
        # Refuse a single oversized frame before materialising it.
        declared = int(getattr(frame, "samples", 0) or 0)
        if total + declared > MAX_SAMPLES:
            raise AudioError(422, "audio_too_long")
        samples = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
        total += int(samples.size)
        if total > MAX_SAMPLES:
            raise AudioError(422, "audio_too_long")
        chunks.append(samples)

    try:
        try:
            audio_streams = container.streams.audio
            if not audio_streams:
                raise AudioError(422, "no_audio")
            stream = audio_streams[0]
            try:
                resampler = AudioResampler(
                    format="flt", layout="mono", rate=SAMPLE_RATE
                )
            except Exception:
                raise AudioError(503, "decoder_unavailable") from None

            for frame in container.decode(stream):
                for converted in _resampled(resampler, frame):
                    consume(converted)
            # Return the resampler's buffered tail so a stream that is just
            # over the limit cannot hide the excess in the flush.
            for converted in _resampled(resampler, None):
                consume(converted)
        except AudioError:
            raise
        except Exception:
            raise AudioError(422, "malformed_audio") from None
    finally:
        container.close()

    if total == 0:
        raise AudioError(422, "no_audio")

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if not bool(np.all(np.isfinite(audio))):
        raise AudioError(422, "nonfinite_audio")

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < SILENCE_RMS:
        raise AudioError(422, "silent_audio")

    return audio


def _first(result: Any) -> Any:
    if result is None:
        raise AudioError(502, "empty_result")
    if isinstance(result, (list, tuple)):
        if not result:
            raise AudioError(502, "empty_result")
        return result[0]
    return result


def _extract(result: Any) -> dict[str, str]:
    item = _first(result)
    if isinstance(item, dict):
        text = item.get("text")
        language = item.get("language")
    else:
        text = getattr(item, "text", None)
        language = getattr(item, "language", None)

    if not isinstance(text, str):
        raise AudioError(502, "bad_transcript")
    text = text.strip()
    if not text:
        raise AudioError(422, "empty_transcript")
    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise AudioError(422, "transcript_too_long")

    if language is None:
        language = ""
    if not isinstance(language, str) or len(language) > MAX_LANGUAGE_CHARS:
        raise AudioError(502, "bad_language")
    return {"text": text, "language": language}


def transcribe(data: bytes) -> dict[str, str]:
    """Decode and transcribe one bounded audio payload.

    Runs synchronously in a worker thread.  The lock is taken without blocking
    and is released in ``finally`` on every path, including decode failures and
    exceptions from the model.
    """

    if not _inference_lock.acquire(blocking=False):
        raise BusyError()

    try:
        audio = decode_audio(data)
        model = _get_model()
        try:
            result = model.transcribe(audio=(audio, SAMPLE_RATE), language=None)
        except Exception:
            raise AudioError(503, "inference_failed") from None
        return _extract(result)
    finally:
        _inference_lock.release()


async def _read_audio(request: Request) -> bytes:
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


@app.get("/health")
def health() -> dict[str, bool]:
    """Report readiness without loading the model or leaking its path."""

    return {
        "ok": True,
        "configured": _model_dir() is not None,
        "loaded": _model is not None,
    }


@app.post("/transcribe")
async def transcribe_endpoint(request: Request) -> dict[str, str]:
    content_type = _audio_type(request.headers.get("content-type"))
    if content_type is None:
        raise HTTPException(status_code=415, detail="unsupported_media_type")

    data = await _read_audio(request)
    if not data:
        raise HTTPException(status_code=400, detail="empty_audio")

    try:
        return await run_in_threadpool(transcribe, data)
    except BusyError:
        raise HTTPException(status_code=429, detail="busy") from None
    except AudioError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.code) from None


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18090)


if __name__ == "__main__":
    main()
