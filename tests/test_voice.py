"""Offline tests for voice input: the HTTP edge and the local ASR service.

No live weights and no network: the client's upstream is an ``httpx``
MockTransport, the service's model is a fake, and real PyAV decoding is used
only when PyAV is installed.
"""

from __future__ import annotations

import array
import hashlib
import hmac
import io
import math
import sys
import types
import wave
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient

from agent.planner import ScriptedPlanner
from api import asr_service, voice
from api.app import create_app
from api.voice import MAX_AUDIO_BYTES, register_voice

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32
ENDPOINT = "http://127.0.0.1:18090"


# ------------------------------------------------------------------- helpers


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


class RecordingPlanner(ScriptedPlanner):
    """A scripted planner that also proves it was never called."""

    def __init__(self) -> None:
        super().__init__(lambda _user: '{"kind":"refuse","reason":"out_of_scope"}')
        self.calls = 0

    def plan(self, system: str, user: str) -> str:
        self.calls += 1
        return super().plan(system, user)


def _tone_wav(
    seconds: float = 0.1,
    rate: int = 16000,
    freq: float = 440.0,
    amplitude: float = 0.3,
) -> bytes:
    frames = int(rate * seconds)
    samples = array.array(
        "h",
        (
            int(32767 * amplitude * math.sin(2 * math.pi * freq * index / rate))
            for index in range(frames)
        ),
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _dependency(app):
    """Mirror the app's bearer-token check for a route registered by a test."""

    def current_customer(authorization: str | None = Header(default=None)) -> str:
        runtime = app.state.runtime
        if not authorization:
            raise HTTPException(status_code=401, detail="unauthorized")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="unauthorized")
        parts = token.strip().split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="unauthorized")
        customer_id, expiry_text, signature = parts
        try:
            expiry = int(expiry_text)
        except ValueError:
            raise HTTPException(status_code=401, detail="unauthorized") from None
        if expiry <= int(runtime.clock().timestamp()):
            raise HTTPException(status_code=401, detail="unauthorized")
        expected = hmac.new(
            app.state.secret_key,
            f"{customer_id}.{expiry_text}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise HTTPException(status_code=401, detail="unauthorized")
        if customer_id not in runtime.customer_ids:
            raise HTTPException(status_code=401, detail="unauthorized")
        return customer_id

    return current_customer


def _wire(app) -> None:
    # create_app may already wire voice in production; never add a duplicate.
    if any(getattr(route, "path", None) == "/voice/transcribe" for route in app.routes):
        return
    register_voice(app, _dependency(app))


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply = lambda _request: httpx.Response(
            200, json={"text": "你好", "language": "zh"}
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply(request)


@pytest.fixture
def upstream(monkeypatch) -> Upstream:
    fake = Upstream()
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        assert kwargs.get("trust_env") is False
        transport = httpx.MockTransport(fake.handle)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(voice.httpx, "AsyncClient", factory)
    return fake


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("ANBAN_ASR_URL", ENDPOINT)
    planner = RecordingPlanner()
    clock = FixedClock(NOW)
    app = create_app(
        secret_key=SECRET, planner=planner, clock=clock, demo_mode=True
    )
    _wire(app)
    with TestClient(app) as client:
        yield client, planner, clock
    app.state.runtime.conn.close()


def _login(client: TestClient) -> str:
    response = client.post("/auth/demo-login", json={"persona": "young_pro"})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _post_voice(
    client: TestClient,
    token: str | None,
    content: bytes,
    content_type: str = "audio/wav",
):
    headers = {"Content-Type": content_type}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/voice/transcribe", content=content, headers=headers)


# ------------------------------------------------------- client: happy path


def test_transcribe_returns_validated_transcript_without_touching_the_agent(
    harness, upstream
) -> None:
    client, planner, _clock = harness
    token = _login(client)
    before = client.get("/me/summary", headers={"Authorization": f"Bearer {token}"}).json()
    audio = _tone_wav()

    response = _post_voice(client, token, audio)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "text": "你好",
        "language": "zh",
        "model": "Qwen3-ASR-1.7B",
    }
    assert planner.calls == 0
    after = client.get("/me/summary", headers={"Authorization": f"Bearer {token}"}).json()
    assert after["balance_fen"] == before["balance_fen"]

    assert len(upstream.requests) == 1
    forwarded = upstream.requests[0]
    assert str(forwarded.url) == ENDPOINT + "/transcribe"
    assert forwarded.method == "POST"
    assert forwarded.content == audio
    assert forwarded.headers["content-type"] == "audio/wav"
    # The banking session's credentials never reach the speech service.
    assert forwarded.headers.get("authorization") is None
    assert forwarded.headers.get("cookie") is None


def test_content_type_parameters_are_accepted_and_ignored(harness, upstream) -> None:
    client, _planner, _clock = harness
    token = _login(client)

    response = _post_voice(
        client, token, _tone_wav(), content_type="audio/webm; codecs=opus"
    )

    assert response.status_code == 200, response.text
    assert upstream.requests[0].headers["content-type"] == "audio/webm"


# ---------------------------------------------------------- client: auth


def test_transcribe_requires_a_valid_bearer_token(harness, upstream) -> None:
    client, _planner, _clock = harness

    assert _post_voice(client, None, _tone_wav()).status_code == 401
    assert _post_voice(client, "not.a.token", _tone_wav()).status_code == 401
    assert upstream.requests == []


# ---------------------------------------------------- client: input errors


def test_unsupported_media_type_is_415(harness, upstream) -> None:
    client, _planner, _clock = harness
    token = _login(client)

    response = _post_voice(client, token, b"{}", content_type="application/json")

    assert response.status_code == 415, response.text
    assert upstream.requests == []


def test_empty_body_is_400(harness, upstream) -> None:
    client, _planner, _clock = harness
    token = _login(client)

    response = _post_voice(client, token, b"")

    assert response.status_code == 400, response.text
    assert upstream.requests == []


def test_body_over_the_audio_limit_is_413(harness, upstream, monkeypatch) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    monkeypatch.setattr(voice, "MAX_AUDIO_BYTES", 1024)

    response = _post_voice(client, token, b"x" * 2048)

    assert response.status_code == 413, response.text
    assert upstream.requests == []


def test_public_limit_is_eight_mebibytes() -> None:
    assert MAX_AUDIO_BYTES == 8 * 1024 * 1024


# ------------------------------------------------ client: service contract


def test_unset_endpoint_is_503(harness, upstream, monkeypatch) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    monkeypatch.delenv("ANBAN_ASR_URL", raising=False)

    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 503, response.text
    assert upstream.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:18090",
        "http://evil.example:18090",
        "http://127.0.0.1:18090/admin",
        "http://user:pass@127.0.0.1:18090",
        "http://127.0.0.1:18090?next=http://evil.example",
    ],
)
def test_non_loopback_or_pathed_endpoints_are_refused(
    harness, upstream, monkeypatch, url: str
) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    monkeypatch.setenv("ANBAN_ASR_URL", url)

    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 503, response.text
    assert upstream.requests == []


def test_unreachable_service_is_503(harness, upstream) -> None:
    client, _planner, _clock = harness
    token = _login(client)

    def refuse(_request):
        raise httpx.ConnectError("offline")

    upstream.reply = refuse
    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 503, response.text
    assert "offline" not in response.text


def test_non_200_upstream_is_503_and_does_not_echo_the_body(harness, upstream) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    upstream.reply = lambda _request: httpx.Response(500, content=b"SECRET_UPSTREAM")

    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 503, response.text
    assert "SECRET_UPSTREAM" not in response.text


@pytest.mark.parametrize(
    "reply",
    [
        lambda _request: httpx.Response(200, content=b"not-json-SECRET"),
        lambda _request: httpx.Response(200, json=["SECRET"]),
        lambda _request: httpx.Response(200, json={"text": "hi"}),
        lambda _request: httpx.Response(200, json={"text": 1, "language": "zh"}),
        lambda _request: httpx.Response(
            200, json={"text": "hi", "language": "x" * 40}
        ),
    ],
)
def test_malformed_upstream_is_502(harness, upstream, reply) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    upstream.reply = reply

    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 502, response.text
    assert "SECRET" not in response.text


@pytest.mark.parametrize(
    "text",
    ["", "   ", "x" * 501],
)
def test_empty_or_overlong_transcript_is_422(harness, upstream, text: str) -> None:
    client, _planner, _clock = harness
    token = _login(client)
    upstream.reply = lambda _request: httpx.Response(
        200, json={"text": text, "language": "zh"}
    )

    response = _post_voice(client, token, _tone_wav())

    assert response.status_code == 422, response.text


# ------------------------------------------------------------- ASR service


@pytest.fixture
def service_client():
    with TestClient(asr_service.app) as client:
        yield client


def test_service_health_never_loads_or_leaks_the_model(service_client, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANBAN_ASR_MODEL", str(tmp_path))
    monkeypatch.setattr(asr_service, "_model", None)

    response = service_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "configured": True, "loaded": False}
    assert str(tmp_path) not in response.text


def test_service_rejects_bad_mime_empty_and_oversized_bodies(
    service_client, monkeypatch
) -> None:
    assert (
        service_client.post(
            "/transcribe", content=b"x", headers={"Content-Type": "application/json"}
        ).status_code
        == 415
    )
    assert (
        service_client.post(
            "/transcribe", content=b"", headers={"Content-Type": "audio/wav"}
        ).status_code
        == 400
    )

    monkeypatch.setattr(asr_service, "MAX_AUDIO_BYTES", 1024)
    assert (
        service_client.post(
            "/transcribe", content=b"x" * 2048, headers={"Content-Type": "audio/wav"}
        ).status_code
        == 413
    )


def test_service_transcribes_through_an_injected_fake_model(
    service_client, monkeypatch
) -> None:
    class FakeModel:
        def transcribe(self, *, audio, language):
            samples, rate = audio
            assert samples == "pcm"
            assert rate == asr_service.SAMPLE_RATE
            assert language is None
            return [types.SimpleNamespace(text="  你好  ", language="zh")]

    monkeypatch.setattr(asr_service, "_model", FakeModel())
    monkeypatch.setattr(asr_service, "decode_audio", lambda _data: "pcm")

    response = service_client.post(
        "/transcribe",
        content=_tone_wav(),
        headers={"Content-Type": "audio/wav; codecs=1"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "你好", "language": "zh"}


def test_service_busy_slot_is_429(service_client) -> None:
    with asr_service._inference_lock:
        response = service_client.post(
            "/transcribe", content=_tone_wav(), headers={"Content-Type": "audio/wav"}
        )

    assert response.status_code == 429, response.text


def test_service_lock_is_released_after_a_decode_failure(monkeypatch) -> None:
    def fail(_data):
        raise asr_service.AudioError(422, "silent_audio")

    monkeypatch.setattr(asr_service, "decode_audio", fail)

    with pytest.raises(asr_service.AudioError):
        asr_service.transcribe(b"audio")

    assert asr_service._inference_lock.acquire(blocking=False) is True
    asr_service._inference_lock.release()


def test_service_loads_the_official_local_model_arguments(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_qwen = types.ModuleType("qwen_asr")
    fake_qwen.Qwen3ASRModel = FakeModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen)
    monkeypatch.setattr(asr_service, "_model", None)
    monkeypatch.setenv("ANBAN_ASR_MODEL", str(tmp_path))

    loaded = asr_service._get_model()

    assert isinstance(loaded, FakeModel)
    assert calls[0][0] == str(tmp_path)
    assert calls[0][1] == {
        "dtype": "bfloat16",
        "device_map": "cuda:0",
        "max_inference_batch_size": 1,
        "max_new_tokens": 256,
    }
    assert "trust_remote_code" not in calls[0][1]


def test_service_refuses_a_missing_local_model_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(asr_service, "_model", None)
    monkeypatch.setenv("ANBAN_ASR_MODEL", str(tmp_path / "missing"))

    with pytest.raises(asr_service.AudioError) as caught:
        asr_service._get_model()

    assert caught.value.status == 503


# ------------------------------------------------ real decoding (if PyAV is present)


def test_real_decoder_accepts_a_short_tone() -> None:
    pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")

    samples = asr_service.decode_audio(_tone_wav(seconds=0.25))

    assert samples.dtype == numpy.float32
    assert abs(len(samples) - 4000) <= 32
    assert float(numpy.max(numpy.abs(samples))) > 0.1


def test_real_decoder_rejects_silence() -> None:
    pytest.importorskip("av")
    pytest.importorskip("numpy")

    with pytest.raises(asr_service.AudioError) as caught:
        asr_service.decode_audio(_tone_wav(seconds=0.2, amplitude=0.0))

    assert caught.value.status == 422
    assert caught.value.code == "silent_audio"


def test_real_decoder_rejects_audio_over_thirty_seconds() -> None:
    pytest.importorskip("av")
    pytest.importorskip("numpy")

    with pytest.raises(asr_service.AudioError) as caught:
        asr_service.decode_audio(_tone_wav(seconds=31.0))

    assert caught.value.status == 422
    assert caught.value.code == "audio_too_long"


def test_real_decoder_rejects_malformed_bytes() -> None:
    pytest.importorskip("av")
    pytest.importorskip("numpy")

    with pytest.raises(asr_service.AudioError) as caught:
        asr_service.decode_audio(b"this is not an audio container")

    assert caught.value.status == 422


def test_real_decoder_rejects_playlists_with_external_resources():
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXTINF:5,\nhttp://127.0.0.1:1/private.wav\n#EXT-X-ENDLIST\n"
    with pytest.raises(asr_service.AudioError) as caught:
        asr_service.decode_audio(playlist)
    assert caught.value.code == "malformed_audio"
