"""Teacher-LLM utterance generation per persona and scenario.

The teacher rewrites a deterministic seed request into several natural variants
in a persona's style.  Everything here is synthetic training data; nothing is
sent to a real bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_CACHE_PATH = Path("training/data/teacher_cache.jsonl")
MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 1.0
MAX_VARIANT_CHARS = 200
LONG_DIGITS = re.compile(r"\d{11,}")

PROMPT_TEMPLATE = """You are generating synthetic training data for a banking assistant.
你在为银行助手生成合成训练数据。All people, accounts and requests are fictional.
所有人物、账户和请求均为虚构。

Rewrite the seed request below into $n DIFFERENT utterances that this persona would naturally use for the SAME request.
把下面的种子请求改写成 $n 条不同说法，表达同一个请求；要像这个人设自然会说出来的话。

Persona style / 人设风格:
$style

Channel / 渠道: $channel
$voice_rule

Output language / 输出语言: $lang_name
- zh = Simplified Chinese only / 只用简体中文
- en = English only / 只用英文
- mixed = natural Chinese-English code-switching (中英夹杂) / 自然的中英夹杂

Rules / 规则:
- Same meaning, but vary sentence structure, word order, fillers, politeness and minor typos.
  意思不变，但句式、语序、语气词、礼貌程度和轻微错别字要多样化。
- MUST keep every string in must_keep EXACTLY as written (names, amounts, brands, dates).
  必须原样保留 must_keep 中的每一个字符串（姓名、金额、品牌、日期），一个字都不能改。
- Do NOT add new people, amounts, accounts, or extra requests.
  不要新增人物、金额、账户或额外请求。
- Do NOT flip negation or change the intent.
  不要反转否定，也不要改变意图。
- Each variant must be self-contained and sound like a real user request.
  每条都要独立完整，像真实用户说的话。

Seed request / 种子请求:
$seed_text

must_keep (verbatim) / 必须原样保留:
$must_keep

Return ONLY a JSON object, no markdown:
{"variants": ["...", "..."]}
Exactly $n strings. 只返回上述 JSON，不要解释。
"""

_LANG_NAMES = {
    "zh": "zh - Simplified Chinese / 简体中文",
    "en": "en - English / 英文",
    "mixed": "mixed - Chinese-English code-switching / 中英夹杂",
}
_VOICE_RULE = (
    "This persona often uses voice input: no punctuation, colloquial spoken style. "
    "/ 这个人设常用语音输入：不加标点，口语化。"
)
_TEXT_RULE = (
    "This persona types in the app: use a natural app-text style. "
    "/ 这个人设用文字输入：自然的 App 文字风格。"
)


class TeacherError(RuntimeError):
    """The teacher API could not produce usable variants."""


class TeacherUnavailable(TeacherError):
    """No API key (or otherwise unusable configuration) for the teacher."""


def cache_key(model: str, prompt: str) -> str:
    """Cache key for one full prompt under one model."""

    return hashlib.sha256((model + prompt).encode("utf-8")).hexdigest()


# Kept as a private alias for callers that treat helpers as module internals.
_cache_key = cache_key


def build_prompt(
    seed_text: str,
    persona: dict[str, Any],
    lang: str,
    n: int,
    must_keep: list[str],
) -> str:
    """Render the teacher prompt for one seed request."""

    channel = persona.get("channel") or "app_text"
    if isinstance(channel, str):
        channel_text = channel
        has_voice = channel == "voice"
    else:
        channel_text = ", ".join(str(item) for item in channel)
        has_voice = "voice" in channel

    return Template(PROMPT_TEMPLATE).substitute(
        n=n,
        style=persona.get("style") or "natural, ordinary",
        channel=channel_text,
        voice_rule=_VOICE_RULE if has_voice else _TEXT_RULE,
        lang_name=_LANG_NAMES.get(lang, lang),
        seed_text=seed_text,
        must_keep=json.dumps(list(must_keep), ensure_ascii=False),
    )


def _normalise(text: str) -> str:
    """Casefold and strip whitespace/punctuation for duplicate detection."""

    return re.sub(r"[\W_]+", "", text.casefold())


def _filter_variants(
    variants: list[Any], seed_text: str, must_keep: list[str]
) -> list[str]:
    """Code-side safety net around the teacher's output."""

    keep = [item for item in must_keep if isinstance(item, str) and item]
    seed_length = len(seed_text)
    seed_normalised = _normalise(seed_text)
    seen = {seed_normalised}
    survivors: list[str] = []

    for raw in variants:
        if not isinstance(raw, str):
            continue
        variant = raw.strip()
        if not variant:
            continue
        if len(variant) > MAX_VARIANT_CHARS or len(variant) > 3 * seed_length:
            continue
        if any(item not in variant for item in keep):
            continue
        if any(run not in seed_text for run in LONG_DIGITS.findall(variant)):
            continue
        normalised = _normalise(variant)
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        survivors.append(variant)

    return survivors


def _parse_variants(content: str) -> list[str]:
    """Extract ``{"variants": [...]}`` from a teacher response."""

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise TeacherError("teacher response did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        raise TeacherError("teacher response contained invalid JSON") from None
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list):
        raise TeacherError("teacher response JSON is missing a variants list")
    return [item for item in variants if isinstance(item, str)]


class TeacherClient:
    """Synchronous DeepSeek-compatible client with retries and a JSONL cache."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout: float = 60,
        cache_path: Path = DEFAULT_CACHE_PATH,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        key = (key or "").strip()
        if not key:
            raise TeacherUnavailable(
                "DEEPSEEK_API_KEY is not set; export it or pass api_key to TeacherClient"
            )

        model_name = model if model is not None else os.environ.get("DEEPSEEK_MODEL")
        self._api_key = key
        self.model = (model_name or "").strip() or DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: dict[str, list[str]] = {}
        self._load_cache()
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"TeacherClient(model={self.model!r}, base_url={self.base_url!r})"

    def __enter__(self) -> TeacherClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- cache ---------------------------------------------------------

    def _load_cache(self) -> None:
        try:
            with self.cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    key = row.get("key")
                    variants = row.get("variants")
                    if isinstance(key, str) and isinstance(variants, list):
                        self._cache[key] = [
                            item for item in variants if isinstance(item, str)
                        ]
        except FileNotFoundError:
            pass

    def _store_cache(self, key: str, variants: list[str]) -> None:
        line = json.dumps(
            {"key": key, "model": self.model, "variants": variants},
            ensure_ascii=False,
        )
        with self._lock:
            self._cache[key] = list(variants)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    # -- HTTP ----------------------------------------------------------

    def _complete(self, prompt: str) -> list[str]:
        key = cache_key(self.model, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException:
                if attempt == MAX_ATTEMPTS:
                    raise TeacherError(
                        f"teacher API timed out after {MAX_ATTEMPTS} attempts"
                    ) from None
                time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
                continue
            except httpx.TransportError:
                # Connection resets are common on a flaky network; retry them
                # like timeouts instead of aborting the whole build.
                if attempt == MAX_ATTEMPTS:
                    raise TeacherError("teacher API request failed") from None
                time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
                continue

            status = response.status_code
            if status == 429 or 500 <= status < 600:
                if attempt == MAX_ATTEMPTS:
                    raise TeacherError(
                        f"teacher API returned HTTP {status} after "
                        f"{MAX_ATTEMPTS} attempts"
                    )
                time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
                continue
            if not 200 <= status < 300:
                raise TeacherError(f"teacher API returned HTTP {status}")

            try:
                body = response.json()
                content = body["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                raise TeacherError("teacher API returned a malformed response") from None
            if not isinstance(content, str):
                raise TeacherError("teacher API returned a malformed response")

            variants = _parse_variants(content)
            self._store_cache(key, variants)
            return list(variants)

        # The loop always returns or raises; keep type checkers happy.
        raise TeacherError("teacher API failed")  # pragma: no cover

    # -- public API ----------------------------------------------------

    def paraphrase(
        self,
        seed_text: str,
        persona: dict[str, Any],
        lang: str,
        n: int = 6,
        must_keep: list[str] | None = None,
    ) -> list[str]:
        """Return up to ``n`` filtered variants of one seed request."""

        keep = list(must_keep or [])
        prompt = build_prompt(seed_text, persona, lang, n, keep)
        return _filter_variants(self._complete(prompt), seed_text, keep)

    def paraphrase_many(
        self, jobs: list[dict[str, Any]], workers: int = 8
    ) -> list[list[str]]:
        """Run independent paraphrase jobs concurrently, preserving input order."""

        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(self.paraphrase, **job) for job in jobs]
            return [future.result() for future in futures]

    def check(self) -> tuple[str, float]:
        """Make one tiny request and return ``(model, latency_seconds)``."""

        prompt = 'Return ONLY JSON: {"variants": ["ok"]}'
        started = time.perf_counter()
        self._complete(prompt)
        return self.model, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that DEEPSEEK_API_KEY works with one tiny request",
    )
    args = parser.parse_args(argv)

    if not args.check:
        parser.print_help()
        return 2

    configured = bool((os.environ.get("DEEPSEEK_API_KEY") or "").strip())
    print(f"DEEPSEEK_API_KEY: {'configured' if configured else 'not configured'}")
    if not configured:
        print("Set DEEPSEEK_API_KEY to run the check.")
        return 1

    try:
        client = TeacherClient()
        model, latency = client.check()
        client.close()
    except TeacherError as exc:
        print(f"teacher check failed: {exc}")
        return 1

    print(f"model: {model}")
    print(f"latency: {latency:.2f}s")
    print("teacher check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
