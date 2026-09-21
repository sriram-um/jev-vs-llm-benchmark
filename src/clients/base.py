"""Evaluator interface plus the bits every evaluator shares."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

from src.schemas import EvalRecord, Scenario


class Evaluator(ABC):
    """One system under test.

    Implementations own their own transport and are responsible for producing an
    `EvalRecord` even when the request fails -- a failed request is data, not an
    exception to swallow at the runner level.
    """

    #: Short stable key used in summaries, filenames, and plot legends.
    name: str

    @abstractmethod
    async def evaluate(self, scenario: Scenario) -> EvalRecord:
        ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

    async def __aenter__(self) -> "Evaluator":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def cost_usd(input_tokens: int, output_tokens: int, price: dict[str, float]) -> float:
    """Cost of one request given a `PRICES_USD_PER_MTOK` entry."""
    return (
        input_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"]
    )


def token_counter() -> Callable[[str], int]:
    """An approximate tokenizer, for sizing SIMULATED requests only.

    Uses tiktoken's `o200k_base`, falling back to the ~4-chars-per-token rule of
    thumb if tiktoken is unavailable. Both simulators share it so their token
    counts -- and therefore their simulated costs -- are measured the same way.

    Live paths never call this: they use the `usage` the API actually reports.
    Note that it is the wrong tokenizer for at least one of the two systems; it
    keeps simulated cost dimensionally sane, and is not a billing figure.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda text: len(enc.encode(text))
    except Exception:  # pragma: no cover - tiktoken is optional at runtime
        return lambda text: max(1, len(text) // 4)


class Stopwatch:
    """Times only the network call, using a monotonic clock.

    `time.perf_counter` is monotonic and has nanosecond resolution, which
    matters when the thing being measured is ~100ms and the claim being tested
    is about latency.
    """

    __slots__ = ("_start", "elapsed_ms")

    def __enter__(self) -> "Stopwatch":
        self.elapsed_ms = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class RateLimiter:
    """Sliding-window limiter, so we stay under a documented requests/minute cap.

    Jev's documented ceiling is 1200 req/min; exceeding it earns a 429 whose
    retry would pollute the latency distribution we are trying to measure.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(1, per_minute)
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._hits and now - self._hits[0] >= 60.0:
                    self._hits.popleft()
                if len(self._hits) < self.per_minute:
                    self._hits.append(now)
                    return
                sleep_for = 60.0 - (now - self._hits[0])
            await asyncio.sleep(max(sleep_for, 0.01))
