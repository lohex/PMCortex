"""Shared lifecycle management for synchronous HTTP clients."""

from collections.abc import Callable
import time
from types import TracebackType
from typing import Self

import httpx


class RequestRateLimiter:
    """Space synchronous request starts by a configured minimum interval."""

    def __init__(
        self,
        max_requests_per_second: float,
        *,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        """
        Configure the maximum request-start rate.

        Args:
            max_requests_per_second: Positive upper bound for request starts.
            time_fn: Monotonic clock, injectable for deterministic tests.
            sleep_fn: Blocking sleep function, injectable for deterministic tests.

        Raises:
            ValueError: If ``max_requests_per_second`` is not positive.

        Notes:
            This limiter is intended for sequential clients and is not thread-safe.
        """
        if max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be positive")

        self._min_interval_s = 1.0 / max_requests_per_second
        self._time_fn = time.monotonic if time_fn is None else time_fn
        self._sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self._next_request_not_before = 0.0

    def wait(self) -> None:
        """Block until the next request may start and reserve that slot."""
        now = self._time_fn()
        if now < self._next_request_not_before:
            self._sleep_fn(self._next_request_not_before - now)
            now = self._time_fn()

        self._next_request_not_before = now + self._min_interval_s


class HTTPContextManager:
    """Own a synchronous HTTP client and expose context-manager cleanup."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float = 30.0,
        accept: str = "*/*;q=0.1",
    ) -> None:
        """Create an HTTP client with the supplied request defaults."""
        if not user_agent.strip():
            raise ValueError("user_agent must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": accept,
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> Self:
        """Return this manager for use in a `with` statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the client when leaving a `with` statement."""
        self.close()
