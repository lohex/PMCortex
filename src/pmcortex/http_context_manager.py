"""Shared lifecycle management for synchronous HTTP clients."""

from types import TracebackType
from typing import Self

import httpx


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
