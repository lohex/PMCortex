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
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
