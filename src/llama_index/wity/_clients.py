"""The Wity clients a splitter or extractor uses: one blocking, and one async per event loop."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any

import httpx
import wity

from ._update_notice import show_update_notice_once


class WityClients:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
        http_client: httpx.Client | None,
        http_async_client: httpx.AsyncClient | None,
    ) -> None:
        # Kept for the async clients, made later, one per event loop. Not in repr().
        self._options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        self._http_client = http_client
        self._http_async_client = http_async_client
        hide_sdk_notice()
        # Made here, so a missing key or a bad address fails now, not at the first call.
        self.sync = wity.WityClient(**self._options, http_client=http_client)
        show_update_notice_once("llama-index-wity", "pip install -U llama-index-wity")
        # One async client per event loop: a connection opened in one loop can't be used from another.
        self._async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, wity.AsyncWityClient] = (
            weakref.WeakKeyDictionary()
        )

    def __getstate__(self) -> dict[str, Any]:
        # Pickle the options, not the open connections. A worker process makes its own.
        # Clients you passed in can't be pickled, so a worker uses its own connections instead.
        return {"options": self._options}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Like __init__, but without the update notice, so a pool of workers doesn't print one each.
        self._options = state["options"]
        self._http_client = None
        self._http_async_client = None
        hide_sdk_notice()
        self.sync = wity.WityClient(**self._options)
        self._async_clients = weakref.WeakKeyDictionary()

    def __deepcopy__(self, memo: dict[int, Any]) -> WityClients:
        # A copy gets its own connections. Clients you passed in are shared, not copied.
        return WityClients(**self._options, http_client=self._http_client, http_async_client=self._http_async_client)

    def for_this_loop(self) -> wity.AsyncWityClient:
        """The async client for the running event loop, made on first use in that loop."""
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            # A client holds its loop through its open connections, so the weak key alone never lets a
            # closed loop go. Drop those clients here, or each `asyncio.run` would leave a socket open.
            for old_loop in [old for old in self._async_clients if old.is_closed()]:
                del self._async_clients[old_loop]
            client = wity.AsyncWityClient(**self._options, http_client=self._http_async_client)
            self._async_clients[loop] = client
        return client

    def close(self) -> None:
        self.sync.close()

    async def aclose(self) -> None:
        client = self._async_clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.close()


def hide_sdk_notice() -> None:
    """Keep wity-sdk's own notice quiet. It says to upgrade wity-sdk alone, which could pass the version we pin."""
    try:
        from wity._update_notice import _shown  # type: ignore[import-not-found,unused-ignore]  # wity-sdk 0.2.1+
    except ImportError:
        return
    _shown.add("wity-sdk")
