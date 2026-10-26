"""``WityExtractor``: a LlamaIndex metadata extractor that pulls named fields out of each node with Wity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import wity
from llama_index.core.async_utils import run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, PrivateAttr
from typing_extensions import override
from wity import ExtractType

from ._clients import WityClients


class WityExtractor(BaseExtractor):
    """Pull named fields out of each node and add them to its metadata.

    ``WityExtractor(fields={"party": "Who signs the contract"})`` adds ``metadata["party"]``
    to each node: the values found, as written and joined with ``"; "``, or ``""`` when there are none.
    It reads the node's own text, without its metadata.

    Use it in an ``IngestionPipeline``, or call ``extract(nodes)`` for the metadata alone.
    """

    fields: dict[str, str]
    """1 to 16 fields: a name you choose, to what it asks for in plain words. Each name becomes a metadata key."""
    types: dict[str, ExtractType] | None = None
    """Field name to ``"date"``, ``"duration"`` or ``"number"``, for values that should come back normalised."""

    _clients: WityClients = PrivateAttr()

    def __init__(
        self,
        *,
        fields: dict[str, str],
        types: dict[str, ExtractType] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = wity.DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            fields: 1 to 16 fields: name to what it asks for, in plain words.
            types: Field name to ``"date"`` (YYYY-MM-DD), ``"duration"`` or ``"number"``.
            api_key: Defaults to the WITY_API_KEY environment variable.
            base_url: Defaults to WITY_BASE_URL, then the SDK's default address.
            timeout: Seconds per attempt. Extract waits at least two minutes, even with a shorter timeout.
            http_client: Your own ``httpx.Client``, for example for a proxy.
            http_async_client: Your own ``httpx.AsyncClient``.
            kwargs: Extractor options, like ``num_workers`` (requests at once) and ``show_progress``.
                Each request is already retried on 408, 429, 5xx and network failures. LlamaIndex's
                ``max_retries`` runs the whole batch again, so leave it at 0 unless you want that.
        """
        if not fields:
            raise ValueError("fields is empty. Ask for at least one field.")
        super().__init__(fields=fields, types=types, **kwargs)  # type: ignore[call-arg]  # pydantic fields mypy can't see
        self._clients = WityClients(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=wity.DEFAULT_MAX_RETRIES,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        # LlamaIndex unpickles through __init__, which would drop the key and options passed in code.
        BaseModel.__setstate__(self, state)

    @classmethod
    def class_name(cls) -> str:
        return "WityExtractor"

    @override
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        jobs = [self._extract_one(node) for node in nodes]
        results: list[dict[str, Any]] = await run_jobs(
            jobs, show_progress=self.show_progress, workers=self.num_workers, desc="Extracting with Wity"
        )
        return results

    async def _extract_one(self, node: BaseNode) -> dict[str, Any]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        response = await self._clients.for_this_loop().extract(text=text, fields=self.fields, types=self.types)
        # Text, not a list: many vector stores, like Chroma, only take text and numbers as metadata.
        return {name: "; ".join(values) for name, values in response.fields.items()}

    def close(self) -> None:
        """Close the blocking connections. Clients you passed in stay open."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close the connections used in this event loop. Clients you passed in stay open."""
        await self._clients.aclose()
