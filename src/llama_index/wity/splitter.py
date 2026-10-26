"""``WitySplitter``: a LlamaIndex node parser that cuts documents into whole items with Wity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import wity
from llama_index.core.node_parser import TextSplitter
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import BaseModel, PrivateAttr
from typing_extensions import override
from wity import SplitItem, SplitSpec

from ._clients import WityClients


class WitySplitter(TextSplitter):
    """Cut documents into whole items: emails in a thread, clauses in a contract, entries in a log.

    Each item becomes one node. Every item is an exact slice of the text, so each node's
    ``start_char_idx`` and ``end_char_idx`` point at it in the original document.

    Use it like any LlamaIndex node parser: ``get_nodes_from_documents``, or in an ``IngestionPipeline``.
    """

    split: str | SplitSpec | dict[str, Any] | None = None
    """What one item is: plain words like ``"one item per email"``, or a ``SplitSpec``."""
    skill: str | None = None
    """Instead of ``split``: a split skill's id or name."""

    _clients: WityClients = PrivateAttr()

    def __init__(
        self,
        *,
        split: str | SplitSpec | dict[str, Any] | None = None,
        skill: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = wity.DEFAULT_TIMEOUT,
        max_retries: int = wity.DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            split: What one item is, in plain words or as a ``SplitSpec``.
                Leave out for the API default, ``"one item per paragraph"``.
            skill: Instead of ``split``: a split skill's id or name. It splits the way its examples do.
            api_key: Defaults to the WITY_API_KEY environment variable.
            base_url: Defaults to WITY_BASE_URL, then the SDK's default address.
            timeout: Seconds per attempt. Split waits at least two minutes, even with a shorter timeout.
            max_retries: Retries on 408, 429, 5xx and network failures.
            http_client: Your own ``httpx.Client``, for example for a proxy.
            http_async_client: Your own ``httpx.AsyncClient`` for async parsing.
            kwargs: Node parser options, like ``include_metadata`` and ``include_prev_next_rel``.
        """
        if split is not None and skill is not None:
            raise ValueError("Give split or skill, not both. A split skill brings its own split.")
        super().__init__(split=split, skill=skill, **kwargs)  # type: ignore[call-arg]  # pydantic fields mypy can't see
        self._clients = WityClients(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        # LlamaIndex unpickles through __init__, which would drop the key and options passed in code.
        BaseModel.__setstate__(self, state)

    @classmethod
    def class_name(cls) -> str:
        return "WitySplitter"

    @override
    def split_text(self, text: str) -> list[str]:
        """The items of ``text``, as text."""
        return [item.text for item in self._split(text)]

    async def asplit_text(self, text: str) -> list[str]:
        """Async ``split_text``."""
        return [item.text for item in await self._asplit(text)]

    def _split(self, text: str) -> list[SplitItem]:
        return self._clients.sync.split(text=text, split=self.split, skill=self.skill).items

    async def _asplit(self, text: str) -> list[SplitItem]:
        response = await self._clients.for_this_loop().split(text=text, split=self.split, skill=self.skill)
        return response.items

    @override
    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> list[BaseNode]:
        parsed: list[BaseNode] = []
        for node in get_tqdm_iterable(nodes, show_progress, "Splitting with Wity"):
            parsed.extend(self._nodes(node, self._split(node.get_content())))
        return parsed

    @override
    async def _aparse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        parsed: list[BaseNode] = []
        for node in nodes:
            parsed.extend(self._nodes(node, await self._asplit(node.get_content())))
        return parsed

    def _nodes(self, node: BaseNode, items: list[SplitItem]) -> list[TextNode]:
        built = build_nodes_from_splits([item.text for item in items], node, id_func=self.id_func)
        for new, item in zip(built, items, strict=True):
            new.start_char_idx = item.start
            new.end_char_idx = item.end
        return built

    def close(self) -> None:
        """Close the connections sync parsing uses. Clients you passed in stay open."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close the connections async parsing uses in this event loop. Clients you passed in stay open."""
        await self._clients.aclose()
