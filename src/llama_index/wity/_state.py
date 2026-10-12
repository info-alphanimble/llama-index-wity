"""Turn state that holds LlamaIndex chat messages into plain JSON for Wity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, TextBlock, ToolCallBlock


def to_wity_state(state: Any) -> Any:
    """Return ``state`` with every ``ChatMessage`` turned into ``{role, content}``, at the top or nested.

    Tool calls in a message are kept as ``tool_calls``. Images, audio and files are sent by type only,
    like ``{"type": "image"}``: sent as text, a small image is thousands of tokens Wity can't read.
    """
    if isinstance(state, ChatMessage):
        return _message(state)
    if isinstance(state, Mapping):
        return {key: to_wity_state(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return [to_wity_state(item) for item in state]
    return state


def _message(message: ChatMessage) -> dict[str, Any]:
    text = [block.text for block in message.blocks if isinstance(block, TextBlock)]
    tool_calls = [
        {"name": block.tool_name, "args": block.tool_kwargs}
        for block in message.blocks
        if isinstance(block, ToolCallBlock)
    ]
    other = [
        {"type": block.block_type} for block in message.blocks if not isinstance(block, (TextBlock, ToolCallBlock))
    ]
    out: dict[str, Any] = {"role": message.role.value, "content": "\n".join(text)}
    if tool_calls:
        out["tool_calls"] = tool_calls
    if other:
        out["attachments"] = other
    return out
