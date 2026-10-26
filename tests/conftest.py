"""Shared test helpers: a fake Wity server on a fake network, and a fake LLM for agents."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.llms import MockFunctionCallingLLM

from llama_index.wity import WityClassifier, WityExtractor, WitySplitter

TEST_KEY = "wity_test_secret_1234567890"
"""A made-up key. Tests check that it never leaks."""

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_body(name: str) -> dict[str, Any]:
    """The body of a real API response, saved in tests/fixtures/."""
    saved: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    body: dict[str, Any] = saved["body"]
    return body


def choice_body(question: str, pick: str, probabilities: dict[str, float]) -> dict[str, Any]:
    """A choice answer in the API's shape."""
    answer = {
        "type": "choice",
        "choice": pick,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
    }
    return {**fixture_body("choice"), "answers": {question: answer}}


def noul_body(question: str, probability: float) -> dict[str, Any]:
    """A noul answer in the API's shape."""
    return {**fixture_body("noul"), "answers": {question: {"type": "noul", "noul": probability}}}


Reply = httpx.Response | Callable[[httpx.Request], httpx.Response] | Exception


@dataclass
class FakeWity:
    """Answers every request with the next queued reply, or ``default``. Records what was sent."""

    default: Reply = field(default_factory=lambda: httpx.Response(200, json=fixture_body("noul")))
    queue: list[Reply] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def json(self, body: dict[str, Any], status: int = 200) -> None:
        self.queue.append(httpx.Response(status, json=body))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.queue.pop(0) if self.queue else self.default
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return reply(request)

    def classifier(self, **options: Any) -> WityClassifier:
        """A classifier wired to this fake server. No retries unless asked for."""
        transport = httpx.MockTransport(self._handle)
        options.setdefault("api_key", TEST_KEY)
        options.setdefault("max_retries", 0)
        return WityClassifier(
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            **options,
        )

    def splitter(self, **options: Any) -> WitySplitter:
        """A splitter wired to this fake server. No retries unless asked for."""
        transport = httpx.MockTransport(self._handle)
        options.setdefault("api_key", TEST_KEY)
        options.setdefault("max_retries", 0)
        return WitySplitter(
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            **options,
        )

    def extractor(self, **options: Any) -> WityExtractor:
        """An extractor wired to this fake server, with no progress bar."""
        transport = httpx.MockTransport(self._handle)
        options.setdefault("api_key", TEST_KEY)
        options.setdefault("show_progress", False)
        return WityExtractor(
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            **options,
        )


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never see the developer's real key or address."""
    for name in ("WITY_API_KEY", "WITY_BASE_URL", "WITY_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    yield


Step = list[tuple[str, dict[str, Any]]]
"""The tool calls (name, args) the fake LLM makes in one turn."""


def scripted_llm(*steps: Step, final: str = "Done.") -> MockFunctionCallingLLM:
    """A fake LLM that makes each step's tool calls in turn, then answers ``final``."""

    def reply(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        done = sum(1 for m in messages if any(isinstance(b, ToolCallBlock) for b in m.blocks))
        if done >= len(steps):
            return ChatMessage(role=MessageRole.ASSISTANT, content=final)
        calls = [
            ToolCallBlock(tool_call_id=f"call_{done}_{i}", tool_name=name, tool_kwargs=args)
            for i, (name, args) in enumerate(steps[done])
        ]
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=calls)

    return MockFunctionCallingLLM(response_generator=reply)


def calling_llm(tool_name: str, args: dict[str, Any], final: str = "Done.") -> MockFunctionCallingLLM:
    """A fake LLM that asks to run one tool, then answers ``final`` once it sees the tool's result."""
    return scripted_llm([(tool_name, args)], final=final)
