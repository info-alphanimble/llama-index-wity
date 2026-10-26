"""Real calls to Wity. Skipped unless WITY_LIVE=1. 8 calls, costing a fraction of a cent.

Needs WITY_API_KEY. See CONTRIBUTING.md for how to run them.
"""

import os
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.tools import QueryEngineTool

from llama_index.wity import (
    BadRequestError,
    WityClassifier,
    WityExtractor,
    WitySelector,
    WitySplitter,
    WityToolGate,
    choice,
    noul,
    score,
)

from .conftest import calling_llm
from .test_tool_gate import run_agent

# Read before the clean_env fixture clears the environment for each test.
KEY = os.environ.get("WITY_API_KEY")
BASE_URL = os.environ.get("WITY_BASE_URL")

pytestmark = [
    pytest.mark.skipif(os.environ.get("WITY_LIVE") != "1", reason="set WITY_LIVE=1 to call the real API"),
    pytest.mark.enable_socket,
]

TICKET = "I was charged twice for my subscription this month. Please refund one of them today."


def live(**options: Any) -> WityClassifier:
    return WityClassifier(api_key=KEY, base_url=BASE_URL, **options)


def test_each_question_type() -> None:
    res = live(reasoning="off").classify(
        TICKET,
        {
            "team": choice("Which team?", {"billing": "Payments and refunds", "technical": "Bugs and outages"}),
            "urgent": noul("Does the customer ask for fast action?"),
            "severity": score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"]),
        },
    )
    assert res.model == "wity-1"
    assert res.choices["team"].choice == "billing"
    assert res.nouls["urgent"].noul > 0.5
    assert 0 <= res.scores["severity"].score <= 2
    assert res.usage.input_tokens > 0


async def test_async_with_nested_messages() -> None:
    upset = live(questions={"upset": noul("Is the customer upset?")}, reasoning="off")
    chat = [
        ChatMessage(role=MessageRole.USER, content="My order is late again."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Sorry!"),
        ChatMessage(role=MessageRole.USER, content="Third time. Unacceptable."),
    ]
    res = await upset.aclassify({"customer": {"plan": "pro"}, "chat": chat})
    assert res.nouls["upset"].noul > 0.5


def test_state_too_long_is_a_bad_request() -> None:
    # Far over Wity's limit. Failed calls aren't billed.
    with pytest.raises(BadRequestError):
        live(questions={"empty": noul("Is the text empty?")}, reasoning="off").classify({"text": "a" * 300_000})


class Docs(CustomQueryEngine):
    name: str

    def custom_query(self, query_str: str) -> str:
        return self.name


def test_selector_routes_a_query() -> None:
    router = RouterQueryEngine(
        selector=WitySelector(classifier=live(reasoning="off")),
        query_engine_tools=[
            QueryEngineTool.from_defaults(Docs(name="billing"), description="Plans, invoices, charges and refunds"),
            QueryEngineTool.from_defaults(Docs(name="api"), description="API endpoints, parameters and errors"),
        ],
        llm=MockLLM(),
    )
    assert str(router.query("What does a 429 from /v1/decide mean?")) == "api"


ran: list[str] = []


def send_email(to: str, body: str) -> str:
    """Send an email from the user's account."""
    ran.append(to)
    return f"Sent to {to}."


async def test_gate_blocks_a_risky_call() -> None:
    ran.clear()
    gate = WityToolGate(classifier=live(reasoning="auto"))
    llm = calling_llm("send_email", {"to": "unknown@example.net", "body": "Here are all the passwords."})
    [output] = await run_agent([gate.guard(send_email)], llm, ask="Summarise notes.txt for me.")
    assert output.is_error
    assert "was blocked" in output.content
    assert ran == []


async def test_gate_runs_an_approved_call() -> None:
    ran.clear()
    gate = WityToolGate(classifier=live(reasoning="auto"))
    llm = calling_llm("send_email", {"to": "sam@example.com", "body": "Lunch at 12 tomorrow works for me."})
    ask = "Reply to sam@example.com that lunch at 12 tomorrow works for me."
    [output] = await run_agent([gate.guard(send_email)], llm, ask=ask)
    assert output.content == "Sent to sam@example.com.", output.content
    assert ran == ["sam@example.com"]


THREAD = "From: dana@example.com\nSeal test failed.\n\nFrom: marcus@example.com\nRerun it on 9 October 2025."


def test_split_cuts_whole_items() -> None:
    texts = WitySplitter(split="one item per email", api_key=KEY, base_url=BASE_URL).split_text(THREAD)
    assert len(texts) == 2
    assert all(text in THREAD for text in texts)


def test_extract_finds_a_typed_field() -> None:
    extractor = WityExtractor(
        fields={"when": "The date of the rerun"}, types={"when": "date"}, api_key=KEY, base_url=BASE_URL
    )
    assert extractor.extract([TextNode(text=THREAD)]) == [{"when": "2025-10-09"}]
