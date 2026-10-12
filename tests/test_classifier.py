from __future__ import annotations

import asyncio
import datetime

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock, ToolCallBlock
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler

from llama_index.wity import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    WityClassifier,
    WityError,
    choice,
    noul,
    score,
)

from .conftest import TEST_KEY, FakeWity, fixture_body

URGENT = {"urgent": noul("Is this urgent?")}


def test_questions_in_constructor(fake: FakeWity) -> None:
    res = fake.classifier(questions=URGENT).classify("Refund me today.")
    assert res.nouls["urgent"].noul == fixture_body("noul")["answers"]["urgent"]["noul"]
    assert fake.bodies[0]["state"] == "Refund me today."
    assert fake.bodies[0]["questions"]["urgent"]["type"] == "noul"


def test_questions_with_each_call(fake: FakeWity) -> None:
    fake.json(fixture_body("choice"))
    question = {"team": choice("Which team?", {"billing": "Payments", "technical": "Bugs"})}
    fake.classifier().classify({"ticket": "Charged twice"}, question)
    assert fake.bodies[0]["state"] == {"ticket": "Charged twice"}
    assert fake.bodies[0]["questions"]["team"]["criteria"] == {"billing": "Payments", "technical": "Bugs"}


def test_score_question(fake: FakeWity) -> None:
    fake.json(fixture_body("score"))
    fake.classifier().classify("text", {"severity": score("How bad?", ["Low", "Mid", "High"])})
    assert fake.bodies[0]["questions"]["severity"]["criteria"] == ["Low", "Mid", "High"]


def test_questions_in_both_places_is_an_error(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="has its own questions"):
        fake.classifier(questions=URGENT).classify("text", URGENT)
    assert fake.requests == []


def test_no_questions_is_an_error(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="Pass questions"):
        fake.classifier().classify("text")
    with pytest.raises(ValueError, match="Pass questions"):
        fake.classifier().classify("text", {})


async def test_async(fake: FakeWity) -> None:
    res = await fake.classifier(questions=URGENT).aclassify("Refund me today.")
    assert res.nouls["urgent"].noul > 0.5
    assert fake.bodies[0]["state"] == "Refund me today."


def test_chat_messages_become_role_and_content(fake: FakeWity) -> None:
    chat = [
        ChatMessage(role=MessageRole.USER, content="My order is late."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Sorry!"),
    ]
    fake.classifier(questions=URGENT).classify(chat)
    assert fake.bodies[0]["state"] == [
        {"role": "user", "content": "My order is late."},
        {"role": "assistant", "content": "Sorry!"},
    ]


def test_nested_messages_tool_calls_and_images(fake: FakeWity) -> None:
    call = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[ToolCallBlock(tool_call_id="c1", tool_name="lookup", tool_kwargs={"id": 7})],
    )
    photo = ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text="See this"), ImageBlock(url="https://example.com/a.png")],
    )
    fake.classifier(questions=URGENT).classify({"customer": {"plan": "pro"}, "chat": (call, photo)})
    assert fake.bodies[0]["state"] == {
        "customer": {"plan": "pro"},
        "chat": [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "args": {"id": 7}}]},
            {"role": "user", "content": "See this", "attachments": [{"type": "image"}]},
        ],
    }


def test_options_reach_the_api(fake: FakeWity) -> None:
    fake.classifier(questions=URGENT, reasoning="off", max_latency_ms=500, base_url="https://wity.test").classify("x")
    request = fake.requests[0]
    assert request.url.host == "wity.test"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert fake.bodies[0]["reasoning"] == "off"
    assert fake.bodies[0]["max_latency_ms"] == 500


def test_key_from_environment(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env")
    transport = httpx.MockTransport(fake._handle)
    WityClassifier(questions=URGENT, http_client=httpx.Client(transport=transport)).classify("x")
    assert fake.requests[0].headers["authorization"] == "Bearer wity_from_env"


def test_key_never_in_repr(fake: FakeWity) -> None:
    classifier = fake.classifier(questions=URGENT)
    assert TEST_KEY not in repr(classifier)
    assert TEST_KEY not in str(vars(classifier).get("questions"))


def test_errors_are_the_sdk_errors(fake: FakeWity) -> None:
    fake.json({"error": {"message": "bad key"}}, status=401)
    with pytest.raises(AuthenticationError) as err:
        fake.classifier(questions=URGENT).classify("secret state text")
    assert TEST_KEY not in str(err.value)
    assert "secret state text" not in str(err.value)

    fake.json({"error": {"message": "state too long"}}, status=400)
    with pytest.raises(BadRequestError):
        fake.classifier(questions=URGENT).classify("x")


def test_each_call_is_a_span(fake: FakeWity) -> None:
    handler = SimpleSpanHandler()
    dispatcher = get_dispatcher("llama_index.wity.classifier")
    root = dispatcher.root
    root.add_span_handler(handler)
    try:
        fake.classifier(questions=URGENT).classify("x")
    finally:
        root.span_handlers.remove(handler)
    names = [span.id_ for span in handler.completed_spans]
    assert any("WityClassifier.classify" in name for name in names)


def test_empty_questions_in_constructor_is_an_error() -> None:
    with pytest.raises(ValueError, match="questions is empty"):
        WityClassifier(api_key=TEST_KEY, questions={})


def test_missing_key_fails_when_created() -> None:
    with pytest.raises(WityError):
        WityClassifier()


def test_roles_given_as_text_and_several_text_blocks(fake: FakeWity) -> None:
    message = ChatMessage(role="tool", blocks=[TextBlock(text="line 1"), TextBlock(text="line 2")])
    fake.classifier(questions=URGENT).classify(message)
    assert fake.bodies[0]["state"] == {"role": "tool", "content": "line 1\nline 2"}


def test_state_that_isnt_json_fails_before_sending(fake: FakeWity) -> None:
    with pytest.raises(WityError) as err:
        fake.classifier(questions=URGENT).classify({"when": datetime.date(2026, 1, 2), "note": "private note"})
    assert fake.requests == []
    assert "private note" not in str(err.value)


def test_retries_use_the_option(fake: FakeWity) -> None:
    fake.json({"error": {"message": "busy"}}, status=503)
    res = fake.classifier(questions=URGENT, max_retries=1).classify("x")
    assert res.nouls["urgent"].noul > 0.5
    assert len(fake.requests) == 2


def test_network_errors_are_sdk_errors(fake: FakeWity) -> None:
    fake.queue.append(httpx.ConnectError("no route"))
    with pytest.raises(APIConnectionError):
        fake.classifier(questions=URGENT).classify("x")
    fake.queue.append(httpx.ReadTimeout("slow"))
    with pytest.raises(APITimeoutError):
        fake.classifier(questions=URGENT).classify("x")


async def test_async_errors_are_sdk_errors(fake: FakeWity) -> None:
    fake.json({"error": {"message": "slow down"}}, status=429)
    with pytest.raises(RateLimitError):
        await fake.classifier(questions=URGENT).aclassify("x")


def test_works_from_several_event_loops(fake: FakeWity) -> None:
    classifier = fake.classifier(questions=URGENT)
    asyncio.run(classifier.aclassify("first"))
    asyncio.run(classifier.aclassify("second"))
    assert len(fake.requests) == 2
    # Clients of closed loops aren't kept.
    assert len(classifier._async_clients) <= 1


async def test_close_keeps_your_own_clients_open() -> None:
    mine = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    mine_async = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    classifier = WityClassifier(api_key=TEST_KEY, http_client=mine, http_async_client=mine_async)
    classifier.close()
    await classifier.aclose()
    assert not mine.is_closed
    assert not mine_async.is_closed


async def test_async_call_is_a_span(fake: FakeWity) -> None:
    handler = SimpleSpanHandler()
    root = get_dispatcher("llama_index.wity.classifier").root
    root.add_span_handler(handler)
    try:
        await fake.classifier(questions=URGENT).aclassify("x")
    finally:
        root.span_handlers.remove(handler)
    assert any("WityClassifier.aclassify" in span.id_ for span in handler.completed_spans)
