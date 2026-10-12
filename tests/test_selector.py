from __future__ import annotations

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import QueryEngineTool, RetrieverTool, ToolMetadata

from llama_index.wity import AuthenticationError, InternalServerError, WityClassifier, WitySelector, noul

from .conftest import FakeWity, choice_body


class Answer(CustomQueryEngine):
    """A query engine that always answers the same text."""

    text: str

    def custom_query(self, query_str: str) -> str:
        return self.text


class Finds(BaseRetriever):
    """A retriever that always finds one node with the same text."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=self.text), score=1.0)]


def engines() -> list[QueryEngineTool]:
    return [
        QueryEngineTool.from_defaults(Answer(text="from billing docs"), description="Billing and refunds"),
        QueryEngineTool.from_defaults(Answer(text="from api docs"), description="API reference"),
    ]


def test_router_query_engine_uses_wity_pick(fake: FakeWity) -> None:
    fake.json(choice_body("route", "1", {"0": 0.125, "1": 0.875}))
    router = RouterQueryEngine(
        selector=WitySelector(classifier=fake.classifier()), query_engine_tools=engines(), llm=MockLLM()
    )
    response = router.query("What does the /v1/decide endpoint return?")

    assert str(response) == "from api docs"
    sent = fake.bodies[0]
    assert sent["state"] == "What does the /v1/decide endpoint return?"
    assert sent["questions"]["route"]["criteria"] == {"0": "Billing and refunds", "1": "API reference"}
    # The probability is passed on exactly as Wity gave it.
    assert response.metadata is not None
    assert response.metadata["selector_result"].reason == "Wity picked this choice with probability 0.875."


async def test_router_query_engine_async(fake: FakeWity) -> None:
    fake.json(choice_body("route", "0", {"0": 0.75, "1": 0.25}))
    router = RouterQueryEngine(
        selector=WitySelector(classifier=fake.classifier()), query_engine_tools=engines(), llm=MockLLM()
    )
    response = await router.aquery("Why was I charged twice?")
    assert str(response) == "from billing docs"


def test_router_retriever(fake: FakeWity) -> None:
    fake.json(choice_body("route", "1", {"0": 0.25, "1": 0.75}))
    retriever = RouterRetriever(
        selector=WitySelector(classifier=fake.classifier()),
        retriever_tools=[
            RetrieverTool.from_defaults(Finds("billing"), description="Billing"),
            RetrieverTool.from_defaults(Finds("api"), description="API"),
        ],
        llm=MockLLM(),
    )
    nodes = retriever.retrieve("rate limits")
    assert [n.node.get_content() for n in nodes] == ["api"]


def test_custom_instructions(fake: FakeWity) -> None:
    fake.json(choice_body("route", "0", {"0": 0.5, "1": 0.5}))
    selector = WitySelector(instructions="Which docs answer this?", classifier=fake.classifier())
    selector.select(["Billing", "API"], "refund")
    assert fake.bodies[0]["questions"]["route"]["instructions"] == "Which docs answer this?"


def test_one_choice_needs_no_call(fake: FakeWity) -> None:
    result = WitySelector(classifier=fake.classifier()).select(["Billing"], "refund")
    assert result.ind == 0
    assert fake.requests == []


def test_wity_errors_are_raised(fake: FakeWity) -> None:
    fake.json({"error": {"message": "down"}}, status=500)
    selector = WitySelector(classifier=fake.classifier())
    with pytest.raises(InternalServerError):
        selector.select(["Billing", "API"], "refund")


def test_classifier_with_questions_is_refused(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="without questions"):
        WitySelector(classifier=fake.classifier(questions={"x": noul("x?")}))


def test_default_classifier_thinks_on_close_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_test")
    classifier = WitySelector().classifier
    assert isinstance(classifier, WityClassifier)
    assert classifier.reasoning == "auto"


async def test_async_one_choice_needs_no_call(fake: FakeWity) -> None:
    result = await WitySelector(classifier=fake.classifier()).aselect(["Billing"], "refund")
    assert result.ind == 0
    assert fake.requests == []


def test_no_choices_is_an_error(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="at least one choice"):
        WitySelector(classifier=fake.classifier()).select([], "refund")
    assert fake.requests == []


def test_choices_with_same_or_missing_names(fake: FakeWity) -> None:
    fake.json(choice_body("route", "2", {"0": 0.25, "1": 0.25, "2": 0.5}))
    choices = [
        ToolMetadata(name="docs", description="Billing docs"),
        ToolMetadata(name="docs", description="API docs"),
        ToolMetadata(description="Status page"),
    ]
    result = WitySelector(classifier=fake.classifier()).select(choices, "Is the API down?")
    assert result.ind == 2
    assert fake.bodies[0]["questions"]["route"]["criteria"] == {
        "0": "Billing docs",
        "1": "API docs",
        "2": "Status page",
    }


async def test_async_errors_are_raised(fake: FakeWity) -> None:
    fake.json({"error": {"message": "bad key"}}, status=401)
    with pytest.raises(AuthenticationError):
        await WitySelector(classifier=fake.classifier()).aselect(["Billing", "API"], "refund")


def test_query_bundle_sends_only_the_query_text(fake: FakeWity) -> None:
    fake.json(choice_body("route", "0", {"0": 0.75, "1": 0.25}))
    query = QueryBundle(query_str="refund", custom_embedding_strs=["refund policy"], embedding=[0.1, 0.2])
    WitySelector(classifier=fake.classifier()).select(["Billing", "API"], query)
    assert fake.bodies[0]["state"] == "refund"
