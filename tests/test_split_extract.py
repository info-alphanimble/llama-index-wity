"""WitySplitter and WityExtractor: nodes, offsets, metadata, pipelines, options and errors."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import httpx
import pytest
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from llama_index.wity import AuthenticationError, BadRequestError, SplitSpec, WityExtractor, WitySplitter

from .conftest import TEST_KEY, FakeWity

EMAIL = "From: dana@example.com\nSeal test failed.\n\nFrom: marcus@example.com\nRerun tomorrow."
SPLIT_REPLY: dict[str, Any] = {
    "items": [
        {"start": 0, "end": 40, "text": EMAIL[0:40]},
        {"start": 42, "end": len(EMAIL), "text": EMAIL[42:]},
    ],
    "how": "model",
    "timing_ms": {"total": 84},
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
EXTRACT_REPLY: dict[str, Any] = {
    "fields": {"sender": ["dana@example.com"], "when": []},
    "entities": [{"field": "sender", "text": "dana@example.com", "start": 6, "end": 22, "confidence": 0.98}],
    "timing_ms": 503,
    "usage": {"input_tokens": 301, "output_tokens": 0},
}
FIELDS = {"sender": "Who the email is from", "when": "A date the email mentions"}


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity(default=httpx.Response(200, json=SPLIT_REPLY))


# ---------------------------------------------------------------------------
# WitySplitter
# ---------------------------------------------------------------------------


def test_split_text_returns_the_items(fake: FakeWity) -> None:
    assert fake.splitter(split="one item per email").split_text(EMAIL) == [EMAIL[0:40], EMAIL[42:]]
    assert fake.requests[0].url.path == "/v1/split"
    assert fake.bodies[0] == {"text": EMAIL, "split": "one item per email"}


def test_nodes_point_at_their_place_in_the_document(fake: FakeWity) -> None:
    doc = Document(text=EMAIL, metadata={"source": "thread-1"})
    nodes = fake.splitter().get_nodes_from_documents([doc])
    assert len(nodes) == 2
    for node in nodes:
        assert isinstance(node, TextNode)
        assert node.start_char_idx is not None
        assert EMAIL[node.start_char_idx : node.end_char_idx] == node.text
        assert node.metadata == {"source": "thread-1"}
        assert node.ref_doc_id == doc.doc_id
    next_node = nodes[0].next_node
    assert next_node is not None
    assert next_node.node_id == nodes[1].node_id


def test_offsets_are_set_for_nodes_that_arent_documents(fake: FakeWity) -> None:
    nodes = fake.splitter()([TextNode(text=EMAIL)])
    assert [(n.start_char_idx, n.end_char_idx) for n in nodes] == [(0, 40), (42, len(EMAIL))]  # type: ignore[attr-defined]


async def test_async_parsing(fake: FakeWity) -> None:
    splitter = fake.splitter()
    nodes = await splitter.aget_nodes_from_documents([Document(text=EMAIL)])
    assert [node.get_content() for node in nodes] == [EMAIL[0:40], EMAIL[42:]]
    assert await splitter.asplit_text(EMAIL) == [EMAIL[0:40], EMAIL[42:]]
    await splitter.aclose()


def test_a_split_spec_and_a_skill(fake: FakeWity) -> None:
    fake.splitter(split=SplitSpec(unit="email", starts_at=["From:"])).split_text(EMAIL)
    fake.splitter(skill="emails").split_text(EMAIL)
    assert fake.bodies[0]["split"] == {"unit": "email", "starts_at": ["From:"]}
    assert fake.bodies[1] == {"text": EMAIL, "skill": "emails"}


def test_the_api_default_split_is_left_to_the_api(fake: FakeWity) -> None:
    fake.splitter().split_text(EMAIL)
    assert fake.bodies[0] == {"text": EMAIL}


def test_split_and_skill_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        WitySplitter(split="one item per email", skill="emails", api_key=TEST_KEY)


def test_split_errors_are_sdk_errors(fake: FakeWity) -> None:
    fake.json({"error": "text must be 1 to 60000 characters"}, status=400)
    with pytest.raises(BadRequestError):
        fake.splitter().split_text(EMAIL)


def test_splitter_pickles_and_reads_the_key_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", TEST_KEY)
    splitter = pickle.loads(pickle.dumps(WitySplitter(split="one item per email")))
    assert splitter.split == "one item per email"
    assert TEST_KEY not in repr(WitySplitter(api_key=TEST_KEY))


@pytest.mark.parametrize(
    "make",
    [
        lambda **options: WitySplitter(split="one item per email", **options),
        lambda **options: WityExtractor(fields=FIELDS, **options),
    ],
    ids=["splitter", "extractor"],
)
def test_a_pickled_copy_keeps_the_options_passed_in_code(make: Any) -> None:
    # What IngestionPipeline(num_workers=...) does to send a step to a worker process. No WITY_API_KEY is set.
    # Unpickling makes a new client, which would raise if the key passed in code were lost.
    original = make(api_key=TEST_KEY, base_url="https://wity.example.com", timeout=30.0)
    copy_ = pickle.loads(pickle.dumps(original))
    client = copy_._clients.sync
    assert (client.base_url, client.timeout) == ("https://wity.example.com", 30.0)
    assert client is not original._clients.sync
    assert copy_.class_name() == original.class_name()


def test_a_pickled_copy_makes_calls_with_its_own_connections(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[httpx.Request] = []

    def own_connection(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=SPLIT_REPLY)

    copy_ = pickle.loads(pickle.dumps(fake.splitter()))
    # The network, as a new connection sees it. The fake's own transport is a different class.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", own_connection)
    assert copy_.split_text(EMAIL) == [EMAIL[0:40], EMAIL[42:]]
    assert sent[0].headers["authorization"] == f"Bearer {TEST_KEY}"
    assert fake.requests == []  # The client passed in stayed behind.


def test_a_deep_copy_shares_the_clients_passed_in(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    splitter = copy.deepcopy(fake.splitter())
    extractor = copy.deepcopy(fake.extractor(fields=FIELDS))
    assert extractor.extract([TextNode(text=EMAIL)]) == [{"sender": "dana@example.com", "when": ""}]
    assert splitter.split_text(EMAIL) == [EMAIL[0:40], EMAIL[42:]]
    assert [request.url.path for request in fake.requests] == ["/v1/extract", "/v1/split"]


# ---------------------------------------------------------------------------
# WityExtractor
# ---------------------------------------------------------------------------


def test_extract_adds_fields_to_each_nodes_metadata(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    nodes = fake.extractor(fields=FIELDS, types={"when": "date"})([TextNode(text=EMAIL, metadata={"n": 1})])
    assert nodes[0].metadata == {"n": 1, "sender": "dana@example.com", "when": ""}
    assert fake.requests[0].url.path == "/v1/extract"
    assert fake.bodies[0] == {"text": EMAIL, "fields": FIELDS, "types": {"when": "date"}}


def test_several_values_are_joined_into_flat_metadata(fake: FakeWity) -> None:
    fake.json({**EXTRACT_REPLY, "fields": {"sender": ["dana@example.com", "marcus@example.com"], "when": []}})
    nodes = fake.extractor(fields=FIELDS)([TextNode(text=EMAIL)])
    assert nodes[0].metadata == {"sender": "dana@example.com; marcus@example.com", "when": ""}
    # What vector stores like Chroma check before they store a node. Raises on a list.
    node_to_metadata_dict(nodes[0], flat_metadata=True)


def test_extract_reads_the_text_without_metadata(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    fake.extractor(fields=FIELDS).extract([TextNode(text=EMAIL, metadata={"secret": "not for Wity"})])
    assert fake.bodies[0]["text"] == EMAIL


def test_split_then_extract_in_a_pipeline(fake: FakeWity) -> None:
    fake.json(SPLIT_REPLY)
    fake.json(EXTRACT_REPLY)
    fake.json(EXTRACT_REPLY)
    pipeline = IngestionPipeline(transformations=[fake.splitter(), fake.extractor(fields=FIELDS, num_workers=1)])
    nodes = pipeline.run(documents=[Document(text=EMAIL)])
    assert [request.url.path for request in fake.requests] == ["/v1/split", "/v1/extract", "/v1/extract"]
    assert all(node.metadata["sender"] == "dana@example.com" for node in nodes)


async def test_async_extract(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    fake.json(EXTRACT_REPLY)
    extractor = fake.extractor(fields=FIELDS)
    metadata = await extractor.aextract([TextNode(text=EMAIL), TextNode(text=EMAIL)])
    assert [m["sender"] for m in metadata] == ["dana@example.com"] * 2
    await extractor.aclose()


def test_no_fields_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        WityExtractor(fields={}, api_key=TEST_KEY)


def test_extract_errors_are_sdk_errors_without_the_key(fake: FakeWity) -> None:
    fake.json({"error": "bad key"}, status=401)
    with pytest.raises(AuthenticationError) as caught:
        fake.extractor(fields=FIELDS).extract([TextNode(text=EMAIL)])
    assert TEST_KEY not in str(caught.value)
