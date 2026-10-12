"""Wity picks which query engine answers each question. Run: uv run python examples/router.py

The two query engines here answer fixed text, so this runs without an LLM key.
In your app they'd be real indexes, for example `VectorStoreIndex(...).as_query_engine()`.
"""

from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool

from llama_index.wity import WitySelector


class Docs(CustomQueryEngine):
    name: str

    def custom_query(self, query_str: str) -> str:
        return f"(answer from the {self.name})"


router = RouterQueryEngine(
    selector=WitySelector(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(Docs(name="billing docs"), description="Plans, invoices, charges and refunds"),
        QueryEngineTool.from_defaults(Docs(name="API reference"), description="API endpoints, parameters and errors"),
    ],
    llm=MockLLM(),  # Only used to merge answers from several engines. WitySelector picks one.
)

for question in ["Why was I charged twice this month?", "What does a 429 from /v1/decide mean?"]:
    response = router.query(question)
    assert response.metadata is not None
    print(question)
    print("  ", response)
    print("  ", response.metadata["selector_result"].reason)
