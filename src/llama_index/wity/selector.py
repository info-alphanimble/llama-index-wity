"""``WitySelector``: Wity picks the query engine or retriever for each query."""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core.base.base_selector import BaseSelector, SelectorResult, SingleSelection
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from wity import Choice, SystemOneResponse

from .classifier import WityClassifier

_QUESTION = "route"

DEFAULT_INSTRUCTIONS = "Which choice is best suited to answer this query?"


class WitySelector(BaseSelector):
    """Pick one choice with a Wity ``choice`` question, for ``RouterQueryEngine`` and ``RouterRetriever``.

    Each choice's description is an option, and the query is the state. The reason on the selection holds
    the probability Wity gave its pick. If Wity fails, the error is raised: it never quietly picks a choice.

    Reasoning is ``"auto"`` by default: Wity thinks only when choices are close. To change that, pass
    your own ``WityClassifier``.
    """

    def __init__(self, *, instructions: str = DEFAULT_INSTRUCTIONS, classifier: WityClassifier | None = None) -> None:
        """
        Args:
            instructions: How to pick, for example "Which data source can answer this question?"
            classifier: Your own ``WityClassifier``, without questions of its own.
                Defaults to one with reasoning ``"auto"``.
        """
        if classifier is not None and classifier.questions is not None:
            raise ValueError("Pass a WityClassifier without questions. The selector sends its own.")
        self.instructions = instructions
        self.classifier = classifier or WityClassifier(reasoning="auto")

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass

    def _select(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        if len(choices) == 1:
            return _only_choice()
        response = self.classifier.classify(query.query_str, {_QUESTION: self._question(choices)})
        return _result(response)

    async def _aselect(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        if len(choices) == 1:
            return _only_choice()
        response = await self.classifier.aclassify(query.query_str, {_QUESTION: self._question(choices)})
        return _result(response)

    def _question(self, choices: Sequence[ToolMetadata]) -> Choice:
        if not choices:
            raise ValueError("WitySelector needs at least one choice.")
        # Options are keyed by position, so names can be missing or repeated.
        return Choice(
            instructions=self.instructions,
            criteria={str(i): choice.description for i, choice in enumerate(choices)},
        )


def _result(response: SystemOneResponse) -> SelectorResult:
    answer = response.choices[_QUESTION]
    reason = f"Wity picked this choice with probability {answer.probabilities[answer.choice]}."
    return SelectorResult(selections=[SingleSelection(index=int(answer.choice), reason=reason)])


def _only_choice() -> SelectorResult:
    return SelectorResult(selections=[SingleSelection(index=0, reason="The only choice.")])
