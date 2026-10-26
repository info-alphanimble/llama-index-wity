"""``WityClassifier``: ask Wity typed questions from LlamaIndex code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import wity
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]  # re-exported without __all__
from wity import Question, Reasoning, SystemOneResponse

from ._clients import WityClients
from ._state import to_wity_state

dispatcher = get_dispatcher(__name__)


class WityClassifier:
    """Ask Wity typed questions and get typed answers with probabilities.

    Give questions here, or with each call, but not both.

    State can be text, a dict, a list, a LlamaIndex ``ChatMessage`` or a list of messages.
    Messages can also sit inside dicts and lists. They're sent as ``{role, content}``.

    The answer is the ``wity-sdk`` response: read ``res.choices``, ``res.nouls`` and ``res.scores``.
    Keep one classifier around and reuse it, so connections stay open.
    """

    def __init__(
        self,
        *,
        questions: Mapping[str, Question] | None = None,
        reasoning: Reasoning | None = None,
        max_latency_ms: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = wity.DEFAULT_TIMEOUT,
        max_retries: int = wity.DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Args:
            questions: Questions by a name you choose. Then each call takes only the state.
            reasoning: ``"off"``, ``"auto"`` or ``"always"``. Leave out for the API default, ``"auto"``.
            max_latency_ms: Time limit per request, a whole number from 200 to 120000.
            api_key: Defaults to the WITY_API_KEY environment variable.
            base_url: Defaults to WITY_BASE_URL, then the SDK's default address.
            timeout: Seconds per attempt.
            max_retries: Retries on 408, 429, 5xx and network failures.
            http_client: Your own ``httpx.Client`` for ``classify``, for example for a proxy.
            http_async_client: Your own ``httpx.AsyncClient`` for ``aclassify``.
        """
        if questions is not None and not questions:
            raise ValueError("questions is empty. Pass at least one question, or leave it out.")
        self.questions = dict(questions) if questions is not None else None
        self.reasoning: Reasoning | None = reasoning
        self.max_latency_ms = max_latency_ms
        self._clients = WityClients(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    def __repr__(self) -> str:
        return f"WityClassifier(questions={self.questions!r}, reasoning={self.reasoning!r})"

    @dispatcher.span
    def classify(self, state: Any, questions: Mapping[str, Question] | None = None) -> SystemOneResponse:
        """Ask the questions about ``state``. Pass ``questions`` only if the classifier has none."""
        return self._clients.sync.system_one(
            state=to_wity_state(state),
            questions=self._questions(questions),
            reasoning=self.reasoning,
            max_latency_ms=self.max_latency_ms,
        )

    @dispatcher.span
    async def aclassify(self, state: Any, questions: Mapping[str, Question] | None = None) -> SystemOneResponse:
        """Async ``classify``."""
        return await self._clients.for_this_loop().system_one(
            state=to_wity_state(state),
            questions=self._questions(questions),
            reasoning=self.reasoning,
            max_latency_ms=self.max_latency_ms,
        )

    def _questions(self, questions: Mapping[str, Question] | None) -> Mapping[str, Question]:
        if self.questions is not None and questions is not None:
            raise ValueError("This WityClassifier has its own questions. Don't pass questions to each call too.")
        chosen = self.questions if questions is None else questions
        if not chosen:
            raise ValueError("Pass questions, here or when you create the classifier.")
        return chosen

    def close(self) -> None:
        """Close the connections ``classify`` uses. Clients you passed in stay open."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close the connections ``aclassify`` uses in this event loop. Clients you passed in stay open."""
        await self._clients.aclose()
