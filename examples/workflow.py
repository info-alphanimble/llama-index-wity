"""Use Wity inside a LlamaIndex Workflow step. Run: uv run python examples/workflow.py

The step sends each ticket to a team, and hands it to a person when Wity isn't sure.
"""

import asyncio

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from llama_index.wity import WityClassifier, choice

team = WityClassifier(
    questions={
        "team": choice(
            "Which team should handle this chat?",
            {"billing": "Payments, charges and refunds", "technical": "Bugs, errors and outages"},
        )
    },
    reasoning="off",
)


class Triage(Workflow):
    @step
    async def route(self, ev: StartEvent) -> StopEvent:
        res = await team.aclassify(ev.chat)
        answer = res.choices["team"]
        if answer.confidence < 0.7:
            return StopEvent(result="Sent to a person: Wity isn't sure.")
        return StopEvent(result=f"Sent to {answer.choice}.")


async def main() -> None:
    chat = [
        ChatMessage(role=MessageRole.USER, content="Hi, the app crashes when I upload a photo."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Sorry! Which phone are you using?"),
        ChatMessage(role=MessageRole.USER, content="An iPhone 15. It happens every time."),
    ]
    print(await Triage().run(chat=chat))


asyncio.run(main())
