# llama-index-wity

Use [Wity](https://wity.alphanimble.com), the typed-decision API, in LlamaIndex. Ask Wity questions with fixed options and get a probability for every option, in a workflow, a router or an agent.

It has five parts:

- `WityClassifier`: asks Wity typed questions from any LlamaIndex code.
- `WitySplitter`: a node parser that cuts documents into whole items, like emails in a thread or clauses in a contract.
- `WityExtractor`: a metadata extractor that pulls named fields out of each node.
- `WitySelector`: Wity picks the query engine or retriever for each query.
- `WityToolGate`: Wity checks risky tool calls before they run.

## Install

```sh
pip install llama-index-wity
```

Or with uv: `uv add llama-index-wity`. Needs Python 3.10 or newer. It's built on the [`wity-sdk`](https://pypi.org/project/wity-sdk/) package.

## Quick start

Set your key in the environment:

```sh
export WITY_API_KEY="wity_..."
```

Then ask questions:

```python
from llama_index.wity import WityClassifier, choice, noul, score

classifier = WityClassifier()

res = classifier.classify(
    "I was charged twice for my subscription this month. Please refund one of them today.",
    {
        "team": choice("Which team should handle this ticket?", {
            "billing": "Payments, charges and refunds",
            "technical": "Bugs, errors and outages",
            "other": "Anything else",
        }),
        "urgent": noul(
            "Does the customer ask for something to happen soon?",
            yes="They ask for fast action",
            no="No time pressure",
        ),
        "severity": score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"]),
    },
)

res.choices["team"].choice         # "billing"
res.choices["team"].probabilities  # {"billing": 0.99993..., "technical": 0.00004..., "other": 0.00001...}
res.nouls["urgent"].noul           # 0.99986... = probability of yes
res.scores["severity"].score       # 1.97852... = expected level, from 0 (Not at all) to 2 (A lot)
```

The answer is the `wity-sdk` response, unchanged. Answers come back under the names you gave the questions, in `res.choices`, `res.nouls` and `res.scores`. The [SDK README](https://pypi.org/project/wity-sdk/) has every field.

Probabilities sum to 1, but they aren't calibrated. Treat them as a ranking of how sure Wity is, not as exact odds.

Use `await classifier.aclassify(...)` in async code. Keep one classifier around and reuse it: it keeps its connections open, so later calls are faster. Call `close()` (or `await aclose()`) when you're done with it.

## In a workflow

Give the classifier its questions when you create it. Then each call takes only the state. State can be text, a dict, a list, a LlamaIndex `ChatMessage` or a list of messages. Messages are sent as `{role, content}`, and they can also sit inside dicts and lists:

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.wity import WityClassifier, choice

team = WityClassifier(
    questions={
        "team": choice("Which team should handle this chat?", {
            "billing": "Payments, charges and refunds",
            "technical": "Bugs, errors and outages",
        })
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

await Triage().run(chat=chat)  # "Sent to technical."
```

Give questions to the constructor or to each call, not both.

Tool calls in a message are sent as `tool_calls`. Images, audio and files are sent by type only, like `{"type": "image"}`. Sent as text, even a small image would be thousands of tokens Wity can't read.

## Router

`WitySelector` plugs into `RouterQueryEngine` and `RouterRetriever`. Each choice's description is an option, and Wity picks one for each query:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.wity import WitySelector

router = RouterQueryEngine(
    selector=WitySelector(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(billing_engine, description="Plans, invoices, charges and refunds"),
        QueryEngineTool.from_defaults(api_engine, description="API endpoints, parameters and errors"),
    ],
    llm=llm,
)

response = router.query("What does a 429 from /v1/decide mean?")  # answered by api_engine
response.metadata["selector_result"].reason
# "Wity picked this choice with probability 0.999333297708751."
```

- Change how Wity picks with `WitySelector(instructions="Which docs can answer this question?")`.
- With only one choice, it's picked without a call to Wity.
- If Wity fails, the error is raised. The selector never quietly picks a choice for you.
- It picks one choice. Merging answers from several engines isn't supported.
- `RouterQueryEngine` still wants an `llm`, even though Wity does the picking. Without one, it falls back to LlamaIndex's default, OpenAI.

## Tool-risk gate

`WityToolGate` checks a tool before it runs. Wrap each risky tool with `guard` and give the wrapped tools to your agent. Tools you don't wrap run without a check:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.wity import WityToolGate

gate = WityToolGate()
agent = FunctionAgent(tools=[read_page, gate.guard(send_email), gate.guard(delete_file)], llm=llm)
```

Wity sees the proposed call, the tool's description, every message the user wrote and the 30 most recent messages of the run. It answers one yes/no question: should this call be stopped? It stops calls the user didn't ask for or said not to do yet, calls that do more than the user asked, and calls that expose private data. A call that does exactly what the user asked runs, even a lasting one like deleting the file they named.

- When the probability of risky is at or above `threshold` (0.5 by default), the tool doesn't run. The agent gets an error result instead (`is_error=True`): "The tool call `send_email` was blocked because Wity rated it risky (probability 0.9999203269944515). The tool was not run." A blocked `return_direct` tool doesn't end the run, so the agent can tell the user.
- If Wity fails, the tool doesn't run. The agent gets the error as the tool's result, and the run goes on.
- Wity treats everything in the state as data, not instructions. Only the user's own messages can approve an action. So a web page or file can't approve a call by pretending to give orders. `examples/tool_gate.py` shows this.
- The wrapped tool keeps the original's name, description and arguments, so the LLM sees no difference.
- The wrapped tool reads the conversation from the agent's `Context`, so call it through a LlamaIndex agent, like `FunctionAgent` or `ReActAgent`. Called on its own, it raises an error and doesn't run.
- With `ReActAgent`, Wity sees the chat history but not the current run's earlier steps. `ReActAgent` keeps those apart from the chat messages.
- The gate blocks; it doesn't ask a person. For a person's approval, use LlamaIndex's human-in-the-loop events in the tool itself.

### Tuning the threshold

Since the probabilities aren't calibrated, try the gate on some of your own tool calls, safe ones and risky ones. Then set `threshold` between where they land:

```python
gate = WityToolGate(threshold=0.8)
```

You can also change the question and what risky and safe mean:

```python
from llama_index.wity import NoulCriteria

gate = WityToolGate(
    criteria=NoulCriteria(
        true="The call writes, deletes, publishes or changes access.",
        false="The call only reads public or user-provided data.",
    ),
)
```

## Split and extract

`WitySplitter` cuts documents into whole items, one node per item. `WityExtractor` pulls named fields out of each node and adds them to its metadata. Use them together in an ingestion pipeline:

```python
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.wity import WityExtractor, WitySplitter

pipeline = IngestionPipeline(transformations=[
    WitySplitter(split="one item per email"),
    WityExtractor(
        fields={"sender": "The email address the email is from", "when": "A date the email mentions"},
        types={"when": "date"},
    ),
])

nodes = pipeline.run(documents=[Document(text=thread)])
nodes[0].text                  # "From: dana@example.com\n..."
nodes[0].metadata["sender"]    # "dana@example.com"
nodes[0].start_char_idx        # where the email starts in the thread
```

**Splitting.** Say what one item is in plain words. Every item is an exact slice of the text, so `start_char_idx` and `end_char_idx` point at it in the original document. Without `split`, Wity cuts one item per paragraph. For more control, pass a `SplitSpec(unit=..., starts_at=[...], keep_together=[...], never_split=[...])`. To split the way a saved split skill does, pass `skill="name"` instead of `split`. `split_text(text)` gives the items as text.

**Extracting.** Ask for 1 to 16 fields. Each field's name becomes a metadata key, holding the values found, as written. Several values are joined with `"; "`, like `"Acme Ltd; Bob"`. A field with no value gets `""`. Metadata stays text, because many vector stores, like Chroma, take only text and numbers. For the values as a list, call `extract` in `wity-sdk`. `types` takes `"date"`, `"duration"` or `"number"` for values that should come back normalised. The extractor reads the node's own text, not its metadata. It sends `num_workers` requests at once (4 by default).

A document can be up to 60,000 characters. Split and extract read the whole document, so they wait at least two minutes before timing out.

Both work with `pipeline.run(num_workers=4)`. Each worker process opens its own connections, with the same key and options. An `http_client` you passed in stays in the main process.

## Reasoning and speed

Wity answers most questions in one fast pass. With `reasoning="auto"` it thinks first when a question needs it, for example when the top options are close. That usually takes 1–4 seconds.

| Setting | What it does |
|---|---|
| `"off"` | Never thinks. Fastest. |
| `"auto"` | Thinks only when the question needs it. The API default. |
| `"always"` | Always thinks. |

`WityClassifier`, `WitySelector` and `WityToolGate` all use `"auto"`. To change the selector's or the gate's setting, pass your own classifier. It must not have questions of its own:

```python
selector = WitySelector(classifier=WityClassifier(reasoning="off"))
gate = WityToolGate(classifier=WityClassifier(reasoning="always"))
```

`max_latency_ms=` (a whole number from 200 to 120000) caps the time per request. Thinking stops early to meet it.

## Options

| Option | Default | What it does |
|---|---|---|
| `questions` | none | `WityClassifier`: fixed questions. Then each call takes only the state. |
| `reasoning` | API default (`"auto"`) | `"off"`, `"auto"` or `"always"` |
| `max_latency_ms` | none | Time limit per request, a whole number from 200 to 120000 ms |
| `split`, `skill` | API default | `WitySplitter`: what one item is, or a split skill. Not both. |
| `fields`, `types` | required, none | `WityExtractor`: the fields to pull out, and which ones to normalise |
| `api_key` | `WITY_API_KEY` | Your Wity key |
| `base_url` | `WITY_BASE_URL`, then the SDK's default | The API address |
| `timeout` | 60 | Seconds per attempt |
| `max_retries` | 2 | Retries on 408, 429, 5xx and network failures. On `WityExtractor`, this is LlamaIndex's own setting: it runs the whole batch again, so leave it at 0. |
| `http_client`, `http_async_client` | none | Your own `httpx` clients, for example for a proxy |

## Errors, retries and tracing

**Errors** are the `wity-sdk` errors, also importable from `llama_index.wity`:

| Error | When |
|---|---|
| `BadRequestError` | 400, including a state longer than Wity accepts |
| `AuthenticationError` | 401, a missing or wrong key |
| `InsufficientBalanceError` | 402 |
| `RateLimitError` | 429 |
| `InternalServerError` | 5xx |
| `APIConnectionError` | The network failed |
| `APITimeoutError` | A request took longer than `timeout` |

All of them are a `WityError`. Error messages and logs never include your state or your key.

**Retries.** The SDK already retries failed calls (`max_retries=2`). Don't add your own retries on top without setting `max_retries=0`.

**Tracing.** Each `classify` and `aclassify` call is a span in LlamaIndex's instrumentation, so it shows up in Arize Phoenix, Langfuse and the other LlamaIndex observability tools. The span's input includes the state you sent. Your key never appears in spans or `repr()`.

## Update notice

When code that uses `llama-index-wity` runs at a terminal, it tells you on stderr when a newer version is out, at most once a day. It asks PyPI at most once a day, in the background, so nothing waits for it. It stays quiet in CI, when output is piped, and on servers. Set `WITY_NO_UPDATE_CHECK=1` to turn it off. The same variable turns it off in every Wity package.

## Examples

The `examples/` folder in the repository has a script for each section above. The router and gate scripts use LlamaIndex's mock LLMs, so they run with only a Wity key:

```sh
export WITY_API_KEY="wity_..."
uv run python examples/quick_start.py
```

## Contributing

See `CONTRIBUTING.md` in the repository for how to run the checks and tests.

## License

Apache-2.0
