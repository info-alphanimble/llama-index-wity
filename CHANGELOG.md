# Changelog

All notable changes to `llama-index-wity`. Versions follow [semver](https://semver.org). Before 1.0, a minor version can add features.

## 0.2.0 (2026-10-26)

### Added

- **`WitySplitter`**: a node parser that cuts documents into whole items, like emails in a thread or clauses in a contract. Each item becomes one node.
  - Every item is an exact slice of the text, so `start_char_idx` and `end_char_idx` point at it in the original document.
  - `split` says what one item is, in plain words or as a `SplitSpec`. `skill=` splits the way a saved split skill does.
- **`WityExtractor`**: a metadata extractor that pulls named fields out of each node. Each field's name becomes a metadata key, with the values found as text. Several values are joined with `"; "`, and a field with no value gets `""`. Text works with vector stores that take only text and numbers, like Chroma.
- Both work in an `IngestionPipeline`, including `run(num_workers=...)`. Each worker process opens its own connections, with the same key and options.
- **Update notice.** At a terminal, `llama-index-wity` tells you on stderr when a newer version is on PyPI, at most once a day. The check runs in a background thread, so nothing waits for it. It stays quiet in CI, when output is piped, and on servers. `WITY_NO_UPDATE_CHECK=1` turns it off.
- `SplitSpec`, `SplitItem`, `SplitResponse`, `ExtractType`, `ExtractEntity` and `ExtractResponse`, exported from `llama_index.wity`.

### Changed

- Needs `wity-sdk` 0.2.0 or newer.
- `WityClassifier` can be pickled, and so can a `WitySelector` or `WityToolGate` that holds one. A pickled copy keeps its options and opens its own connections.

## 0.1.0 (2026-10-12)

First release.

- `WityClassifier`: asks Wity typed questions from any LlamaIndex code, with a probability for every option. Calls show up in LlamaIndex's instrumentation.
- `WitySelector`: Wity picks the query engine or retriever for each query.
- `WityToolGate`: Wity checks risky tool calls before they run.
- Chat messages as state, `reasoning` modes and `max_latency_ms`.
