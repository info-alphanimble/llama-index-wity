# Developing llama-index-wity

## Commands

```sh
uv sync            # install the package and the dev tools
./check.sh         # format check, lint, types (mypy strict), tests, build
uv run ruff format .   # fix formatting
```

- The tests run offline and cost nothing. `pytest-socket` blocks the network. A fake Wity answers through `httpx.MockTransport`, and agent tests use LlamaIndex's `MockFunctionCallingLLM`, so no LLM key is needed.
- The fake Wity replays real API responses saved in `tests/fixtures/`.
- Don't add `from __future__ import annotations` to a file that defines tools taking `ctx: Context`. LlamaIndex compares the annotation to `Context` directly, and a string annotation doesn't match.
- To test another Python version: `uv run --isolated --python 3.13 --with pytest --with pytest-asyncio --with pytest-socket pytest`.
- To test the oldest allowed dependencies, add `--with "llama-index-core==0.14.12" --with "wity-sdk==0.1.1"` to that command.

## Live tests

The live tests make 6 real calls to Wity, costing a fraction of a cent. Put your key in a `.env` file in this folder (it's git-ignored):

```sh
WITY_API_KEY=wity_...
```

Then run:

```sh
WITY_LIVE=1 uv run --env-file .env pytest tests/test_live.py -v
```

Set `WITY_BASE_URL` in the same file to test against another Wity address.

## Examples

Every script in `examples/` must run. After changing one, run it with a real key:

```sh
uv run --env-file .env python examples/quick_start.py
```

The README's code matches the examples. If an example's output changes, update the README to match.

## Publishing

1. Bump the version in `pyproject.toml` and `src/llama_index/wity/__init__.py`.
2. Run `./check.sh` and the live tests.
3. Run `uv publish` with a PyPI token.
