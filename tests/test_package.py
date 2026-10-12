import re
from pathlib import Path

import llama_index.wity
from llama_index.wity import __all__, __version__


def test_everything_in_all_imports() -> None:
    for name in __all__:
        assert getattr(llama_index.wity, name) is not None


def test_version_matches_pyproject() -> None:
    # A regex, not tomllib: tomllib needs Python 3.11 and we support 3.10.
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    match = re.search(r'^version = "(.+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)


def test_core_still_imports_next_to_us() -> None:
    from llama_index.core import VectorStoreIndex  # noqa: F401
