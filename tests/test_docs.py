"""Documentation guards: relative links resolve and CLI pages point somewhere real.

``scripts/check_docs.py`` is the same logic CI runs; these tests make it part of
``make test`` too, so a moved or renamed doc fails locally before the push.
"""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_docs", ROOT / "scripts" / "check_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_docs"] = module
    spec.loader.exec_module(module)
    return module


def test_relative_links_resolve():
    problems = _checker().check_links()
    assert not problems, "\n".join(problems)


def test_every_cli_page_points_at_its_own_non_cli_folder():
    problems = _checker().check_structure()
    assert not problems, "\n".join(problems)


def test_every_algorithm_has_a_non_cli_guide():
    """11 algorithms, 11 guides: the mapping that was missing for Double DQN."""
    cli_pages = {
        page.stem
        for page in (ROOT / "docs" / "cli").glob("*.md")
        if page.name != "README.md"
    }
    assert len(cli_pages) == 11, sorted(cli_pages)
    for page in sorted(cli_pages):
        readme = ROOT / "docs" / "cli" / f"{page}.md"
        text = readme.read_text(encoding="utf-8")
        assert "../non-cli/" in text, f"{readme} does not link to a non-cli guide"
