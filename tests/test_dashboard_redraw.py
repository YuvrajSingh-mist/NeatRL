"""Regression tests for the terminal dashboard's in-place redraw.

The dashboard used to reposition with a relative cursor-up (``ESC[nA``).
Anything that scrolled the terminal -- a frame taller than the viewport, or a
line printed by another library such as wandb -- shifted the origin for good, so
each frame landed lower than the last and whole panels accumulated in the
scrollback.
"""

from __future__ import annotations

import io
import os
import re

import pytest

from neatrl.cli.dashboard import Dashboard


class _FakeStdout(io.StringIO):
    """Stand-in for stdout: a tty that records what the dashboard writes."""

    def isatty(self) -> bool:
        return True


ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


@pytest.fixture()
def make_dash(monkeypatch):
    """Build a Dashboard whose stdout is captured, on a fixed 100x30 terminal.

    ``sys.stdout`` is patched when the factory is *called*, not when the fixture
    is built: pytest restores the capture wrapper between fixture setup and the
    test body, so a patch made during setup would not reach the test.
    """
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *a, **k: os.terminal_size((100, 30))
    )

    def _make():
        fake = _FakeStdout()
        monkeypatch.setattr("sys.stdout", fake)
        dash = Dashboard("DQN", "CartPole-v1", 1000)
        # Drop the one-time setup output (cursor hide) so each test inspects
        # only what its own redraw wrote.
        fake.seek(0)
        fake.truncate(0)
        return dash, fake

    return _make


def test_redraw_never_uses_relative_cursor_movement(make_dash):
    """Relative repositioning is what let panels stack after a scroll."""
    dash, out = make_dash()
    dash._redraw("frame one\nframe two")
    dash._redraw("frame three\nframe four")
    written = out.getvalue()

    assert not re.search(r"\x1b\[\d+A", written), "relative cursor-up is back"
    assert written.startswith("\x1b[2J\x1b[H"), "first frame should clear the screen"
    assert written.count("\x1b[H") >= 2, "later frames must home to the origin"


def test_redraw_stays_within_the_viewport(make_dash):
    """Printing past the last row scrolls, which used to break positioning."""
    dash, out = make_dash()
    dash._viewport = lambda: (100, 20)
    dash._redraw("\n".join(f"row {i}" for i in range(60)))

    visible = ANSI.sub("", out.getvalue()).split("\n")
    assert len(visible) <= 20, f"wrote {len(visible)} rows into a 20-row terminal"


def test_redraw_keeps_the_closing_border_when_trimming(make_dash):
    """Trimming must not slice the panel in half."""
    dash, out = make_dash()
    dash._viewport = lambda: (100, 10)
    dash._redraw("top\n" + "\n".join(f"body {i}" for i in range(30)) + "\nbottom")

    visible = ANSI.sub("", out.getvalue()).split("\n")
    assert visible[0] == "top"
    assert visible[-1] == "bottom"


def test_redraw_does_not_end_with_a_newline(make_dash):
    """A trailing newline on the final row scrolls the screen by one."""
    dash, out = make_dash()
    dash._redraw("only line")
    before_clear = out.getvalue().split("\x1b[J")[0]
    assert not before_clear.endswith("\n"), "trailing newline pushes the screen"


def test_redraw_clears_each_row_to_end_of_line(make_dash):
    """Leftovers from a previously wider frame must not survive."""
    dash, out = make_dash()
    dash._redraw("a\nb")
    assert out.getvalue().count("\x1b[K") == 2


def test_viewport_survives_a_zero_sized_tty(monkeypatch):
    """Pseudo-terminals can report 0 rows; that must not collapse the panel."""
    dash = Dashboard("DQN", "CartPole-v1", 1000)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *a, **k: os.terminal_size((0, 0))
    )
    cols, rows = dash._viewport()
    assert cols >= 40 and rows >= 12
