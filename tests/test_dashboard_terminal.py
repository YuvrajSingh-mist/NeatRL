"""Terminal-level behaviour: cursor ownership, and the Windows console path.

The dashboard repaints ~20 rows in place five times a second. Without hiding
the terminal's own cursor, it blinks over the panel -- and over SSH, where the
frame arrives in chunks, it appears to wander around it.

The redraw also speaks ANSI directly, which a POSIX terminal understands and a
Windows console only does after ENABLE_VIRTUAL_TERMINAL_PROCESSING. Neither
`termios`/`tty` nor `select` on stdin work there, so the POSIX plumbing is
imported as an optional capability and `msvcrt` covers the rest.
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

from neatrl.cli import dashboard
from neatrl.cli.dashboard import CURSOR_HIDE, CURSOR_SHOW, Dashboard


class _FakeStdout(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_the_cursor_is_hidden_for_the_whole_run(monkeypatch):
    """A visible cursor is drawn on top of whatever row is being rewritten."""
    fake = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake)

    dash = Dashboard("DQN", "CartPole-v1", 1000)
    assert fake.getvalue().startswith(CURSOR_HIDE), fake.getvalue()[:20]
    assert dash.cursorHidden

    dash.close()
    assert fake.getvalue().endswith(CURSOR_SHOW), "close() left the cursor hidden"


def test_hiding_the_cursor_happens_once(monkeypatch):
    """Re-hiding on every frame would flood the stream with escapes."""
    fake = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake)

    dash = Dashboard("DQN", "CartPole-v1", 1000)
    dash.hideCursor()
    dash.hideCursor()
    assert fake.getvalue().count(CURSOR_HIDE) == 1


def test_cursor_restore_is_idempotent(monkeypatch):
    fake = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake)

    dash = Dashboard("DQN", "CartPole-v1", 1000)
    dash.showCursor()
    dash.showCursor()
    assert fake.getvalue().count(CURSOR_SHOW) == 1


def test_plain_stream_gets_no_cursor_escapes(monkeypatch):
    """Piped output and CI logs must not be polluted with control sequences."""

    class _NotATty(io.StringIO):
        def isatty(self) -> bool:
            return False

    fake = _NotATty()
    monkeypatch.setattr("sys.stdout", fake)

    dash = Dashboard("DQN", "CartPole-v1", 1000)
    dash._redraw("a\nb")
    written = fake.getvalue()

    assert CURSOR_HIDE not in written
    assert "\x1b[" not in written, "escape sequence written to a non-tty"
    assert "a" in written and "b" in written


def test_ansi_support_is_implicit_off_windows(monkeypatch):
    """POSIX terminals speak ANSI; only Windows needs the console mode change."""
    monkeypatch.setattr(dashboard, "IS_WINDOWS", False)
    assert dashboard.enableAnsi() is True


def test_the_dashboard_imports_where_termios_does_not_exist():
    """`import termios` at module level used to break the whole package on Windows."""
    script = textwrap.dedent(
        """
        import sys

        class BlockPosixTty:
            blocked = {"termios", "tty"}

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.blocked:
                    raise ImportError(name)
                return None

        sys.meta_path.insert(0, BlockPosixTty())

        import neatrl.cli.dashboard as d

        assert d.HAS_POSIX_TTY is False, "the platform check did not take effect"

        dash = d.Dashboard("DQN", "CartPole-v1", 10)
        # No msvcrt on a POSIX box: the probe must decline, not raise.
        assert dash.pollKey() is None
        assert "CPU" in dash.renderDash(0, 0, {}, None, "")
        dash.close()
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_windows_arrow_keys_are_normalised(monkeypatch):
    """msvcrt reports arrows as \\x00 + 'H'/'P'; the view code expects ANSI."""
    keys = iter(["\x00", "H"])
    fake = type(
        "FakeMsvcrt",
        (),
        {
            "kbhit": staticmethod(lambda: True),
            "getwch": staticmethod(lambda: next(keys)),
        },
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(dashboard, "HAS_POSIX_TTY", False)

    assert Dashboard._pollKeyWindows() == "\x1b[A"


def test_windows_key_probe_reports_no_key(monkeypatch):
    fake = type("FakeMsvcrt", (), {"kbhit": staticmethod(lambda: False)})
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    assert Dashboard._pollKeyWindows() is None


def test_no_utilization_counter_is_reported_as_not_available(monkeypatch):
    """ "GPU: 0.0%" is a real reading; a host with no counter must say n/a."""
    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "cpu"))
    assert dashboard.gpuCell(dashboard.GPU_UNKNOWN) == "GPU: [cyan]n/a[/cyan]"
    assert dashboard.gpuMemCell(0.0) == "VRAM: [cyan]n/a[/cyan]"
