"""NeatRL terminal training dashboard — Rich-based, PufferLib-style."""

import atexit
import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from typing import Any, Optional

import torch
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .charts import (
    METRICS,
    chart_cell,
    compact,
    progress_bar,
)
from .history import HistoryTracker

# POSIX-only terminal plumbing. `select` itself exists on Windows but refuses
# non-socket file descriptors, so the whole trio is treated as one capability
# and `msvcrt` covers the Windows side. Importing `termios` unconditionally at
# module level made the entire package unimportable on Windows.
try:
    import select
    import termios
    import tty

    HAS_POSIX_TTY = True
except ImportError:  # pragma: no cover - exercised only on Windows
    HAS_POSIX_TTY = False

IS_WINDOWS = sys.platform.startswith("win")

# Escape sequences the dashboard writes itself, outside Rich.
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"

# Package metadata is the single source of the version string; the dashboard
# used to hardcode it and drifted from pyproject.toml.
try:
    from .. import __version__ as _NEATRL_VERSION
except Exception:  # pragma: no cover - partial install
    _NEATRL_VERSION = "0.0.0+unknown"

try:
    import psutil as psutilMod

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

LOGBUFSIZE = 500
LOG_VISIBLE = 30
HISTORY_MAXLEN = 120
# Columns in the charts view's grid.
CHART_COLUMNS = 2
# Lines one chart occupies: the label row, the trace, and a spacer.
CHART_CELL_LINES = 3


# Live dashboards in this process. A trainer's tqdm bar and the dashboard both
# write to the same tty, so the bar has to know to stay quiet -- see `progress`.
_ACTIVE_DASHBOARDS = 0


def dashboardActive() -> bool:
    """True while a dashboard owns the screen."""
    return _ACTIVE_DASHBOARDS > 0


def progress(iterable, **kwargs):
    """tqdm that stands aside while the dashboard is drawing.

    The dashboard repaints the whole panel from the top on every frame; a tqdm
    bar redraws its own line with a carriage return on every step. With both
    writing, the bar lands inside the panel and is erased by the next frame --
    that is the flicker. The frame already carries the same counters, so the
    bar has nothing to add while the dashboard is up.
    """
    from tqdm import tqdm

    kwargs.setdefault("disable", dashboardActive())
    return tqdm(iterable, **kwargs)


def enableAnsi() -> bool:
    """Turn on ANSI escape handling for this process.

    POSIX terminals speak ANSI natively. A Windows console only does once
    ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` is set, and only Windows 10 1511+
    supports it at all -- Windows Terminal and ConPTY always do. When it cannot
    be enabled the dashboard degrades to plain, scrolling output instead of
    printing escape sequences as literal text.
    """
    if not IS_WINDOWS:
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(
            kernel32.SetConsoleMode(
                handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
        )
    except Exception:  # pragma: no cover - depends on the host console
        return False


def deviceTag() -> tuple[str, str, str]:
    if torch.cuda.is_available():
        return "GPU", "VRAM", "cuda"
    if torch.backends.mps.is_available():
        # Not "VRAM": the GPU shares system RAM here, and the figure is an
        # absolute size rather than a percentage of a separate pool. "MPS"
        # also fits the header at 80 columns, where a longer label cropped
        # the value for the longest trainer names.
        return "GPU", "MPS", "mps"
    return "GPU", "VRAM", "cpu"


DT = deviceTag()

# (monotonic expiry, utilization %) for the throttled ioreg probe.
_appleGpuCache: tuple[float, float] = (0.0, 0.0)


def abbrev(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def duration(secs: float) -> str:
    secs = int(secs)
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _apple_gpu_util() -> float:
    """GPU utilization % on Apple Silicon.

    macOS has no NVML equivalent and ``powermetrics`` needs root, but ``ioreg``
    publishes the Metal driver's own counters unprivileged. The call costs about
    10 ms, so the result is reused for a second: the dashboard only samples at
    5 Hz and the number does not change faster than that.
    """
    global _appleGpuCache
    if sys.platform != "darwin":
        # No AGX accelerator to ask; avoid spawning ioreg on every host.
        return 0.0
    now = time.monotonic()
    if _appleGpuCache[0] > now:
        return _appleGpuCache[1]

    util = 0.0
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        ).stdout
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        if m:
            util = float(m.group(1))
    except (OSError, subprocess.SubprocessError):
        util = 0.0

    _appleGpuCache = (now + 1.0, util)
    return util


# Sentinel for "this host has no utilization counter", rendered as n/a. Zero
# would be a real reading: an idle GPU.
GPU_UNKNOWN = -1.0

# NVML is initialised once per process; doing it per frame cost a driver call
# 5x a second and spammed the log when the driver refused.
_nvmlHandle: Any = None
_nvmlTried = False


def _nvidia_gpu_util() -> float:
    """Utilization % from NVML. Works on Linux and Windows (not macOS)."""
    global _nvmlHandle, _nvmlTried
    if not _nvmlTried:
        _nvmlTried = True
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvmlHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            _nvmlHandle = None
    if _nvmlHandle is None:
        return GPU_UNKNOWN
    try:
        import pynvml

        return float(pynvml.nvmlDeviceGetUtilizationRates(_nvmlHandle).gpu)
    except Exception:
        return GPU_UNKNOWN


def _amd_gpu_util() -> float:
    """Utilization % on ROCm, where NVML does not exist.

    ``amdsmi`` is the supported interface and ships with ROCm 5.7+. Linux only;
    there is no AMD GPU path on macOS.
    """
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        activity = amdsmi.amdsmi_get_gpu_activity(handles[0])
        return float(activity["gfx_activity"])
    except Exception:
        return GPU_UNKNOWN


def gpuUtil() -> float:
    """Utilization % for the active backend, or GPU_UNKNOWN."""
    if DT[2] == "cuda":
        util = _nvidia_gpu_util()
        if util == GPU_UNKNOWN:
            util = _amd_gpu_util()
        return util
    if DT[2] == "mps":
        return _apple_gpu_util() if sys.platform == "darwin" else GPU_UNKNOWN
    return GPU_UNKNOWN


def getHw() -> tuple[float, float, float, float]:
    cpu = psutilMod.cpu_percent() if HAS_PSUTIL else 0.0
    # System RAM in use. "DRAM" is technically right (it is the memory chips,
    # not the SSD) but reads as jargon next to the disk, so the panel says RAM.
    ram = psutilMod.virtual_memory().percent if HAS_PSUTIL else 0.0
    gpu, vram = GPU_UNKNOWN, 0.0

    if DT[2] == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            vram = 100.0 * torch.cuda.memory_allocated(0) / props.total_memory
        except (RuntimeError, AttributeError):
            vram = 0.0
        gpu = gpuUtil()
    elif DT[2] == "mps":
        # Apple GPUs share system RAM, so "VRAM" is the share of the MPS memory
        # budget the driver has handed out. The old code reported the whole
        # Python process RSS against total RAM, which is not GPU usage at all.
        try:
            budget = torch.mps.recommended_max_memory()
            if budget > 0:
                vram = 100.0 * torch.mps.driver_allocated_memory() / budget
        except (RuntimeError, AttributeError):
            vram = 0.0
        gpu = _apple_gpu_util()
    return cpu, gpu, ram, vram


def gpuMemCell(vram: float) -> str:
    """Header cell for GPU-side memory.

    CUDA gets a percentage of the card's VRAM, which is the real ceiling. On
    MPS a percentage compares against PyTorch's soft budget (~74% of RAM), so
    two different denominators sat side by side and the figure read as
    surprisingly small. An absolute size answers "how much is the GPU holding"
    without inviting that comparison; GPU utilization is the number that moves
    during training anyway.
    """
    if DT[2] == "mps":
        try:
            held = torch.mps.driver_allocated_memory()
        except (RuntimeError, AttributeError):
            return f"{DT[1]}: [cyan]n/a[/cyan]"
        # RL networks are small next to the box, so GB alone showed "0.0G"
        # for a whole training run. Milliseconds in MB until it matters.
        if held < 1024**3:
            return f"{DT[1]}: [cyan]{held / 1024**2:.1f}M[/cyan]"
        return f"{DT[1]}: [cyan]{held / 1024**3:.1f}G[/cyan]"
    if DT[2] == "cpu":
        # Nothing was measured, and "VRAM: 0.0%" would read as a real reading.
        return f"{DT[1]}: [cyan]n/a[/cyan]"
    return f"{DT[1]}: [cyan]{vram:.1f}%[/cyan]"


def gpuCell(gpu: float) -> str:
    """Header cell for GPU utilization, or n/a where no counter exists."""
    if gpu < 0:
        return f"{DT[0]}: [cyan]n/a[/cyan]"
    return f"{DT[0]}: [cyan]{gpu:.1f}%[/cyan]"


LOG_LEVEL_STYLES = {
    "DEBUG": Style(dim=True, color="bright_black"),
    "INFO": Style(color="white"),
    "WARNING": Style(color="yellow", bold=True),
    "ERROR": Style(color="red", bold=True),
    "CRITICAL": Style(color="red", bold=True, bgcolor="yellow"),
}


def colorizeLogLine(line: str) -> Text:
    """Apply Rich styling to a log line based on its log level."""
    t = Text()
    # Pattern: "2026-06-26 07:03:29 | INFO     | dqn_mlp | message"
    m = re.match(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| )"
        r"(\w+)(\s*\| .*? \| )(.*)",
        line,
    )
    if m:
        t.append(m.group(1), style="dim")
        level = m.group(2)
        t.append(level, style=LOG_LEVEL_STYLES.get(level, Style()))
        t.append(m.group(3), style="dim cyan")
        t.append(m.group(4))
    else:
        t.append(line)
    return t


class LogCaptureHandler(logging.Handler):
    def __init__(self, buffer: deque) -> None:
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))


LOGFMT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


class Dashboard:
    def __init__(
        self, algo: str, env_id: str, total_timesteps: int, config: Optional[Any] = None
    ) -> None:
        self.algo = algo
        self.envId = env_id
        self.totalTimesteps = total_timesteps
        self.configObj = config
        self.t0 = time.time()
        self.console = Console(highlight=False)
        self.isFirst = True
        self.viewportRows = 0
        self.showLogs = False
        self.showConfig = False
        self.showCharts = False  # 'v' toggles full chart view
        self.lastRenderTime = 0.0
        self.lastEpoch = 0
        self.lastLosses: dict = {}
        self.lastEvalStats: Optional[dict] = None
        self.lastAgentSteps: int = 0
        self.logScrollOffset = 0  # 0 = newest, positive = scroll back

        # History tracker for time-series charts
        self.history = HistoryTracker(maxlen=HISTORY_MAXLEN)

        # Terminal raw mode. This is best effort: stdin is a pseudo-file under
        # pytest capture, notebooks and some CI runners, where fileno() raises
        # instead of returning a descriptor.
        self.fd = None
        try:
            self.fd = sys.stdin.fileno()
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            pass
        self.oldTerm = None
        # ANSI must be enabled before the first frame: on a Windows console
        # without VT processing the escape sequences are printed as text.
        self.ansi = enableAnsi()
        if self.fd is not None and os.isatty(self.fd) and HAS_POSIX_TTY:
            self.oldTerm = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            oldTermCopy = self.oldTerm
            fdCopy = self.fd

            def _restore():
                try:
                    termios.tcsetattr(fdCopy, termios.TCSADRAIN, oldTermCopy)
                except Exception:
                    pass

            atexit.register(_restore)

        # Hide the terminal's own cursor for as long as the dashboard owns the
        # screen. Without this the blinking cursor is visible mid-panel: every
        # frame homes to the origin and rewrites ~20 rows, so the cursor is
        # caught wherever the write happens to have reached -- and over SSH,
        # where the output arrives in chunks, it wanders around the panel.
        self.cursorHidden = False
        if self.ansi and sys.stdout.isatty():
            self.hideCursor()
            atexit.register(self.showCursor)

        # Only claim the screen when one is actually there. Under pytest or a
        # pipe nothing is drawn, and silencing tqdm there would swallow a
        # caller's own progress bars for the rest of the process.
        self._ownsScreen = bool(self.ansi and sys.stdout.isatty())
        if self._ownsScreen:
            global _ACTIVE_DASHBOARDS
            _ACTIVE_DASHBOARDS += 1
        self._lastFrame: Optional[str] = None
        self._lastSpsPush = 0.0

        # Log capture
        self.logBuf: deque[str] = deque(maxlen=LOGBUFSIZE)
        self.logHandler = LogCaptureHandler(self.logBuf)
        self.logHandler.setFormatter(logging.Formatter(LOGFMT, datefmt=DATEFMT))
        root = logging.getLogger()
        self.savedHandlers: list[logging.Handler] = []
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and h.stream in (
                sys.stdout,
                sys.stderr,
            ):
                root.removeHandler(h)
                self.savedHandlers.append(h)
        root.addHandler(self.logHandler)

    def hideCursor(self) -> None:
        """Stop the terminal drawing its cursor over the panel."""
        if self.cursorHidden or not self.ansi:
            return
        try:
            # Remember the stream: an atexit handler must write the restore
            # sequence back to the same place, not to whatever stdout has
            # become by then.
            self._cursorStream = sys.stdout
            sys.stdout.write(CURSOR_HIDE)
            sys.stdout.flush()
            self.cursorHidden = True
        except (OSError, ValueError):
            pass

    def showCursor(self) -> None:
        """Restore the cursor, including when the run dies on an exception."""
        if not self.cursorHidden:
            return
        stream = getattr(self, "_cursorStream", None) or sys.stdout
        try:
            stream.write(CURSOR_SHOW)
            stream.flush()
        except (OSError, ValueError):
            pass
        self.cursorHidden = False

    @staticmethod
    def _pollKeyWindows() -> Optional[str]:
        """Windows equivalent of the select() probe: msvcrt has its own queue."""
        try:
            import msvcrt

            if not msvcrt.kbhit():
                return None
            c = msvcrt.getwch()
            # Arrows and function keys arrive as a two-character sequence with
            # a \x00 or \xe0 prefix; normalise to the ANSI spelling the rest of
            # the dashboard already understands.
            if c in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                return {"H": "\x1b[A", "P": "\x1b[B"}.get(code)
            return c
        except Exception:  # pragma: no cover - no console available
            return None

    def pollKey(self) -> Optional[str]:
        """Non-blocking check for a single keypress on stdin.
        Returns a single character, or '\x1b[A' for up-arrow, '\x1b[B' for down-arrow.
        """
        if not HAS_POSIX_TTY:
            return self._pollKeyWindows()
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1)
                if c == "\x1b":
                    # Escape sequence — check for arrow keys
                    more = select.select([sys.stdin], [], [], 0.05)[0]
                    if more:
                        c += sys.stdin.read(2)
                return c
        except Exception:
            pass
        return None

    def renderLogs(self) -> str:
        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white")
        total = len(self.logBuf)
        scrollInfo = (
            f"  [dim](\u2191\u2193 {self.logScrollOffset})[/dim]"
            if self.logScrollOffset
            else ""
        )
        header.add_row(
            f"NeatRL {_NEATRL_VERSION}  [{self.algo}]  [yellow][LOGS][/yellow]  [dim][l: back][/dim]",
            f"Lines: {total}/{LOGBUFSIZE}{scrollInfo}",
        )

        logLines = list(self.logBuf)
        end = total - self.logScrollOffset
        start = max(0, end - LOG_VISIBLE)
        visible = logLines[start:end]
        self.logScrollOffset = max(0, total - start - LOG_VISIBLE)

        body = Text()
        for line in visible:
            body.append_text(colorizeLogLine(line))
            body.append("\n")

        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(body)
        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def renderConfig(self) -> str:
        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white")
        header.add_row(
            f"NeatRL {_NEATRL_VERSION}  [{self.algo}]  [yellow][CONFIG][/yellow]  [dim][l: logs][c: back][/dim]",
            f"Device: {DT[2]}",
        )
        rows: list[tuple[str, str]] = []
        if self.configObj is not None and dataclasses.is_dataclass(self.configObj):
            for f in dataclasses.fields(self.configObj):
                val = getattr(self.configObj, f.name)
                if isinstance(val, float):
                    rows.append((f.name, f"{val:.6g}"))
                elif isinstance(val, bool):
                    rows.append((f.name, str(val)))
                elif val is None:
                    rows.append((f.name, "None"))
                else:
                    rows.append((f.name, str(val)))
        half = (len(rows) + 1) // 2

        def statCol(data: list) -> Table:
            t = Table(box=None, show_header=True, padding=(0, 1))
            t.add_column("[cyan]Param[/cyan]", style="bright_white", min_width=28)
            t.add_column("[cyan]Value[/cyan]", style="white", min_width=16)
            for k, v in data:
                t.add_row(k, v)
            return t

        body = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        body.add_column(min_width=48)
        body.add_column(min_width=48)
        body.add_row(statCol(rows[:half]), statCol(rows[half:]))
        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(body)
        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def update(
        self,
        *,
        agent_steps: int,
        epoch: int,
        losses: dict,
        eval_stats: Optional[dict] = None,
        message: str = "",
    ) -> None:
        key = self.pollKey()
        if key == "l":
            self.showLogs = not self.showLogs
            self.showConfig = False
            self.showCharts = False
            self.isFirst = True
            self.logScrollOffset = 0
        elif key == "c":
            self.showConfig = not self.showConfig
            self.showLogs = False
            self.showCharts = False
            self.isFirst = True
        elif key == "v":
            self.showCharts = not self.showCharts
            self.showLogs = False
            self.showConfig = False
            self.isFirst = True
        elif key == "\x1b[A":  # up arrow
            if self.showLogs:
                self.logScrollOffset = min(
                    self.logScrollOffset + 1, max(0, len(self.logBuf) - 1)
                )
        elif key == "\x1b[B":  # down arrow
            if self.showLogs:
                self.logScrollOffset = max(0, self.logScrollOffset - 1)

        self.lastEpoch = epoch
        self.lastLosses = losses
        self.lastEvalStats = eval_stats
        self.lastAgentSteps = agent_steps

        now = time.time()
        if not self.isFirst and now - self.lastRenderTime < 0.2:
            return
        self.lastRenderTime = now

        # One SPS sample a second is enough to show the trend, and it keeps the
        # series from being filled by 5 Hz samples of the same running average.
        if agent_steps > 0 and now - self._lastSpsPush >= 1.0:
            elapsed = now - self.t0
            if elapsed > 0:
                self.history.push("sps", agent_steps / elapsed)
            self._lastSpsPush = now

        # Full clear on view switch — paired right before the render
        if self.isFirst:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

        # Size the console to the terminal *before* rendering, so every frame is
        # drawn at the same width. Rendering first and resizing afterwards left
        # a stray column of the previous, wider frame on screen.
        cols, rows = self._viewport()
        self.viewportRows = rows
        self.console.width = max(cols - 1, 40)

        if self.showLogs:
            rendered = self.renderLogs()
        elif self.showConfig:
            rendered = self.renderConfig()
        elif self.showCharts:
            rendered = self.renderChartsView()
        else:
            rendered = self.renderDash(agent_steps, epoch, losses, eval_stats, message)

        self._redraw(rendered)

    def _viewport(self) -> tuple[int, int]:
        """Terminal size, with a usable fallback when the tty reports nothing.

        Some pseudo-terminals report 0 rows; treating that literally would clamp
        the panel down to its minimum height.
        """
        cols, rows = shutil.get_terminal_size(fallback=(100, 30))
        return (cols if cols >= 40 else 100), (rows if rows >= 12 else 30)

    def _redraw(self, rendered: str) -> None:
        """Draw the frame in place: one panel, redrawn on the same screen.

        Positioning is absolute (cursor home) rather than a relative cursor-up.
        With a relative move, anything that scrolls the terminal -- a frame
        taller than the viewport, or a line printed by another library -- shifts
        the origin for good, so every later frame lands lower than the previous
        one and whole panels accumulate in the scrollback.
        """
        if rendered == self._lastFrame and not self.isFirst:
            # Nothing changed, so repainting would only flicker.
            return
        self._lastFrame = rendered

        cols, rows = self._viewport()
        # Leave a column spare: a line exactly as wide as the terminal wraps and
        # costs an extra row, which reintroduces the same drift.
        self.console.width = max(cols - 1, 40)

        lines = rendered.rstrip("\n").split("\n")
        # Never print past the bottom of the screen, or the terminal scrolls.
        budget = max(rows - 1, 8)
        if len(lines) > budget:
            # Keep the top of the frame and its closing border; sacrifice the
            # middle content rather than slicing the panel in half.
            lines = lines[: budget - 1] + [lines[-1]]

        if not (self.ansi and sys.stdout.isatty()):
            # Console without VT processing (old Windows), or output going to
            # a pipe: escape sequences would print as garbage, so print the
            # frame once and let it scroll -- less pretty, still readable.
            if self.isFirst:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
                self.isFirst = False
            return

        if self.isFirst:
            sys.stdout.write("\033[2J\033[H")  # clear once, then overwrite
            self.isFirst = False
        else:
            sys.stdout.write("\033[H")
        # `\033[K` clears the rest of each row, so nothing from a previously
        # wider frame survives to the right of this one. No trailing newline:
        # the cursor must stay on the last row, or writing past it scrolls the
        # screen and the top of the frame is lost.
        sys.stdout.write("\n".join(line + "\033[K" for line in lines))
        sys.stdout.write("\033[J")
        sys.stdout.flush()

    def push_metrics(self, **kwargs: float) -> None:
        """Record time-series metrics into the history tracker for chart rendering."""
        self.history.push_many(**kwargs)

    def renderDash(
        self,
        agent_steps: int,
        epoch: int,
        losses: dict,
        eval_stats: Optional[dict],
        message: str,
    ) -> str:
        elapsed = time.time() - self.t0
        sps = max(int(agent_steps / elapsed), 1) if elapsed > 0 else 1
        remaining = max(self.totalTimesteps - agent_steps, 0) / sps
        cpu, gpu, ram, vram = getHw()

        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_row(
            f"NeatRL {_NEATRL_VERSION}",
            f"CPU: [cyan]{cpu:.1f}%[/cyan]",
            gpuCell(gpu),
            f"RAM: [cyan]{ram:.1f}%[/cyan]",
            gpuMemCell(vram),
        )

        summary = Table(box=None, show_header=True, padding=(0, 1))
        summary.add_column("[cyan]Summary[/cyan]", style="bright_white", min_width=14)
        summary.add_column("[cyan]Value[/cyan]", style="white", min_width=12)
        # The algorithm lives here rather than in the header: names like
        # "Dueling-DQN" did not fit the header row at 80 columns and were
        # silently cut off.
        summary.add_row("Algorithm", self.algo)
        envLabel = self.envId if len(self.envId) <= 14 else self.envId[:12] + "..."
        summary.add_row("Environment", envLabel)
        summary.add_row("Agent Steps", abbrev(agent_steps))
        summary.add_row("SPS", abbrev(sps))
        summary.add_row("Epoch", str(epoch))
        lastRet = next(
            (v for k, v in (eval_stats or {}).items() if k == "last_return"), "-"
        )
        summary.add_row("Return", str(lastRet))
        summary.add_row("Uptime", duration(elapsed))
        summary.add_row("Remaining", duration(remaining))

        lossTable = Table(box=None, show_header=True, padding=(0, 1))
        lossTable.add_column("[cyan]Losses[/cyan]", style="bright_white", min_width=20)
        lossTable.add_column("[cyan]Value[/cyan]", style="white", min_width=10)
        for name, val in losses.items():
            # A diverged TD loss prints as 299886496.0000, which both overflows
            # the column and says nothing below the leading digits.
            lossTable.add_row(name, compact(float(val), ".4f"))

        monitor = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        monitor.add_column(min_width=30)
        monitor.add_column(min_width=34)
        monitor.add_row(summary, lossTable)

        # --- Eval stats ---
        bottom: Optional[Table] = None
        if eval_stats:
            bottomStats = {k: v for k, v in eval_stats.items() if k != "last_return"}
            if bottomStats:
                items = list(bottomStats.items())
                half = (len(items) + 1) // 2

                def col(rows: list) -> Table:
                    t = Table(box=None, show_header=True, padding=(0, 1))
                    t.add_column(
                        "[cyan]Eval Stats[/cyan]", style="bright_white", min_width=22
                    )
                    t.add_column("[cyan]Value[/cyan]", style="white", min_width=10)
                    for k, v in rows:
                        t.add_row(k, f"{v:.3f}")
                    return t

                bottom = Table(box=None, show_header=False, padding=(0, 2))
                bottom.add_column(min_width=36)
                bottom.add_column(min_width=36)
                bottom.add_row(col(items[:half]), col(items[half:]))

        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(monitor)

        # The progress bar replaces the tqdm bar the dashboard suppresses; it
        # is the one number a training run is watched for.
        frame.add_row(self._progressRow(agent_steps))

        # Bottom row: eval stats (left) + model params (right)
        if message or bottom is not None:
            msgParts = message.split(" | ") if message else []
            msgText = "  |  ".join(p.strip() for p in msgParts)
            bottomRow = Table(box=None, show_header=False, padding=(0, 1))
            bottomRow.add_column(min_width=30)
            bottomRow.add_column(min_width=34)
            bottomRow.add_row(
                bottom or Text(""),
                Text(msgText, style="bright_white") if msgText else Text(""),
            )
            frame.add_row(bottomRow)

        frame.add_row(
            "[dim]press [b]l[/b] logs  [b]c[/b] config  [b]v[/b] charts[/dim]"
        )

        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def chartMetrics(self) -> tuple[list[str], list[str]]:
        """Universal and algorithm-specific metric keys that hold data.

        Every series in the history gets a chart, so a trainer that starts
        pushing a new metric needs no change here -- `METRICS` only supplies
        the label, colour and format for the keys it knows.
        """
        present = self.history.all_metrics()
        universalKeys = ["ep_return", "ep_length", "total_loss", "sps"]
        universal = [m for m in universalKeys if m in present]
        algo = [m for m in METRICS if m in present and m not in universal]
        algo += [m for m in present if m not in universal and m not in algo]
        return universal, algo

    def _chartCellWidth(self) -> int:
        """Width of one chart column, borders and padding removed."""
        return max(16, (self.console.width - 8) // CHART_COLUMNS)

    def _chartGrid(self, metrics: list[str], cell: int) -> Table:
        """Two charts per row, each two lines tall: label+value, then the trace."""
        grid = Table(box=None, show_header=False, padding=(0, 1))
        for _ in range(CHART_COLUMNS):
            grid.add_column(no_wrap=True)
        for i in range(0, len(metrics), CHART_COLUMNS):
            cells = [
                Text.from_markup(
                    "\n".join(chart_cell(name, self.history.get(name), cell))
                )
                for name in metrics[i : i + CHART_COLUMNS]
            ]
            while len(cells) < CHART_COLUMNS:
                cells.append(Text(""))
            if i:
                grid.add_row(Text(""), Text(""))
            grid.add_row(*cells)
        return grid

    def _progressRow(self, agent_steps: int) -> str:
        """Progress through the run, shown in place of the suppressed tqdm bar."""
        width = max(12, min(40, self.console.width - 52))
        bar = progress_bar(agent_steps, self.totalTimesteps, width)
        return (
            f"  [dim]Progress[/dim]  {bar}  "
            f"[dim]{abbrev(agent_steps)} / {abbrev(self.totalTimesteps)}[/dim]"
        )

    def renderChartsView(self) -> str:
        """Full-screen grid of every tracked metric, two charts per row."""
        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white")
        header.add_row(
            f"NeatRL {_NEATRL_VERSION}  [{self.algo}]  [yellow][CHARTS][/yellow]  [dim][v: back][/dim]",
            f"Steps: {abbrev(self.lastAgentSteps)}"
            f"  |  {len(self.history.all_metrics())} metrics",
        )

        universal, algo = self.chartMetrics()
        cell = self._chartCellWidth()
        # Whatever the viewport fits: two chart lines plus a spacer per grid
        # row, minus the panel border, header, section titles and key hint.
        budget = max((self.viewportRows or 30) - 8, 6)
        slots = max(CHART_COLUMNS, (budget // CHART_CELL_LINES) * CHART_COLUMNS)
        universal = universal[:slots]
        algo = algo[: max(CHART_COLUMNS, slots - len(universal))]

        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        if universal:
            frame.add_row(Text(""))
            frame.add_row(Text.from_markup("[bold cyan]── Universal Metrics ──[/]"))
            frame.add_row(self._chartGrid(universal, cell))
        if algo:
            if universal:
                frame.add_row(Text(""))
            frame.add_row(Text.from_markup(f"[bold cyan]── {self.algo} Metrics ──[/]"))
            frame.add_row(self._chartGrid(algo, cell))
        if not (universal or algo):
            frame.add_row(Text("No metrics recorded yet.", style="dim"))
        frame.add_row(Text(""))
        frame.add_row("[dim]press [b]v[/b] back  [b]l[/b] logs  [b]c[/b] config[/dim]")

        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def close(self, message: str = "Training complete.") -> None:
        if getattr(self, "_ownsScreen", False):
            global _ACTIVE_DASHBOARDS
            _ACTIVE_DASHBOARDS = max(_ACTIVE_DASHBOARDS - 1, 0)
            self._ownsScreen = False
        if (
            self.oldTerm is not None
            and self.fd is not None
            and HAS_POSIX_TTY
            and os.isatty(self.fd)
        ):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.oldTerm)
        root = logging.getLogger()
        root.removeHandler(self.logHandler)
        for h in self.savedHandlers:
            root.addHandler(h)
        self.savedHandlers.clear()
        # Bypass the frame throttle. The closing frame carries the final
        # counters and the completion message; rendering it less than 0.2 s
        # after the previous frame used to swallow it entirely, leaving the
        # panel frozen mid-run.
        self.lastRenderTime = 0.0
        self.update(
            agent_steps=self.totalTimesteps,
            epoch=self.lastEpoch,
            losses=self.lastLosses,
            eval_stats=self.lastEvalStats,
            message=message,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
        # Only after the final frame: showing it earlier would put the block
        # cursor back in the middle of the panel.
        self.showCursor()
