"""NeatRL terminal training dashboard — Rich-based, PufferLib-style."""

import atexit
import dataclasses
import io
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from collections import deque
from typing import Any, Optional

import torch
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .charts import (
    dual_sparkline,
    labeled_sparkline,
    metric_bar,
)
from .history import HistoryTracker

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
SPARKLINE_WIDTH = 24


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


def getHw() -> tuple[float, float, float, float]:
    cpu = psutilMod.cpu_percent() if HAS_PSUTIL else 0.0
    # System RAM in use. "DRAM" is technically right (it is the memory chips,
    # not the SSD) but reads as jargon next to the disk, so the panel says RAM.
    ram = psutilMod.virtual_memory().percent if HAS_PSUTIL else 0.0
    gpu, vram = 0.0, 0.0

    if DT[2] == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = 100.0 * torch.cuda.memory_allocated(0) / props.total_memory
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu = float(
                pynvml.nvmlDeviceGetUtilizationRates(
                    pynvml.nvmlDeviceGetHandleByIndex(0)
                ).gpu
            )
        except Exception:
            gpu = 0.0
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
    return f"{DT[1]}: [cyan]{vram:.1f}%[/cyan]"


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
        if self.fd is not None and os.isatty(self.fd):
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

    def pollKey(self) -> Optional[str]:
        """Non-blocking check for a single keypress on stdin.
        Returns a single character, or '\x1b[A' for up-arrow, '\x1b[B' for down-arrow.
        """
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
            f"{DT[0]}: [cyan]{gpu:.1f}%[/cyan]",
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
            lossTable.add_row(name, f"{val:.4f}")

        monitor = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        monitor.add_column(min_width=30)
        monitor.add_column(min_width=34)
        monitor.add_row(summary, lossTable)

        # --- Universal charts: Episode Return + Episode Length sparklines ---
        universalLines: list[str] = []
        ep_returns = self.history.get("ep_return")
        ep_lengths = self.history.get("ep_length")
        if ep_returns:
            universalLines.append(
                labeled_sparkline(
                    "Episode Return", ep_returns, SPARKLINE_WIDTH, "green"
                )
            )
        if ep_lengths:
            universalLines.append(
                labeled_sparkline(
                    "Episode Length", ep_lengths, SPARKLINE_WIDTH, "yellow"
                )
            )

        # --- Algo-specific charts ---
        algoLines = self._renderAlgoCharts()

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

        # Universal charts section
        if universalLines:
            chartBody = Text()
            chartBody.append_text(
                Text.from_markup("[bold cyan]── Universal Metrics ──[/bold cyan]\n")
            )
            for line in universalLines:
                chartBody.append_text(Text.from_markup(line + "\n"))
            frame.add_row(chartBody)

        # Algo-specific charts section
        if algoLines:
            algoBody = Text()
            algoBody.append_text(
                Text.from_markup(f"[bold cyan]── {self.algo} Metrics ──[/bold cyan]\n")
            )
            for line in algoLines:
                algoBody.append_text(Text.from_markup(line + "\n"))
            frame.add_row(algoBody)

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

    def _renderAlgoCharts(self) -> list[str]:
        """Render algorithm-specific sparkline charts based on self.algo."""
        lines: list[str] = []
        algo = self.algo.lower()

        # ── DQN / Dueling-DQN ──
        if algo in ("dqn", "dueling-dqn"):
            eps = self.history.get("epsilon")
            if eps:
                lines.append(
                    labeled_sparkline("Epsilon (ε)", eps, SPARKLINE_WIDTH, "magenta")
                )

            q_mean = self.history.get("q_mean")
            if q_mean:
                lines.append(
                    labeled_sparkline("Mean Q-value", q_mean, SPARKLINE_WIDTH, "blue")
                )

            buf_fill = self.history.latest("buffer_fill")
            if buf_fill is not None:
                lines.append(metric_bar("Buffer fill %", buf_fill, 100.0, 12, "green"))

        # ── PPO ──
        elif algo == "ppo":
            entropy = self.history.get("entropy")
            if entropy:
                lines.append(
                    labeled_sparkline("Entropy", entropy, SPARKLINE_WIDTH, "yellow")
                )

            clip_frac = self.history.get("clip_fraction")
            if clip_frac:
                lines.append(
                    labeled_sparkline(
                        "Clip Fraction", clip_frac, SPARKLINE_WIDTH, "red", fmt=".3f"
                    )
                )

            kl = self.history.get("approx_kl")
            if kl:
                lines.append(
                    labeled_sparkline(
                        "KL Divergence", kl, SPARKLINE_WIDTH, "magenta", fmt=".4f"
                    )
                )

            val_loss = self.history.get("value_loss")
            if val_loss:
                lines.append(
                    labeled_sparkline("Value Loss", val_loss, SPARKLINE_WIDTH, "blue")
                )

        # ── SAC ──
        elif algo == "sac":
            alpha = self.history.get("alpha")
            if alpha:
                lines.append(
                    labeled_sparkline(
                        "Alpha (entropy)", alpha, SPARKLINE_WIDTH, "yellow", fmt=".4f"
                    )
                )

            q1 = self.history.get("q1_mean")
            q2 = self.history.get("q2_mean")
            if q1 and q2:
                dual = dual_sparkline(q1, q2, SPARKLINE_WIDTH, "cyan", "magenta")
                v1 = q1[-1] if q1 else 0
                v2 = q2[-1] if q2 else 0
                lines.append(f"  {'Q1/Q2 Mean':<18} {dual}  {v1:>7.2f}:{v2:<7.2f}")

            actor_loss = self.history.get("actor_loss")
            if actor_loss:
                lines.append(
                    labeled_sparkline(
                        "Actor Loss", actor_loss, SPARKLINE_WIDTH, "green"
                    )
                )

        # ── Universal loss sparkline ──
        loss_vals = self.history.get("total_loss")
        if loss_vals and len(loss_vals) >= 2:
            lines.append(
                labeled_sparkline(
                    "Training Loss", loss_vals, SPARKLINE_WIDTH, "red", fmt=".4f"
                )
            )

        return lines

    def renderChartsView(self) -> str:
        """Dedicated full-screen charts view with larger sparklines."""
        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white")
        header.add_row(
            f"NeatRL {_NEATRL_VERSION}  [{self.algo}]  [yellow][CHARTS][/yellow]  [dim][v: back][/dim]",
            f"Steps: {abbrev(self.lastAgentSteps)}  |  History: {HISTORY_MAXLEN} pts",
        )

        body = Text()
        wide = SPARKLINE_WIDTH + 12  # wider sparklines for detail view

        # ── Universal ──
        body.append_text(
            Text.from_markup("[bold cyan]══ Universal Metrics ══[/bold cyan]\n\n")
        )
        ep_ret = self.history.get("return")
        if ep_ret:
            body.append_text(
                Text.from_markup(
                    labeled_sparkline("Episode Return", ep_ret, wide, "green") + "\n"
                )
            )
        ep_len = self.history.get("ep_length")
        if ep_len:
            body.append_text(
                Text.from_markup(
                    labeled_sparkline("Episode Length", ep_len, wide, "yellow") + "\n"
                )
            )
        loss = self.history.get("total_loss")
        if loss:
            body.append_text(
                Text.from_markup(
                    labeled_sparkline("Training Loss", loss, wide, "red", fmt=".4f")
                    + "\n"
                )
            )

        # ── Algo-specific ──
        algo = self.algo.lower()
        body.append_text(
            Text.from_markup(f"\n[bold cyan]══ {self.algo} Metrics ══[/bold cyan]\n\n")
        )

        if algo in ("dqn", "dueling-dqn"):
            eps = self.history.get("epsilon")
            if eps:
                body.append_text(
                    Text.from_markup(
                        labeled_sparkline("Epsilon (ε)", eps, wide, "magenta") + "\n"
                    )
                )
            q_mean = self.history.get("q_mean")
            if q_mean:
                body.append_text(
                    Text.from_markup(
                        labeled_sparkline("Mean Q-value", q_mean, wide, "blue") + "\n"
                    )
                )
            buf = self.history.latest("buffer_fill")
            if buf is not None:
                body.append_text(
                    Text.from_markup(
                        metric_bar("Buffer fill %", buf, 100, 20, "green") + "\n"
                    )
                )

        elif algo == "ppo":
            for name, color, fmt_str in [
                ("entropy", "yellow", ".3f"),
                ("clip_fraction", "red", ".3f"),
                ("approx_kl", "magenta", ".4f"),
                ("value_loss", "blue", ".4f"),
            ]:
                vals = self.history.get(name)
                if vals:
                    body.append_text(
                        Text.from_markup(
                            labeled_sparkline(
                                name.replace("_", " ").title(),
                                vals,
                                wide,
                                color,
                                fmt_str,
                            )
                            + "\n"
                        )
                    )

        elif algo == "sac":
            alpha = self.history.get("alpha")
            if alpha:
                body.append_text(
                    Text.from_markup(
                        labeled_sparkline(
                            "Alpha (entropy)", alpha, wide, "yellow", ".4f"
                        )
                        + "\n"
                    )
                )
            q1 = self.history.get("q1_mean")
            q2 = self.history.get("q2_mean")
            if q1 and q2:
                dual = dual_sparkline(q1, q2, wide, "cyan", "magenta")
                body.append_text(
                    Text.from_markup(
                        f"  {'Q1 / Q2 Mean':<18} {dual}  {q1[-1]:>7.2f} : {q2[-1]:<7.2f}\n"
                    )
                )
            actor_loss = self.history.get("actor_loss")
            if actor_loss:
                body.append_text(
                    Text.from_markup(
                        labeled_sparkline("Actor Loss", actor_loss, wide, "green")
                        + "\n"
                    )
                )

        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(body)
        frame.add_row("[dim]press [b]v[/b] back  [b]l[/b] logs  [b]c[/b] config[/dim]")

        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def close(self, message: str = "Training complete.") -> None:
        if self.oldTerm is not None and self.fd is not None and os.isatty(self.fd):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.oldTerm)
        root = logging.getLogger()
        root.removeHandler(self.logHandler)
        for h in self.savedHandlers:
            root.addHandler(h)
        self.savedHandlers.clear()
        self.update(
            agent_steps=self.totalTimesteps,
            epoch=self.lastEpoch,
            losses=self.lastLosses,
            eval_stats=self.lastEvalStats,
            message=message,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
