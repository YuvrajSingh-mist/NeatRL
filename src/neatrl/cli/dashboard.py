"""NeatRL terminal training dashboard — Rich-based, PufferLib-style."""

import atexit
import dataclasses
import logging
import os
import re
import select
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

try:
    import psutil as psutilMod
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

LOGBUFSIZE = 500
LOG_VISIBLE = 30


def deviceTag() -> tuple[str, str, str]:
    if torch.cuda.is_available():
        return "GPU", "VRAM", "cuda"
    if torch.backends.mps.is_available():
        return "GPU", "VRAM", "mps"
    return "GPU", "VRAM", "cpu"


DT = deviceTag()


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


def getHw() -> tuple[float, float, float, float]:
    cpu = psutilMod.cpu_percent() if HAS_PSUTIL else 0.0
    dram = psutilMod.virtual_memory().percent if HAS_PSUTIL else 0.0
    gpu, vram = 0.0, 0.0

    if DT[2] == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = 100.0 * torch.cuda.memory_allocated(0) / props.total_memory
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu = float(pynvml.nvmlDeviceGetUtilizationRates(
                pynvml.nvmlDeviceGetHandleByIndex(0)).gpu)
        except Exception:
            gpu = 0.0
    elif DT[2] == "mps":
        if HAS_PSUTIL:
            proc = psutilMod.Process()
            mem = proc.memory_info()
            total = psutilMod.virtual_memory().total
            vram = 100.0 * mem.rss / total if total > 0 else 0.0
        gpu = 0.0
    return cpu, gpu, dram, vram


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
        self.prevLines = 0
        self.isFirst = True
        self.showLogs = False
        self.showConfig = False
        self.lastRenderTime = 0.0
        self.lastEpoch = 0
        self.lastLosses: dict = {}
        self.lastEvalStats: Optional[dict] = None
        self.logScrollOffset = 0  # 0 = newest, positive = scroll back

        # Terminal raw mode
        self.fd = sys.stdin.fileno()
        self.oldTerm = None
        if os.isatty(self.fd):
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
            if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
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
        scrollInfo = f"  [dim](\u2191\u2193 {self.logScrollOffset})[/dim]" if self.logScrollOffset else ""
        header.add_row(
            f"NeatRL 1.0.0  [{self.algo}]  [yellow][LOGS][/yellow]  [dim][l: back][/dim]",
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
            f"NeatRL 1.0.0  [{self.algo}]  [yellow][CONFIG][/yellow]  [dim][l: logs][c: back][/dim]",
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
            self.isFirst = True
            self.logScrollOffset = 0
        elif key == "c":
            self.showConfig = not self.showConfig
            self.showLogs = False
            self.isFirst = True
        elif key == "\x1b[A":  # up arrow
            if self.showLogs:
                self.logScrollOffset = min(self.logScrollOffset + 1, max(0, len(self.logBuf) - 1))
        elif key == "\x1b[B":  # down arrow
            if self.showLogs:
                self.logScrollOffset = max(0, self.logScrollOffset - 1)

        self.lastEpoch = epoch
        self.lastLosses = losses
        self.lastEvalStats = eval_stats

        now = time.time()
        if not self.isFirst and now - self.lastRenderTime < 0.1:
            return
        self.lastRenderTime = now

        # Full clear on view switch — paired right before the render
        if self.isFirst:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

        if self.showLogs:
            rendered = self.renderLogs()
        elif self.showConfig:
            rendered = self.renderConfig()
        else:
            rendered = self.renderDash(agent_steps, epoch, losses, eval_stats, message)

        lines = rendered.count("\n")
        if lines < self.prevLines:
            rendered += "\n" * (self.prevLines - lines)
            lines = self.prevLines
        if self.isFirst:
            sys.stdout.write(rendered)
            self.isFirst = False
        else:
            sys.stdout.write(f"\033[{self.prevLines}A\033[J" + rendered)
        sys.stdout.flush()
        self.prevLines = lines

    def renderDash(
        self, agent_steps: int, epoch: int, losses: dict,
        eval_stats: Optional[dict], message: str,
    ) -> str:
        elapsed = time.time() - self.t0
        sps = max(int(agent_steps / elapsed), 1) if elapsed > 0 else 1
        remaining = max(self.totalTimesteps - agent_steps, 0) / sps
        cpu, gpu, dram, vram = getHw()

        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_row(
            f"NeatRL 1.0.0  [{self.algo}]  [dim][l: logs][c: config][/dim]",
            f"CPU: [cyan]{cpu:.1f}%[/cyan]",
            f"{DT[0]}: [cyan]{gpu:.1f}%[/cyan]",
            f"DRAM: [cyan]{dram:.1f}%[/cyan]",
            f"{DT[1]}: [cyan]{vram:.1f}%[/cyan]",
        )

        summary = Table(box=None, show_header=True, padding=(0, 1))
        summary.add_column("[cyan]Summary[/cyan]", style="bright_white", min_width=14)
        summary.add_column("[cyan]Value[/cyan]", style="white", min_width=12)
        envLabel = self.envId if len(self.envId) <= 14 else self.envId[:12] + "..."
        summary.add_row("Environment", envLabel)
        summary.add_row("Agent Steps", abbrev(agent_steps))
        summary.add_row("SPS", abbrev(sps))
        summary.add_row("Epoch", str(epoch))
        lastRet = next((v for k, v in (eval_stats or {}).items() if k == "last_return"), "-")
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

        bottom: Optional[Table] = None
        if eval_stats:
            bottomStats = {k: v for k, v in eval_stats.items() if k != "last_return"}
            if bottomStats:
                items = list(bottomStats.items())
                half = (len(items) + 1) // 2

                def col(rows: list) -> Table:
                    t = Table(box=None, show_header=True, padding=(0, 1))
                    t.add_column("[cyan]Eval Stats[/cyan]", style="bright_white", min_width=22)
                    t.add_column("[cyan]Value[/cyan]", style="white", min_width=10)
                    for k, v in rows:
                        t.add_row(k, f"{v:.3f}")
                    return t

                bottom = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                bottom.add_column(min_width=36)
                bottom.add_column(min_width=36)
                bottom.add_row(col(items[:half]), col(items[half:]))

        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(monitor)

        # Show message (model params etc.) as a table in the bottom area
        if message:
            msgTable = Table(box=None, show_header=True, padding=(0, 1))
            msgTable.add_column("[cyan]Model[/cyan]", style="bright_white", min_width=22)
            msgTable.add_column("[cyan]Value[/cyan]", style="white", min_width=20)
            parts = message.split(" | ")
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    msgTable.add_row(k.strip(), v.strip())
                else:
                    msgTable.add_row("Info", part)

            if bottom is not None:
                merged = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                merged.add_column(min_width=36)
                merged.add_column(min_width=36)
                merged.add_row(bottom, msgTable)
                frame.add_row(merged)
            else:
                frame.add_row(msgTable)
        elif bottom is not None:
            frame.add_row(bottom)

        frame.add_row("[dim]press [b]l[/b] logs  [b]c[/b] config[/dim]")
        frame.add_row("[dim]press [b]l[/b] logs  [b]c[/b] config[/dim]")

        with self.console.capture() as cap:
            self.console.print(frame)
        return cap.get()

    def close(self, message: str = "Training complete.") -> None:
        if self.oldTerm is not None and os.isatty(self.fd):
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
