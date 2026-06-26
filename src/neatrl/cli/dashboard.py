"""NeatRL terminal training dashboard — Rich-based, PufferLib-style."""

import sys
import time
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _abbrev(n: float) -> str:
    """Format a large number with K / M / B suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def _duration(secs: float) -> str:
    """Format a duration in seconds as a human-readable string."""
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


def _hw() -> tuple[float, float, float, float]:
    """Return ``(cpu_pct, gpu_pct, dram_pct, vram_pct)``."""
    cpu = _psutil.cpu_percent() if _HAS_PSUTIL else 0.0
    dram = _psutil.virtual_memory().percent if _HAS_PSUTIL else 0.0
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = 100.0 * torch.cuda.memory_allocated(0) / props.total_memory
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            gpu = float(
                pynvml.nvmlDeviceGetUtilizationRates(
                    pynvml.nvmlDeviceGetHandleByIndex(0)
                ).gpu
            )
        except Exception:
            gpu = 0.0
    else:
        gpu = 0.0
        vram = 0.0
    return cpu, gpu, dram, vram


class Dashboard:
    """In-place terminal training dashboard.

    Renders a live Rich-based panel that updates in-place each call to
    ``update()``.  Replaces tqdm progress output during training.

    Example usage::

        dash = Dashboard("PPO", "CartPole-v1", total_timesteps=500_000)
        for update in range(num_updates):
            # ... training step ...
            dash.update(
                agent_steps=global_step,
                epoch=update,
                losses={"policy_loss": p_loss, "value_loss": v_loss},
            )
        dash.close()
    """

    def __init__(self, algo: str, env_id: str, total_timesteps: int) -> None:
        """Initialise the dashboard.

        Args:
            algo (str): Algorithm name shown in the header (e.g. ``"PPO"``).
            env_id (str): Gymnasium environment ID shown in the summary panel.
            total_timesteps (int): Total steps used to compute remaining-time estimate.
        """
        self.algo = algo
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self._t0 = time.time()
        self._console = Console(highlight=False)
        self._prev_lines = 0
        self._first = True

    def update(
        self,
        *,
        agent_steps: int,
        epoch: int,
        losses: dict,
        eval_stats: Optional[dict] = None,
        message: str = "",
    ) -> None:
        """Render a fresh dashboard frame to stdout, overwriting the previous one.

        Args:
            agent_steps (int): Total environment steps taken so far.
            epoch (int): Current training epoch or update number.
            losses (dict): ``{name: float}`` mapping displayed in the right panel.
                Any keys are accepted (e.g. ``"policy_loss"``, ``"td_loss"``).
            eval_stats (dict | None): Optional ``{name: float}`` mapping shown in the
                bottom eval panel. Pass ``None`` to omit the panel.
            message (str): One-line status message shown at the bottom of the frame.
        """
        elapsed = time.time() - self._t0
        sps = max(int(agent_steps / elapsed), 1) if elapsed > 0 else 1
        remaining = max(self.total_timesteps - agent_steps, 0) / sps

        cpu, gpu, dram, vram = _hw()

        # ── Header row ─────────────────────────────────────────────────
        header = Table(box=None, show_header=False, padding=(0, 1))
        header.add_column(style="bold cyan", min_width=22)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_column(style="white", min_width=13)
        header.add_row(
            f"NeatRL 1.0.0  [{self.algo}]",
            f"CPU: [cyan]{cpu:.1f}%[/cyan]",
            f"GPU: [cyan]{gpu:.1f}%[/cyan]",
            f"DRAM: [cyan]{dram:.1f}%[/cyan]",
            f"VRAM: [cyan]{vram:.1f}%[/cyan]",
        )

        # ── Summary panel (left) ────────────────────────────────────────
        summary = Table(box=None, show_header=True, padding=(0, 1))
        summary.add_column("[cyan]Summary[/cyan]", style="bright_white", min_width=14)
        summary.add_column("[cyan]Value[/cyan]", style="white", min_width=12)
        env_label = self.env_id if len(self.env_id) <= 14 else self.env_id[:12] + "…"
        summary.add_row("Environment", env_label)
        summary.add_row("Agent Steps", _abbrev(agent_steps))
        summary.add_row("SPS", _abbrev(sps))
        summary.add_row("Epoch", str(epoch))
        summary.add_row("Uptime", _duration(elapsed))
        summary.add_row("Remaining", _duration(remaining))

        # ── Losses panel (right) ────────────────────────────────────────
        loss_table = Table(box=None, show_header=True, padding=(0, 1))
        loss_table.add_column("[cyan]Losses[/cyan]", style="bright_white", min_width=20)
        loss_table.add_column("[cyan]Value[/cyan]", style="white", min_width=10)
        for name, val in losses.items():
            loss_table.add_row(name, f"{val:.4f}")

        # ── Main monitor row ────────────────────────────────────────────
        monitor = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        monitor.add_column(min_width=30)
        monitor.add_column(min_width=34)
        monitor.add_row(summary, loss_table)

        # ── Eval stats (optional bottom panel) ─────────────────────────
        bottom: Optional[Table] = None
        if eval_stats:
            items = list(eval_stats.items())
            half = (len(items) + 1) // 2

            def _stat_col(rows: list) -> Table:
                t = Table(box=None, show_header=True, padding=(0, 1))
                t.add_column(
                    "[cyan]Eval Stats[/cyan]", style="bright_white", min_width=22
                )
                t.add_column("[cyan]Value[/cyan]", style="white", min_width=10)
                for k, v in rows:
                    t.add_row(k, f"{v:.3f}")
                return t

            bottom = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            bottom.add_column(min_width=36)
            bottom.add_column(min_width=36)
            bottom.add_row(_stat_col(items[:half]), _stat_col(items[half:]))

        # ── Outer rounded frame ─────────────────────────────────────────
        frame = Table(box=box.ROUNDED, show_header=False, padding=(0, 1), expand=False)
        frame.add_column()
        frame.add_row(header)
        frame.add_row(monitor)
        if bottom is not None:
            frame.add_row(bottom)
        if message:
            frame.add_row(f"[dim]Message: {message}[/dim]")

        # ── Render and overwrite previous frame ─────────────────────────
        with self._console.capture() as cap:
            self._console.print(frame)
        rendered = cap.get()
        lines = rendered.count("\n")

        if self._first:
            sys.stdout.write(rendered)
            self._first = False
        else:
            sys.stdout.write(f"\033[{self._prev_lines}A\033[J" + rendered)
        sys.stdout.flush()
        self._prev_lines = lines

    def close(self, message: str = "Training complete.") -> None:
        """Print a final frame and leave the cursor below the dashboard.

        Args:
            message (str): Final status line shown in the dashboard footer.
        """
        self.update(
            agent_steps=self.total_timesteps,
            epoch=0,
            losses={},
            message=message,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
