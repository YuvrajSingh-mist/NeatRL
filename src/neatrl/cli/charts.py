"""Chart renderers for the NeatRL terminal dashboard.

Two primitives: :func:`sparkline` for one series, and :func:`chart_cell` for the
two-line block the charts view tiles into a grid. Everything here returns Rich
markup, never a `Text`, so callers decide how to compose it.
"""

BAR_CHARS = "▁▂▃▄▅▆▇█"
# Index of the "no variation" level, used for a flat series.
BAR_FLAT = 3
BAR_MAX = len(BAR_CHARS) - 1

# metric key -> (label, colour, number format). Dict order is display order;
# anything pushed that is not listed falls back to a label derived from its key,
# so a new `push_metrics(foo=...)` shows up without editing this table.
METRICS: dict[str, tuple[str, str, str]] = {
    "ep_return": ("Episode Return", "green", ".2f"),
    "ep_length": ("Episode Length", "yellow", ".0f"),
    "total_loss": ("Training Loss", "red", ".4f"),
    "sps": ("Steps / sec", "cyan", ".0f"),
    "epsilon": ("Epsilon (\u03b5)", "magenta", ".4f"),
    "q_mean": ("Mean Q-value", "blue", ".3f"),
    "buffer_fill": ("Buffer Fill", "green", ".0f"),
    "entropy": ("Entropy", "yellow", ".4f"),
    "clip_fraction": ("Clip Fraction", "red", ".3f"),
    "approx_kl": ("KL Divergence", "magenta", ".5f"),
    "value_loss": ("Value Loss", "blue", ".4f"),
    "actor_loss": ("Actor Loss", "green", ".4f"),
    "critic_loss": ("Critic Loss", "blue", ".4f"),
    "policy_loss": ("Policy Loss", "red", ".4f"),
    "alpha": ("Alpha", "yellow", ".4f"),
    "q1_mean": ("Q1 Mean", "cyan", ".3f"),
    "q2_mean": ("Q2 Mean", "magenta", ".3f"),
}

FALLBACK_COLORS = ("cyan", "green", "yellow", "magenta", "blue", "red")


def describe(name: str) -> tuple[str, str, str]:
    """Label, colour and number format for one metric key."""
    known = METRICS.get(name)
    if known is not None:
        return known
    label = name.replace("_", " ").strip().title() or name
    color = FALLBACK_COLORS[sum(map(ord, name)) % len(FALLBACK_COLORS)]
    return label, color, ".3f"


def compact(value: float, fmt: str = ".2f") -> str:
    """Short form of a number, for a cell with a fixed width.

    A loss of 299886496.0 rendered as ".4f" is 15 characters of noise; at that
    magnitude the digits below the leading decimal carry no information.
    """
    if value != value:  # NaN
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.1f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:.1f}K"
    return format(value, fmt)


def _bucket(values: list[float], width: int) -> list[float]:
    """Reduce ``values`` to ``width`` points by averaging equal buckets.

    Averaging rather than striding is what stops the chart shimmering: a
    strided sample picks a different element out of every bucket the moment one
    new value arrives, so every column jumps on every frame. An averaged bucket
    moves by a fraction of one value instead.
    """
    n = len(values)
    if n <= width:
        return list(values)
    out: list[float] = []
    for i in range(width):
        lo = i * n // width
        hi = max((i + 1) * n // width, lo + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def sparkline(values: list[float], width: int = 20, color: str = "cyan") -> str:
    """One line of block characters, scaled to this series' own range.

    A short series is right-aligned, so the trace grows in from the right the
    way a scrolling chart does instead of stretching as data arrives.
    """
    if not values:
        return f"[{color}]{'─' * width}[/{color}]"
    sampled = _bucket(values, width)
    lo = min(sampled)
    span = max(sampled) - lo
    if span <= 0:
        chars = BAR_CHARS[BAR_FLAT] * len(sampled)
    else:
        chars = "".join(
            BAR_CHARS[min(int((v - lo) / span * len(BAR_CHARS)), BAR_MAX)]
            for v in sampled
        )
    return f"[{color}]{chars.rjust(width)}[/{color}]"


def chart_cell(name: str, values: list[float], width: int) -> list[str]:
    """Two lines for one grid column: label with its latest value, then the trace."""
    text, color, fmt = describe(name)
    value = compact(values[-1], fmt) if values else "-"
    room = max(width - len(value) - 1, 4)
    if len(text) > room:
        text = text[: room - 1] + "\u2026"
    pad = " " * max(width - len(text) - len(value), 1)
    return [
        f"{text}{pad}[bold {color}]{value}[/]",
        sparkline(values, width, color),
    ]


def progress_bar(done: int, total: int, width: int = 30, color: str = "green") -> str:
    """Horizontal progress bar with the percentage, e.g. ``████░░░░  9.8%``."""
    if total <= 0:
        return f"[dim]{'─' * width}[/dim]  [bold]n/a[/]"
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(frac * width)
    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"
    return f"{bar} [bold]{frac * 100:5.1f}%[/bold]"
