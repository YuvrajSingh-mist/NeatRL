"""Rich-based sparkline chart renderers for terminal dashboard."""

from typing import Optional

BAR_CHARS = " ▁▂▃▄▅▆▇█"


def _scale(values: list[float], width: int) -> list[float]:
    """Normalize values to [0, 1] range."""
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    rng = mx - mn
    if rng == 0:
        return [0.5] * min(len(values), width)
    # Down-sample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return [(v - mn) / rng for v in sampled]


def sparkline(values: list[float], width: int = 20, color: str = "cyan") -> str:
    """Render a single-line sparkline from a list of values.

    Args:
        values: The time-series values.
        width: Width in characters.
        color: Rich color tag (e.g. "cyan", "green", "yellow").

    Returns:
        A Rich-markup string of the sparkline.
    """
    if not values:
        return f"[{color}]{'─' * width}[/{color}]"
    scaled = _scale(values, width)
    chars = [BAR_CHARS[min(int(v * 8), 8)] for v in scaled]
    return f"[{color}]{''.join(chars)}[/{color}]"


def dual_sparkline(
    values1: list[float],
    values2: list[float],
    width: int = 20,
    color1: str = "cyan",
    color2: str = "magenta",
) -> str:
    """Render two overlaid sparklines (useful for Q1/Q2 comparison)."""
    if not values1 and not values2:
        return f"[dim]{'─' * width}[/dim]"
    all_vals = values1 + values2
    mn = min(all_vals)
    mx = max(all_vals)
    rng = mx - mn if mx != mn else 1.0

    def sample(vals: list[float]) -> list[float]:
        if len(vals) > width:
            step = len(vals) / width
            return [vals[int(i * step)] for i in range(width)]
        return vals

    s1 = sample(values1)
    s2 = sample(values2)
    # Pad to width
    s1 = s1 + [s1[-1]] * (width - len(s1)) if s1 else [mn] * width
    s2 = s2 + [s2[-1]] * (width - len(s2)) if s2 else [mn] * width

    result = ""
    for i in range(min(width, len(s1), len(s2))):
        v1 = int((s1[i] - mn) / rng * 8)
        v2 = int((s2[i] - mn) / rng * 8)
        if v1 >= v2:
            result += f"[{color1}]{BAR_CHARS[min(v1, 8)]}[/{color1}]"
        else:
            result += f"[{color2}]{BAR_CHARS[min(v2, 8)]}[/{color2}]"
    return result


def labeled_sparkline(
    label: str,
    values: list[float],
    width: int = 20,
    color: str = "cyan",
    fmt: str = ".2f",
) -> str:
    """Sparkline with label and current value.

    Returns a string like: "  Episode Return  ▃▄▅▆▇█  42.00"
    """
    latest = values[-1] if values else 0.0
    sl = sparkline(values, width, color)
    val_str = f"{latest:{fmt}}"
    return f"  {label:<18} {sl}  {val_str:>8}"


def metric_bar(
    label: str,
    value: float,
    max_val: float,
    width: int = 12,
    color: str = "green",
) -> str:
    """Single horizontal bar for instantaneous metrics (e.g. buffer fill %).

    Returns something like: "  Buffer fill   ██████░░░░  65%"
    """
    frac = min(max(value / max(max_val, 1e-10), 0.0), 1.0)
    filled = int(frac * width)
    empty = width - filled
    pct = int(frac * 100)
    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    return f"  {label:<18} {bar}  {pct:>3}%"


def make_chart_section(title: str, lines: list[str]) -> str:
    """Wrap chart lines in a Rich panel-style section."""
    header = f"[bold cyan]{title}[/bold cyan]"
    body = "\n".join(lines)
    return f"{header}\n{body}"


def sparkline_header(
    columns: list[tuple[str, int, str]],  # (label, width, color)
) -> str:
    """Generate an aligned header row for sparkline columns."""
    parts = []
    for label, width, color in columns:
        parts.append(f"[{color}]{label:^{width}}[/{color}]")
    return "  " + "  ".join(parts)
