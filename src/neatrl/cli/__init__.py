"""NeatRL command-line interface -- terminal dashboard and training launcher.

``rich`` is the optional ``cli`` extra, so the dashboard import is guarded
here. Trainers import ``Dashboard`` from this package; without the extra it
is simply ``None`` and training runs with plain logging.
"""

try:
    from .dashboard import Dashboard, dashboardActive, progress
except ImportError:  # `rich` not installed: the dashboard is optional
    from tqdm import tqdm as _tqdm

    Dashboard = None

    def dashboardActive() -> bool:
        return False

    def progress(iterable, **kwargs):
        """Plain tqdm, for when there is no dashboard to protect."""
        kwargs.setdefault("disable", False)
        return _tqdm(iterable, **kwargs)


__all__ = ["Dashboard", "dashboardActive", "progress"]
