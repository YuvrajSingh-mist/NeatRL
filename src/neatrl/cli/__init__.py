"""NeatRL command-line interface -- terminal dashboard and training launcher.

``rich`` is the optional ``cli`` extra, so the dashboard import is guarded
here. Trainers import ``Dashboard`` from this package; without the extra it
is simply ``None`` and training runs with plain logging.
"""

try:
    from .dashboard import Dashboard
except ImportError:  # `rich` not installed: the dashboard is optional
    Dashboard = None

__all__ = ["Dashboard"]
