"""Ensure src/ is on sys.path for editable installs."""
import sys
from pathlib import Path

src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
