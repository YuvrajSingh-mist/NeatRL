import logging
import os
from typing import Optional, Union

import torch

from .nn_utils import (
    calculate_param_norm,
    validate_critic_network_dimensions,
    validate_dueling_q_network_dimensions,
    validate_feature_network_dimensions,
    validate_policy_network_dimensions,
    validate_q_network_dimensions,
)

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logger with console output and optional file sink.

    Safe to call multiple times — handlers are only added once.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_device(
    device: Union[str, torch.device] = "cpu", seed: int = 0
) -> torch.device:
    """Resolve device string to torch.device, fall back to CPU if CUDA unavailable,
    and configure hardware-specific determinism and seeds."""
    _dev = torch.device(device)
    if _dev.type == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA not available, falling back to CPU")
        _dev = torch.device("cpu")
    if _dev.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    elif _dev.type == "mps":
        torch.mps.manual_seed(seed)
    return _dev


__all__ = [
    "calculate_param_norm",
    "configure_logging",
    "get_logger",
    "setup_device",
    "validate_critic_network_dimensions",
    "validate_dueling_q_network_dimensions",
    "validate_feature_network_dimensions",
    "validate_policy_network_dimensions",
    "validate_q_network_dimensions",
]
