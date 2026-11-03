"""Shared utilities: logging configuration, device setup, and environment-space helpers."""

import logging
import os
from typing import Any, Optional, Union

import gymnasium as gym
import torch

from .nn_utils import (
    calculate_param_norm,
    validate_critic_network_dimensions,
    validate_dueling_q_network_dimensions,
    validate_feature_network_dimensions,
    validate_policy_network_dimensions,
    validate_q_network_dimensions,
)
from .wrappers import OneHotWrapper

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler and optional file output."""
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
    """Return a named logger, creating it if needed."""
    return logging.getLogger(name)


def get_space_dims(env: Any) -> tuple[int, int]:
    """Return observation and action space dimensions as ints."""
    if hasattr(env, "single_observation_space"):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space
    obs_dim = (
        int(obs_space.n)  # type: ignore[attr-defined]
        if isinstance(obs_space, gym.spaces.Discrete)
        else int(obs_space.shape[0])  # type: ignore[index]
    )
    act_dim = (
        int(act_space.n)  # type: ignore[attr-defined]
        if isinstance(act_space, gym.spaces.Discrete)
        else int(act_space.shape[0])  # type: ignore[index]
    )
    return obs_dim, act_dim


def _mps_usable() -> bool:
    """Whether Apple MPS is present *and* actually works on this machine.

    ``is_available()`` alone is not enough: it reports True on builds where the
    first kernel launch fails, and on macOS versions older than 12.3. Probing
    with a real tiny op is the only reliable check, so ``auto`` never selects a
    backend that would crash later inside a training step.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        probe = torch.zeros(1, device="mps")
        return bool((probe + 1).item() == 1.0)
    except Exception:  # noqa: BLE001 - any runtime failure means "not usable"
        return False


def resolve_device_string(requested: Union[str, torch.device]) -> str:
    """Resolve ``"auto"`` to the best available backend name.

    Order is CUDA, then Apple MPS, then CPU. Anything that is not ``"auto"`` is
    passed through untouched, so explicit choices always win.
    """
    if isinstance(requested, torch.device):
        return requested.type
    if str(requested).lower() != "auto":
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if _mps_usable():
        return "mps"
    return "cpu"


def setup_device(
    device: Union[str, torch.device] = "auto", seed: int = 0
) -> torch.device:
    """Resolve a device string to a ``torch.device``, with a safe fallback.

    ``"auto"`` (the default) picks CUDA when present, then Apple MPS, then CPU.
    An explicit ``"cuda"`` or ``"mps"`` that is unavailable falls back to CPU
    with a warning rather than raising, so a config written for a GPU box still
    runs elsewhere.
    """
    _dev = torch.device(resolve_device_string(device))
    if _dev.type == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA not available, falling back to CPU")
        _dev = torch.device("cpu")
    if _dev.type == "mps" and not _mps_usable():
        logging.getLogger(__name__).warning("MPS not available, falling back to CPU")
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
    "get_space_dims",
    "resolve_device_string",
    "setup_device",
    "validate_critic_network_dimensions",
    "validate_dueling_q_network_dimensions",
    "validate_feature_network_dimensions",
    "validate_policy_network_dimensions",
    "validate_q_network_dimensions",
]
