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


def setup_device(
    device: Union[str, torch.device] = "cpu", seed: int = 0
) -> torch.device:
    """Resolve a device string to a ``torch.device``, with a safe fallback."""
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
    "get_space_dims",
    "setup_device",
    "validate_critic_network_dimensions",
    "validate_dueling_q_network_dimensions",
    "validate_feature_network_dimensions",
    "validate_policy_network_dimensions",
    "validate_q_network_dimensions",
]
