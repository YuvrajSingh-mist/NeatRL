"""Shared Gymnasium observation wrappers used across all NeatRL algorithms."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class OneHotWrapper(gym.ObservationWrapper):
    """Wraps a discrete observation space into a one-hot float vector.

    Converts a ``Discrete(n)`` observation space to ``Box(n,)`` by encoding
    the integer observation as a one-hot float32 vector. Used to make grid
    environments (e.g. FrozenLake) compatible with MLP policies.
    """

    def __init__(self, env: gym.Env, obs_shape: int = 16) -> None:
        """Wrap *env* and set the new Box observation space.

        Args:
            env (gym.Env): The environment whose observations will be wrapped.
            obs_shape (int): Number of discrete states (= ``n`` in ``Discrete(n)``).
        """
        super().__init__(env)
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs: Any) -> np.ndarray:
        """Convert a discrete integer observation to a one-hot float32 vector.

        Args:
            obs (int | np.ndarray): Discrete observation from the wrapped environment.

        Returns:
            np.ndarray: One-hot encoded float32 array of shape ``(obs_shape,)``.
        """
        one_hot = torch.zeros(self.obs_shape, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()
