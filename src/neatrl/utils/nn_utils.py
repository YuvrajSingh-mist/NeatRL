"""Utility functions for neural network validation and parameter inspection."""

import logging
from typing import Optional, Union

import torch.nn as nn

logger = logging.getLogger(__name__)


def calculate_param_norm(model: nn.Module) -> float:
    """Compute the total L2 norm of trainable parameters."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def validate_q_network_dimensions(
    q_network: nn.Module, obs_dim: int, action_dim: int
) -> None:
    """Validate Q-network input/output dims against the env spaces."""
    # Find first Linear layer for input dimension
    first_layer = None
    for module in q_network.modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
    if first_layer is None:
        raise ValueError(
            "Q-network must have at least one Linear layer for dimension validation."
        )
    if first_layer.in_features != obs_dim:
        raise ValueError(
            f"Q-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
        )

    # Find last Linear layer for output dimension
    last_layer = None
    for module in reversed(list(q_network.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise ValueError(
            "Q-network must have at least one Linear layer for dimension validation."
        )
    if last_layer.out_features != action_dim:
        raise ValueError(
            f"Q-network output dimension {last_layer.out_features} does not match action dimension {action_dim}."
        )


def validate_dueling_q_network_dimensions(
    dueling_q_network: nn.Module, obs_dim: int, action_dim: int
) -> None:
    """Validate Dueling Q-network input/output dims against the env spaces."""
    # Find first Linear layer for input dimension
    first_layer = None
    for module in dueling_q_network.modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
    if first_layer is None:
        raise ValueError(
            "Dueling Q-network must have at least one Linear layer for dimension validation."
        )
    if first_layer.in_features != obs_dim:
        raise ValueError(
            f"Dueling Q-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
        )

    # Find the advantage stream's last Linear layer for output dimension
    adv_layer = None
    for module in dueling_q_network.adv.modules():  # type: ignore[union-attr]
        if isinstance(module, nn.Linear):
            adv_layer = module
    if adv_layer is None:
        raise ValueError(
            "Dueling Q-network must have an advantage stream with Linear layers for dimension validation."
        )
    if adv_layer.out_features != action_dim:
        raise ValueError(
            f"Dueling Q-network output dimension {adv_layer.out_features} does not match action dimension {action_dim}."
        )


def validate_policy_network_dimensions(
    policy_network: nn.Module,
    obs_dim: Union[int, tuple[int, ...]],
    action_dim: int,
) -> None:
    """Validate policy-network input/output dims against the env spaces."""
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in policy_network.modules()
        )
        if not has_conv:
            logger.warning(
                "Observation is multi-dimensional but network has no Conv2d layers."
            )
    else:
        # Find first Linear layer for input dimension
        first_layer = None
        for module in policy_network.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            raise ValueError(
                "Policy-network must have at least one Linear layer for dimension validation."
            )
        if first_layer.in_features != obs_dim:
            raise ValueError(
                f"Policy-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
            )

    # Find last Linear layer for output dimension
    last_layer = None
    for module in reversed(list(policy_network.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise ValueError(
            "Policy-network must have at least one Linear layer for dimension validation."
        )
    if last_layer.out_features != action_dim:
        raise ValueError(
            f"Policy-network output dimension {last_layer.out_features} does not match action dimension {action_dim}."
        )


def validate_critic_network_dimensions(
    critic_network: nn.Module,
    obs_dim: Union[int, tuple[int, ...]],
    action_dim: Optional[int] = None,
) -> None:
    """Validate critic-network input dim against the observation space."""
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in critic_network.modules()
        )
        if not has_conv:
            logger.warning(
                "Observation is multi-dimensional but network has no Conv2d layers."
            )
    else:
        # Find first Linear layer for input dimension
        first_layer = None
        for module in critic_network.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            raise ValueError(
                "Critic-network must have at least one Linear layer for dimension validation."
            )
        if first_layer.in_features != obs_dim:
            raise ValueError(
                f"Critic-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
            )

    # Find last Linear layer for output dimension (should be 1 for value)
    last_layer = None
    for module in reversed(list(critic_network.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise ValueError(
            "Critic-network must have at least one Linear layer for dimension validation."
        )
    if last_layer.out_features != 1:
        raise ValueError(
            f"Critic-network output dimension {last_layer.out_features} should be 1 for value estimation."
        )


def validate_feature_network_dimensions(
    feature_network: nn.Module,
    obs_dim: Union[int, tuple[int, ...]],
    feature_dim: int,
) -> None:
    """Validate feature-network input/output dims against the env spaces."""
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in feature_network.modules()
        )
        if not has_conv:
            logger.warning(
                "Observation is multi-dimensional but network has no Conv2d layers."
            )
    else:
        # Find first Linear layer for input dimension
        first_layer = None
        for module in feature_network.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            raise ValueError(
                "Feature-network must have at least one Linear layer for dimension validation."
            )
        if first_layer.in_features != obs_dim:
            raise ValueError(
                f"Feature-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
            )

    # Find last Linear layer for output dimension
    last_layer = None
    for module in reversed(list(feature_network.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise ValueError(
            "Feature-network must have at least one Linear layer for dimension validation."
        )
    if last_layer.out_features != feature_dim:
        raise ValueError(
            f"Feature-network output dimension {last_layer.out_features} does not match expected feature dimension {feature_dim}."
        )
