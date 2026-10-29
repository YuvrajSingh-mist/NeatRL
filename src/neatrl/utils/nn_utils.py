"""Utility functions for neural network validation and parameter inspection."""

import logging
from typing import Optional, Union

import torch
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
    q_network: nn.Module,
    obs_dim: Union[int, tuple[int, ...]],
    action_dim: int,
) -> None:
    """Validate Q-network input/output dims against the env spaces."""
    if isinstance(obs_dim, tuple):
        # For Atari-like image inputs, require a convolutional trunk.
        has_conv = any(isinstance(module, nn.Conv2d) for module in q_network.modules())
        if not has_conv:
            logger.warning(
                "Observation is multi-dimensional but network has no Conv2d layers."
            )
    else:
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


class ImageFeatures(nn.Module):
    """Convolutional trunk that infers its output size for any image geometry.

    The CNN trainers used to hard-code the flattened size for 84x84 Atari
    frames (`64 * 7 * 7`), so any other resolution -- CarRacing's 96x96, for
    one -- raised a shape-mismatch error on the first forward pass. This trunk
    measures its own output instead, and accepts both channel-first
    (Atari preprocessing) and channel-last (CarRacing) observations.
    """

    def __init__(self, obs_shape: tuple[int, ...]) -> None:
        super().__init__()
        shape = tuple(int(s) for s in obs_shape)
        if len(shape) == 2:
            shape = (1, *shape)
        if len(shape) != 3:
            raise ValueError(f"expected an image observation shape, got {obs_shape}")

        # Atari preprocessing yields (C, H, W); CarRacing yields (H, W, C).
        if shape[0] in (1, 3, 4):
            self.channel_last = False
            channels, height, width = shape
        else:
            self.channel_last = True
            height, width, channels = shape

        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, channels, height, width)
            self.out_dim = int(self.trunk(probe).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode a batch of frames into a flat feature vector."""
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.max() > 1.5:  # uint8 frames
            x = x / 255.0
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        return self.trunk(x)


def observation_input_layer(
    obs_shape: Union[int, tuple[int, ...]], out_features: int
) -> nn.Module:
    """Build the first layer of a network for a vector or image observation.

    Image observations get a convolutional trunk followed by a linear
    projection; flat observations keep the original single ``nn.Linear``.
    """
    shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
    if len(shape) >= 2:
        encoder = ImageFeatures(shape)
        return nn.Sequential(encoder, nn.Linear(encoder.out_dim, out_features))
    return nn.Linear(int(shape[0]), out_features)
