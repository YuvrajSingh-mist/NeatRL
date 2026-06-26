"""NeatRL: clean, readable reinforcement learning algorithm implementations."""

import warnings

warnings.filterwarnings(
    "ignore", message="The pynvml package is deprecated", category=FutureWarning
)

from .a2c_cnn import train_a2c_cnn
from .a2c_mlp import train_a2c
from .ddpg_cnn import train_ddpg_cnn
from .ddpg_mlp import train_ddpg
from .dqn_mlp import train_dqn
from .dueling_dqn_mlp import train_dueling_dqn
from .ppo_cnn import train_ppo_cnn
from .ppo_mlp import train_ppo
from .reinforce_cnn import train_reinforce_cnn
from .reinforce_mlp import train_reinforce
from .rnd_cnn import train_ppo_rnd_cnn
from .rnd_mlp import train_ppo_rnd
from .sac_cnn import train_sac_cnn
from .sac_mlp import train_sac
from .td3_cnn import train_td3_cnn
from .td3_mlp import train_td3

__all__ = [
    "train_dqn",
    "train_reinforce",
    "train_dueling_dqn",
    "train_ppo_rnd",
    "train_ppo_rnd_cnn",
    "train_reinforce_cnn",
    "train_ppo",
    "train_ppo_cnn",
    "train_ddpg",
    "train_ddpg_cnn",
    "train_a2c",
    "train_a2c_cnn",
    "train_sac",
    "train_sac_cnn",
    "train_td3",
    "train_td3_cnn",
]
