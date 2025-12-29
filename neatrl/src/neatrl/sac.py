import os
import random
import time
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

import wandb


LOG_STD_MAX = 2
LOG_STD_MIN = -20



# ===== CONFIGURATION =====
class Config:
    """Configuration class for SAC training."""

    # Experiment settings
    exp_name: str = "SAC-Experiment"
    seed: int = 42
    env_id: Optional[str] = "HalfCheetah-v5"

    # Training parameters
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter for target networks
    target_network_frequency: int = 1  # How often to update target networks
    batch_size: int = 256
    
    # SAC-specific parameters
    alpha: float = 0.2  # Entropy regularization coefficient
    autotune_alpha: bool = True  # Whether to automatically tune alpha
    target_entropy_scale: float = -1.0  # Target entropy = target_entropy_scale * action_dim
    
    learning_starts: int = 5000
    policy_frequency: int = 1  # How often to update the policy (1 = every step)

    # Logging & Saving
    capture_video: bool = True  # Whether to capture evaluation videos
    use_wandb: bool = True  # Whether to use Weights & Biases for logging
    wandb_project: str = "cleanRL"  # W&B project name
    wandb_entity: str = ""  # Your WandB username/team
    eval_every: int = 5000  # Frequency of evaluation during training (in steps)
    save_every: int = 10000  # Frequency of saving the model (in steps)
    num_eval_episodes: int = 10  # Number of evaluation episodes
    normalize_reward: bool = False  # Whether to normalize rewards
    normalize_obs: bool = False  # Whether to normalize observations
    atari_wrapper: bool = False  # Whether to use Atari preprocessing and frame stacking
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = (
        None  # Optional custom environment wrapper
    )
    grid_env: bool = False  # Whether it's a grid environment
    n_envs: int = 1  # Number of parallel environments for data collection
    max_grad_norm: float = (
        0.0  # Maximum gradient norm for gradient clipping (0.0 to disable)
    )
    log_gradients: bool = True  # Whether to log gradient norms to W&B
    device: str = "cpu"  # Device for training: "auto", "cpu", "cuda", or "cuda:0" etc.





class ActorNet(nn.Module):
    """Stochastic actor network for SAC that outputs mean and log_std of Gaussian policy."""
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space
      
   
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_space)
        self.log_std = nn.Linear(256, action_space)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action from the policy with reparameterization trick."""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)  # Output a single Q-value
        self.out = nn.Linear(256, 1)

    def forward(self, state, act):
        
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=-1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


class ActorNetCNN(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.mean = nn.Linear(512, action_space)
        self.log_std = nn.Linear(512, action_space)

    def forward(self, x):
        # x shape: (batch, height, width, channels) -> permute to (batch, channels, height, width)
        # x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action from the policy with reparameterization trick."""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
     
        
        return action, log_prob


class QNetCNN(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # State processing
        self.state_fc = nn.Linear(64 * 7 * 7, 512)

        # Action processing
        self.action_fc = nn.Linear(action_space, 512)

        # Combined processing
        self.combined_fc = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        # Process state through conv layers
        state = state.permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
        x = torch.nn.functional.relu(self.conv1(state))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        state_features = torch.nn.functional.relu(self.state_fc(x))

        # Process action
        action_features = torch.nn.functional.relu(self.action_fc(action))

        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        x = torch.nn.functional.relu(self.combined_fc(combined))
        return self.out(x)


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_shape: int = 16) -> None:
        super().__init__(env)
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs: Any) -> np.ndarray:
        one_hot = torch.zeros(self.obs_shape, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    render_mode: Optional[str] = None,
    grid_env: bool = False,
    atari_wrapper: bool = False,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    env: Optional[gym.Env] = None,
) -> Callable[[], gym.Env]:
    # Validate that only one of env_id or env is provided
    if env_id and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium environments or env for custom environments."
        )

    def thunk():
        if env is not None:
            # Use provided environment but still apply wrappers
            env_to_use = env
        else:
            # Create new environment
            env_to_use = gym.make(env_id, render_mode=render_mode)

        # Always apply RecordEpisodeStatistics if not already present
        if env is None:
            env_to_use = gym.wrappers.RecordEpisodeStatistics(env_to_use)

        if grid_env:
            env_to_use = OneHotWrapper(
                env_to_use, obs_shape=env_to_use.observation_space.n
            )

        if Config.normalize_reward:
            env_to_use = gym.wrappers.NormalizeReward(env_to_use)

        if Config.normalize_obs and not atari_wrapper:
            env_to_use = gym.wrappers.NormalizeObservation(env_to_use)
            
        if atari_wrapper:
            env_to_use = gym.wrappers.AtariPreprocessing(
                env_to_use, grayscale_obs=True, scale_obs=True
            )
            env_to_use = gym.wrappers.FrameStackObservation(env_to_use, stack_size=4)
            
        if env_wrapper:
            env_to_use = env_wrapper(env_to_use)
        env_to_use.action_space.seed(seed + idx)
        return env_to_use

    return thunk


def calculate_param_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of all parameters in a model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def validate_policy_network_dimensions(
    policy_network: nn.Module, obs_dim: Union[int, tuple[int, ...]], action_dim: int
) -> None:
    """
    Validate that the Policy-network's input and output dimensions match the environment.

    Args:
        policy_network: The neural network model (nn.Module)
        obs_dim: Expected observation dimension (int or tuple)
        action_dim: Expected action dimension
    """
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in policy_network.modules()
        )
        if not has_conv:
            print(
                "Warning: Observation is multi-dimensional but network has no Conv2d layers."
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
    critic_network: nn.Module, obs_dim: Union[int, tuple[int, ...]]
) -> None:
    """
    Validate that the Critic-network's input dimension matches the environment.

    Args:
        critic_network: The critic neural network model (nn.Module)
        obs_dim: Expected observation dimension (int or tuple)
        action_dim: Expected action dimension
    """
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in critic_network.modules()
        )
        if not has_conv:
            print(
                "Warning: Observation is multi-dimensional but network has no Conv2d layers."
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
    feature_network: nn.Module, obs_dim: Union[int, tuple[int, ...]], feature_dim: int
) -> None:
    """
    Validate that the Feature-network's input and output dimensions match expectations.

    Args:
        feature_network: The feature neural network model (nn.Module)
        obs_dim: Expected observation dimension (int or tuple)
        feature_dim: Expected feature dimension
    """
    if isinstance(obs_dim, tuple):
        # For Atari-like, check if it has conv layers
        has_conv = any(
            isinstance(module, nn.Conv2d) for module in feature_network.modules()
        )
        if not has_conv:
            print(
                "Warning: Observation is multi-dimensional but network has no Conv2d layers."
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


def evaluate(
    model: nn.Module,
    device: torch.device,
    env_id: str,
    env: Optional[gym.Env] = None,
    seed: int = 42,
    num_eval_eps: int = 5,
    record: bool = False,
    render_mode: Optional[str] = None,
    grid_env: bool = False,
    atari_wrapper: bool = False,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    use_wandb: bool = False,
) -> tuple[list[float], list[np.ndarray]]:
    # Validate that only one of env_id or env is provided
    if env_id and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium environments or env for custom environments."
        )

    # Create evaluation environment
    eval_env = make_env(
        env_id if env is None else "",
        seed,
        idx=0,
        render_mode=render_mode,
        grid_env=grid_env,
        atari_wrapper=atari_wrapper,
        env_wrapper=env_wrapper,
        env=env,
    )()
    model.eval()
    returns = []
    frames = []

    for _ in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        rewards = 0.0

        while not done:
            if record:
                frame = eval_env.render()
                frames.append(frame)
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    np.array(obs), device=device, dtype=torch.float32
                ).unsqueeze(0)
                # For SAC, use stochastic action during evaluation
                action, _ = model.get_action(obs_tensor)
                action = action.cpu().numpy()
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    # For continuous action spaces, ensure it's 1D
                    action = action.flatten()

            obs, rewards_curr, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            rewards += rewards_curr

        returns.append(rewards)

    # Save video
    if frames and use_wandb:
        video = np.stack(frames)

        video = np.transpose(video, (0, 3, 1, 2))

        wandb.log(
            {
                "videos/eval_policy": wandb.Video(
                    video,
                    fps=30,
                    format="mp4",
                )
            }
        )
        frames = []
    eval_env.close()
    model.train()

    return returns, frames


def train_sac(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = None,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    learning_rate: float = Config.learning_rate,
    buffer_size: int = Config.buffer_size,
    gamma: float = Config.gamma,
    tau: float = Config.tau,
    target_network_frequency: int = Config.target_network_frequency,
    batch_size: int = Config.batch_size,
    alpha: float = Config.alpha,
    autotune_alpha: bool = Config.autotune_alpha,
    target_entropy_scale: float = Config.target_entropy_scale,
    learning_starts: int = Config.learning_starts,
    policy_frequency: int = Config.policy_frequency,
    capture_video: bool = Config.capture_video,
    use_wandb: bool = Config.use_wandb,
    wandb_project: str = Config.wandb_project,
    wandb_entity: str = Config.wandb_entity,
    exp_name: str = Config.exp_name,
    eval_every: int = Config.eval_every,
    save_every: int = Config.save_every,
    num_eval_episodes: int = Config.num_eval_episodes,
    n_envs: int = Config.n_envs,
    max_grad_norm: float = Config.max_grad_norm,
    log_gradients: bool = Config.log_gradients,
    device: str = Config.device,
    grid_env: bool = False,
    atari_wrapper: bool = False,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    normalize_obs: bool = Config.normalize_obs,
    normalize_reward: bool = Config.normalize_reward,
    actor_class: Any = ActorNet,
    q_network_class: Any = QNet,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id or env.spec.id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.learning_rate = learning_rate
    Config.buffer_size = buffer_size
    Config.gamma = gamma
    Config.tau = tau
    Config.target_network_frequency = target_network_frequency
    Config.batch_size = batch_size
    Config.alpha = alpha
    Config.autotune_alpha = autotune_alpha
    Config.target_entropy_scale = target_entropy_scale
    Config.learning_starts = learning_starts
    Config.policy_frequency = policy_frequency
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.wandb_entity = wandb_entity
    Config.exp_name = exp_name
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.num_eval_episodes = num_eval_episodes
    Config.n_envs = n_envs
    Config.max_grad_norm = max_grad_norm
    Config.log_gradients = log_gradients
    Config.device = device
    Config.grid_env = grid_env
    Config.atari_wrapper = atari_wrapper
    Config.env_wrapper = env_wrapper
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward

    # Validate that only one of env_id or env is provided
    if env_id is not None and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium environments or env for custom environments."
        )

    if Config.capture_video and not Config.use_wandb:
        raise ValueError(
            "Cannot capture video without WandB enabled. Set use_wandb=True to upload videos."
        )

    run_name = f"{Config.env_id}_{Config.seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    os.makedirs(f"videos/{run_name}/train", exist_ok=True)
    os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            entity=Config.wandb_entity,
            config=vars(Config()),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Set seeds
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    device = torch.device(Config.device)

    # Create environments - check for pre-created env first, then default
    if env is not None:
        env_thunks = [
            make_env(
                "",
                Config.seed,
                i,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                env=env,
                render_mode="rgb_array",
            )
            for i in range(Config.n_envs)
        ]
    else:
        # Use default environment creation
        env_thunks = [
            make_env(
                Config.env_id,
                Config.seed,
                i,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                render_mode="rgb_array",
            )
            for i in range(Config.n_envs)
        ]

    envs = gym.vector.SyncVectorEnv(env_thunks)
    if isinstance(envs.single_observation_space, gym.spaces.Discrete):
        obs_space_shape = (envs.single_observation_space.n,)
    else:
        obs_space_shape = envs.single_observation_space.shape

    action_space_n = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else envs.single_action_space.shape[0]
    )

    action_shape = (
        ()
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else (action_space_n,)
    )

    print(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")

    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(actor_class, obs_space_shape, action_shape)
        actor_net = actor_class.to(device)
    else:
        # Use actor class
        actor_net = actor_class(obs_space_shape, action_space_n)
        actor_net = actor_net.to(device)
        
    # Create twin critic networks (SAC uses two Q-networks)
    if isinstance(q_network_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(q_network_class, obs_space_shape, action_space_n)
        q1_network = q_network_class.to(device)
        q2_network = q_network_class.to(device)
    else:
        # Use critic class
        q1_network = q_network_class(obs_space_shape, action_space_n).to(device)
        q2_network = q_network_class(obs_space_shape, action_space_n).to(device)

    # Create target Q-networks
    target_q1_network = q_network_class(obs_space_shape, action_space_n).to(device)
    target_q2_network = q_network_class(obs_space_shape, action_space_n).to(device)
    target_q1_network.load_state_dict(q1_network.state_dict())
    target_q2_network.load_state_dict(q2_network.state_dict())

    # Print network architecture
    print("Actor Network Architecture:")
    print(actor_net)
    print("\nQ1 Network Architecture:")
    print(q1_network)
    print("\nQ2 Network Architecture:")
    print(q2_network)

    # Optimizers
    actor_optim = optim.Adam(actor_net.parameters(), lr=Config.learning_rate)
    q1_optim = optim.Adam(q1_network.parameters(), lr=Config.learning_rate)
    q2_optim = optim.Adam(q2_network.parameters(), lr=Config.learning_rate)
    
    # Automatic entropy tuning
    if Config.autotune_alpha:
        
        target_entropy = Config.target_entropy_scale * action_space_n
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = optim.Adam([log_alpha], lr=Config.learning_rate)
    else:
        alpha = Config.alpha

    # Set networks to training mode
    q1_network.train()
    q2_network.train()
    actor_net.train()

    # Replay buffer
    replay_buffer = ReplayBuffer(
        Config.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
        n_envs=Config.n_envs,
    )

    obs, _ = envs.reset()
    start_time = time.time()

    for step in tqdm(range(Config.total_timesteps)):
        # Sample action from stochastic policy
        with torch.no_grad():
            action, _ = actor_net.get_action(
                torch.tensor(obs, device=device, dtype=torch.float32)
            )

        action_np = action.cpu().numpy()
        new_obs, reward, terminated, truncated, info = envs.step(action_np)
        done = np.logical_or(terminated, truncated)
        replay_buffer.add(
            obs, new_obs, action_np, np.array(reward), np.array(done), [info]
        )

        # Training step
        if step > Config.learning_starts:
            data = replay_buffer.sample(Config.batch_size)
      
            # Update Q-networks
            with torch.no_grad():
                next_actions, next_log_probs = actor_net.get_action(
                    data.next_observations.to(torch.float32)
                )
                
                next_log_probs = next_log_probs.sum(dim=-1, keepdim=True) if len(next_log_probs.shape) > 1 else next_log_probs
                
                target_q1 = target_q1_network(
                    data.next_observations.to(torch.float32), next_actions
                )
                target_q2 = target_q2_network(
                    data.next_observations.to(torch.float32), next_actions
                )
                min_target_q = torch.min(target_q1, target_q2)
                td_target = data.rewards + Config.gamma * (min_target_q - alpha * next_log_probs) * (1 - data.dones)

            # Update Q1
            q1_optim.zero_grad()
            current_q1 = q1_network(
                data.observations.to(torch.float32), data.actions.to(torch.float32)
            )
            
            q1_loss = nn.functional.mse_loss(current_q1, td_target)
            q1_loss.backward()
            q1_optim.step()
            
            # Update Q2
            q2_optim.zero_grad()
            current_q2 = q2_network(
                data.observations.to(torch.float32), data.actions.to(torch.float32)
            )
            q2_loss = nn.functional.mse_loss(current_q2, td_target)
            q2_loss.backward()
            q2_optim.step()

            # Update policy
            if step % Config.policy_frequency == 0:
                actor_optim.zero_grad()
                new_actions, log_probs = actor_net.get_action(
                    data.observations.to(torch.float32)
                )
                q1_new = q1_network(data.observations.to(torch.float32), new_actions)
                q2_new = q2_network(data.observations.to(torch.float32), new_actions)
                min_q_new = torch.min(q1_new, q2_new)
                
                actor_loss = (alpha * log_probs - min_q_new).mean()
                actor_loss.backward()
                actor_optim.step()
                
                # Calculate entropy for logging
                entropy = -log_probs.mean().item()
                
                # Update alpha (temperature parameter)
                alpha_loss_value = None
                if Config.autotune_alpha:
                    alpha_optim.zero_grad()
                    alpha_loss = (-log_alpha.exp() * (log_probs + target_entropy).detach()).mean()
                    alpha_loss_value = alpha_loss.item()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()

            if step % Config.target_network_frequency == 0:
                
                # Soft update target networks
                for param, target_param in zip(q1_network.parameters(), target_q1_network.parameters()):
                    target_param.data.copy_(Config.tau * param.data + (1 - Config.tau) * target_param.data)
                for param, target_param in zip(q2_network.parameters(), target_q2_network.parameters()):
                    target_param.data.copy_(Config.tau * param.data + (1 - Config.tau) * target_param.data)

            # Log gradient norm per layer for critics
            if Config.use_wandb and Config.log_gradients:
                for name, param in q1_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/q1_layer_{name}": grad_norm,
                                "global_step": step,
                            }
                        )
                for name, param in q2_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/q2_layer_{name}": grad_norm,
                                "global_step": step,
                            }
                        )
                        
            # Log training metrics
            if Config.use_wandb and step % 100 == 0:
                log_dict = {
                    "losses/q1_loss": q1_loss.item(),
                    "losses/q2_loss": q2_loss.item(),
                    "entropy/alpha": alpha,
                    "global_step": step,
                }
                
                # Log actor loss and entropy when policy is updated
                if step % Config.policy_frequency == 0:
                    log_dict["losses/actor_loss"] = actor_loss.item()
                    log_dict["entropy/entropy"] = entropy
                    if Config.autotune_alpha and alpha_loss_value is not None:
                        log_dict["losses/alpha_loss"] = alpha_loss_value
                        log_dict["entropy/target_entropy"] = target_entropy
                
                wandb.log(log_dict)

            # Log episode returns
            if "episode" in info:
                if Config.n_envs > 1:
                    for i in range(Config.n_envs):
                        if done[i]:
                            ep_ret = info["episode"]["r"][i]
                            ep_len = info["episode"]["l"][i]

                            if Config.use_wandb:
                                wandb.log(
                                    {
                                        "charts/episodic_return": ep_ret,
                                        "charts/episodic_length": ep_len,
                                        "global_step": step,
                                    }
                                )
                else:
                    if done:
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        if Config.use_wandb:
                            wandb.log(
                                {
                                    "charts/episodic_return": ep_ret,
                                    "charts/episodic_length": ep_len,
                                    "global_step": step,
                                }
                            )

            # Log losses and metrics
            if step % 1000 == 0 and step > Config.learning_starts:
                sps = int(step / (time.time() - start_time))
                print(
                    f"Step {step}, Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, Alpha: {alpha:.4f}, Actor Loss: {actor_loss.item():.4f}, SPS: {sps}"
                )

            # Evaluation
            if Config.eval_every > 0 and step % Config.eval_every == 0:
                # eval_env_id = "" if env is not None else Config.env_id
                # eval_env = env
                episodic_returns, eval_frames = evaluate(
                    actor_net,
                    device,
                    Config.env_id,
                    env=None,
                    seed=Config.seed,
                    num_eval_eps=Config.num_eval_episodes,
                    record=Config.capture_video,
                    render_mode="rgb_array" if Config.capture_video else None,
                    grid_env=Config.grid_env,
                    atari_wrapper=Config.atari_wrapper,
                    env_wrapper=Config.env_wrapper,
                    use_wandb=Config.use_wandb,
                )
                avg_return = np.mean(episodic_returns)

                if Config.use_wandb:
                    wandb.log(
                        {
                            "val_episodic_returns": episodic_returns,
                            "charts/val_avg_return": avg_return,
                            "charts/val_return_std": np.std(episodic_returns),
                            "val_step": step,
                        }
                    )
                print(
                    f"Evaluation returns: {[float(r) for r in episodic_returns]}, Average: {avg_return:.2f}"
                )

                # Save video if frames were captured
                if eval_frames and Config.use_wandb:
                    video = np.stack(eval_frames)
                    video = np.transpose(video, (0, 3, 1, 2))
                    wandb.log(
                        {
                            "videos/eval_policy": wandb.Video(
                                video,
                                fps=30,
                                format="mp4",
                            )
                        }
                    )

        # Save model
        if Config.save_every > 0 and step % Config.save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/model_step_{step}.pth"
            torch.save(
                {
                    "actor": actor_net.state_dict(),
                    "q1_network": q1_network.state_dict(),
                    "q2_network": q2_network.state_dict(),
                    "actor": actor_net.state_dict(),
                    "target_q1_network": target_q1_network.state_dict(),
                    "target_q2_network": target_q2_network.state_dict(),
                },
                model_path,
            )
            print(f"Model saved at step {step} to {model_path}")

        if done.all():
            obs, _ = envs.reset()
        else:
            obs = new_obs

    # Final evaluation and video saving
    if Config.use_wandb:
        print("Capturing final evaluation video...")
        eval_env_id = "" if env is not None else Config.env_id
        eval_env = env
        episodic_returns, eval_frames = evaluate(
            actor_net,
            device,
            eval_env_id,
            env=eval_env,
            record=True,
            render_mode="rgb_array",
            grid_env=Config.grid_env,
            atari_wrapper=Config.atari_wrapper,
            env_wrapper=Config.env_wrapper,
            use_wandb=Config.use_wandb,
        )

        if eval_frames:
            train_video_path = f"videos/final_{Config.env_id}.mp4"
            imageio.mimsave(train_video_path, eval_frames, fps=30, codec="libx264")
            print(f"Final training video saved to {train_video_path}")
            wandb.finish()

    envs.close()
    return actor_net


def train_sac_cnn(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = None,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    learning_rate: float = Config.learning_rate,
    buffer_size: int = Config.buffer_size,
    gamma: float = Config.gamma,
    tau: float = Config.tau,
    target_network_frequency: int = Config.target_network_frequency,
    batch_size: int = Config.batch_size,
    alpha: float = Config.alpha,
    autotune_alpha: bool = Config.autotune_alpha,
    target_entropy_scale: float = Config.target_entropy_scale,
    learning_starts: int = Config.learning_starts,
    policy_frequency: int = Config.policy_frequency,
    capture_video: bool = Config.capture_video,
    use_wandb: bool = Config.use_wandb,
    wandb_project: str = Config.wandb_project,
    wandb_entity: str = Config.wandb_entity,
    exp_name: str = Config.exp_name,
    eval_every: int = Config.eval_every,
    save_every: int = Config.save_every,
    num_eval_episodes: int = Config.num_eval_episodes,
    n_envs: int = Config.n_envs,
    max_grad_norm: float = Config.max_grad_norm,
    log_gradients: bool = Config.log_gradients,
    device: str = Config.device,
    grid_env: bool = False,
    atari_wrapper: bool = False,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    normalize_reward: bool = Config.normalize_reward,
    actor_class: Any = ActorNetCNN,
    q_network_class: Any = QNetCNN,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id or env.spec.id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.learning_rate = learning_rate
    Config.buffer_size = buffer_size
    Config.gamma = gamma
    Config.tau = tau
    Config.target_network_frequency = target_network_frequency
    Config.batch_size = batch_size
    Config.alpha = alpha
    Config.autotune_alpha = autotune_alpha
    Config.target_entropy_scale = target_entropy_scale
    Config.learning_starts = learning_starts
    Config.policy_frequency = policy_frequency
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.wandb_entity = wandb_entity
    Config.exp_name = exp_name
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.num_eval_episodes = num_eval_episodes
    Config.n_envs = n_envs
    Config.max_grad_norm = max_grad_norm
    Config.log_gradients = log_gradients
    Config.device = device
    Config.grid_env = grid_env
    Config.atari_wrapper = atari_wrapper
    Config.env_wrapper = env_wrapper
    Config.normalize_reward = normalize_reward

    # Validate that only one of env_id or env is provided
    if env_id is not None and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium environments or env for custom environments."
        )

    if Config.capture_video and not Config.use_wandb:
        raise ValueError(
            "Cannot capture video without WandB enabled. Set use_wandb=True to upload videos."
        )

    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    os.makedirs(f"videos/{run_name}/train", exist_ok=True)
    os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            entity=Config.wandb_entity,
            config=vars(Config()),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Set seeds
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    device = torch.device(Config.device)

    # Create environments - check for pre-created env first, then default
    if env is not None:
        env_thunks = [
            make_env(
                "",
                Config.seed,
                i,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                env=env,
                render_mode="rgb_array",
            )
            for i in range(Config.n_envs)
        ]
    else:
        # Use default environment creation
        env_thunks = [
            make_env(
                Config.env_id,
                Config.seed,
                i,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                render_mode="rgb_array",
            )
            for i in range(Config.n_envs)
        ]

    envs = gym.vector.SyncVectorEnv(env_thunks)
    if isinstance(envs.single_observation_space, gym.spaces.Discrete):
        obs_space_shape = (envs.single_observation_space.n,)
    else:
        obs_space_shape = envs.single_observation_space.shape

    action_space_n = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else envs.single_action_space.shape[0] if isinstance(envs.single_action_space, gym.spaces.Box) else envs.single_action_space.shape
    )

    action_shape = (
        ()
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else (action_space_n,)
    )

    print(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")


    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(actor_class, obs_space_shape, action_shape)
        actor_net = actor_class.to(device)
    else:
        # Use actor class
        actor_net = actor_class(obs_space_shape, action_space_n).to(device)
    # Create twin critic networks (SAC uses two Q-networks)
    if isinstance(q_network_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(q_network_class, obs_space_shape, action_space_n)
        q1_network = q_network_class.to(device)
        q2_network = q_network_class.to(device)
    else:
        # Use critic class
        q1_network = q_network_class(obs_space_shape, action_space_n).to(device)
        q2_network = q_network_class(obs_space_shape, action_space_n).to(device)

    # Create target Q-networks
    target_q1_network = q_network_class(obs_space_shape, action_space_n).to(device)
    target_q2_network = q_network_class(obs_space_shape, action_space_n).to(device)

    target_q1_network.load_state_dict(q1_network.state_dict())
    target_q2_network.load_state_dict(q2_network.state_dict())

    # Print network architecture
    print("Actor Network Architecture:")
    print(actor_net)
    print("\nQ1 Network Architecture:")
    print(q1_network)
    print("\nQ2 Network Architecture:")
    print(q2_network)

    # Optimizers
    actor_optim = optim.Adam(actor_net.parameters(), lr=Config.learning_rate)
    q1_optim = optim.Adam(q1_network.parameters(), lr=Config.learning_rate)
    q2_optim = optim.Adam(q2_network.parameters(), lr=Config.learning_rate)
    
    # Automatic entropy tuning
    if Config.autotune_alpha:
        
        target_entropy = Config.target_entropy_scale * action_space_n
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = optim.Adam([log_alpha], lr=Config.learning_rate)
    else:
        alpha = Config.alpha

    # Set networks to training mode
    q1_network.train()
    q2_network.train()
    actor_net.train()

    # Replay buffer

    replay_buffer = ReplayBuffer(
        Config.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
        n_envs=Config.n_envs,
    )

    obs, _ = envs.reset()
    start_time = time.time()

    for step in tqdm(range(Config.total_timesteps)):
        # Sample action from stochastic policy
        with torch.no_grad():
            action, log_probs = actor_net.get_action(
                torch.tensor(obs, device=device, dtype=torch.float32)
            )
        action_np = action.cpu().numpy()
      
        new_obs, reward, terminated, truncated, info = envs.step(action_np)
        done = np.logical_or(terminated, truncated)
        replay_buffer.add(
            obs, new_obs, action_np, np.array(reward), np.array(done), [info]
        )

        # Training step
        if step > Config.learning_starts:
            data = replay_buffer.sample(Config.batch_size)
            actions = data.actions.squeeze(-1)  # Squeeze to (batch_size,) for discrete actions
            
           
            # Update Q-networks
            with torch.no_grad():
                next_actions, next_log_probs = actor_net.get_action(
                    data.next_observations.to(torch.float32)
                )
                next_log_probs = next_log_probs.sum(dim=-1, keepdim=True) if len(next_log_probs.shape) > 1 else next_log_probs
                
               
                target_q1 = target_q1_network(
                    data.next_observations.to(torch.float32), next_actions.to(torch.float32)
                )
                target_q2 = target_q2_network(
                    data.next_observations.to(torch.float32), next_actions.to(torch.float32)
                )
             
                min_target_q = torch.min(target_q1, target_q2)
               
                td_target = data.rewards + Config.gamma * (min_target_q - alpha * next_log_probs) * (1 - data.dones)
               
            # Update Q1
            q1_optim.zero_grad()
            current_q1 = q1_network(
                data.observations.to(torch.float32), actions.to(torch.float32)
            )
           
            q1_loss = nn.functional.mse_loss(current_q1, td_target)
            q1_loss.backward()
            q1_optim.step()
            
            # Update Q2
            q2_optim.zero_grad()
            current_q2 = q2_network(
                data.observations.to(torch.float32), actions.to(torch.float32)
            )
            q2_loss = nn.functional.mse_loss(current_q2, td_target)
            q2_loss.backward()
            q2_optim.step()

            # Update policy
            if step % Config.policy_frequency == 0:
                actor_optim.zero_grad()
                new_actions, log_probs = actor_net.get_action(
                    data.observations.to(torch.float32)
                )
                q1_new = q1_network(data.observations.to(torch.float32), new_actions)
                q2_new = q2_network(data.observations.to(torch.float32), new_actions)
                min_q_new = torch.min(q1_new, q2_new)
                
                actor_loss = (alpha * log_probs - min_q_new).mean()
                actor_loss.backward()
                actor_optim.step()
                
                # Calculate entropy for logging
                entropy = -log_probs.mean().item()
                
                # Update alpha (temperature parameter)
                alpha_loss_value = None
                if Config.autotune_alpha:
                    alpha_optim.zero_grad()
                    alpha_loss = (-log_alpha.exp() * (log_probs + target_entropy).detach()).mean()
                    alpha_loss_value = alpha_loss.item()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()

            # Soft update target networks
            if step % Config.target_network_frequency == 0:
                for param, target_param in zip(q1_network.parameters(), target_q1_network.parameters()):
                    target_param.data.copy_(Config.tau * param.data + (1 - Config.tau) * target_param.data)
                for param, target_param in zip(q2_network.parameters(), target_q2_network.parameters()):
                    target_param.data.copy_(Config.tau * param.data + (1 - Config.tau) * target_param.data)

            # Log gradient norm per layer for critics
            if Config.use_wandb and Config.log_gradients:
                for name, param in q1_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/q1_layer_{name}": grad_norm,
                                "global_step": step,
                            }
                        )
                for name, param in q2_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/q2_layer_{name}": grad_norm,
                                "global_step": step,
                            }
                        )
                        
            # Log training metrics
            if Config.use_wandb and step % 100 == 0:
                log_dict = {
                    "losses/q1_loss": q1_loss.item(),
                    "losses/q2_loss": q2_loss.item(),
                    "entropy/alpha": alpha,
                    "global_step": step,
                }
                
                # Log actor loss and entropy when policy is updated
                if step % Config.policy_frequency == 0:
                    log_dict["losses/actor_loss"] = actor_loss.item()
                    log_dict["entropy/entropy"] = entropy
                    if Config.autotune_alpha and alpha_loss_value is not None:
                        log_dict["losses/alpha_loss"] = alpha_loss_value
                        log_dict["entropy/target_entropy"] = target_entropy
                
                wandb.log(log_dict)

            # Log episode returns
            if "episode" in info:
                if Config.n_envs > 1:
                    for i in range(Config.n_envs):
                        if done[i]:
                            ep_ret = info["episode"]["r"][i]
                            ep_len = info["episode"]["l"][i]

                            if Config.use_wandb:
                                wandb.log(
                                    {
                                        "charts/episodic_return": ep_ret,
                                        "charts/episodic_length": ep_len,
                                        "global_step": step,
                                    }
                                )
                else:
                    if done:
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        if Config.use_wandb:
                            wandb.log(
                                {
                                    "charts/episodic_return": ep_ret,
                                    "charts/episodic_length": ep_len,
                                    "global_step": step,
                                }
                            )

            # Log losses and metrics
            if step % 1000 == 0 and step > Config.learning_starts:
                sps = int(step / (time.time() - start_time))
                print(
                    f"Step {step}, Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, Alpha: {alpha:.4f}, SPS: {sps}"
                )

            # Evaluation
            if Config.eval_every > 0 and step % Config.eval_every == 0:
                # eval_env_id = "" if env is not None else Config.env_id
                # eval_env = env
                episodic_returns, eval_frames = evaluate(
                    actor_net,
                    device,
                    Config.env_id,
                    env=None,
                    seed=Config.seed,
                    num_eval_eps=Config.num_eval_episodes,
                    record=Config.capture_video,
                    render_mode="rgb_array" if Config.capture_video else None,
                    grid_env=Config.grid_env,
                    atari_wrapper=Config.atari_wrapper,
                    env_wrapper=Config.env_wrapper,
                    use_wandb=Config.use_wandb,
                )
                avg_return = np.mean(episodic_returns)

                if Config.use_wandb:
                    wandb.log(
                        {
                            "val_episodic_returns": episodic_returns,
                            "charts/val_avg_return": avg_return,
                            "charts/val_return_std": np.std(episodic_returns),
                            "val_step": step,
                        }
                    )
                print(
                    f"Evaluation returns: {[float(r) for r in episodic_returns]}, Average: {avg_return:.2f}"
                )

                # Save video if frames were captured
                if eval_frames and Config.use_wandb:
                    video = np.stack(eval_frames)
                    video = np.transpose(video, (0, 3, 1, 2))
                    wandb.log(
                        {
                            "videos/eval_policy": wandb.Video(
                                video,
                                fps=30,
                                format="mp4",
                            )
                        }
                    )

        # Save model
        if Config.save_every > 0 and step % Config.save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/model_step_{step}.pth"
            torch.save(
                {
                    "actor": actor_net.state_dict(),
                    "q1_network": q1_network.state_dict(),
                    "q2_network": q2_network.state_dict(),
                    "actor": actor_net.state_dict(),
                    "target_q1_network": target_q1_network.state_dict(),
                    "target_q2_network": target_q2_network.state_dict(),
                },
                model_path,
            )
            print(f"Model saved at step {step} to {model_path}")

        if done.all():
            obs, _ = envs.reset()
        else:
            obs = new_obs

    # Final evaluation and video saving
    if Config.use_wandb:
        print("Capturing final evaluation video...")
        eval_env_id = "" if env is not None else Config.env_id
        eval_env = env
        episodic_returns, eval_frames = evaluate(
            actor_net,
            device,
            eval_env_id,
            env=eval_env,
            record=True,
            render_mode="rgb_array",
            grid_env=Config.grid_env,
            atari_wrapper=Config.atari_wrapper,
            env_wrapper=Config.env_wrapper,
            use_wandb=Config.use_wandb,
        )

        if eval_frames:
            train_video_path = f"videos/final_{Config.env_id}.mp4"
            imageio.mimsave(train_video_path, eval_frames, fps=30, codec="libx264")
            print(f"Final training video saved to {train_video_path}")
            wandb.finish()

    env.close()
    return actor_net


# --- Main Execution ---
if __name__ == "__main__":
    train_sac()
