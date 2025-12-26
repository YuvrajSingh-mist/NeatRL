import os
import random
import time
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import wandb


# ===== CONFIGURATION =====
class Config:
    """Configuration class for A2C training."""

    # Experiment settings
    exp_name: str = "A2C"  # Experiment name for logging and saving
    seed: int = 42  # Random seed for reproducibility
    env_id: Optional[str] = "LunarLander-v3"  # Gymnasium environment ID
    total_timesteps: int = 1000000  # Total timesteps for training (for compatibility)

    # A2C & Agent settings
    lr: float = 2e-3  # Learning rate for optimizer
    gamma: float = 0.99  # Discount factor for rewards
    VALUE_COEFF: float = (
        0.5  # Value loss coefficient in total loss (not used in pure A2C)
    )
    num_eval_episodes: int = 10  # Number of evaluation episodes
    max_grad_norm: float = (
        0.0  # Maximum gradient norm for gradient clipping (0.0 to disable)
    )

    # PPO-specific (kept for compatibility but not used in pure A2C)
    n_envs: int = 1  # Number of parallel environments
    update_epochs: int = 1  # Update epochs (A2C uses 1)
    clip_value: float = 0.2  # Not used in A2C
    ENTROPY_COEFF: float = 0.01  # Not used in pure A2C
    anneal_lr: bool = False  # Learning rate annealing

    # Logging & Saving
    capture_video: bool = True  # Whether to capture evaluation videos
    use_wandb: bool = True  # Whether to use Weights & Biases for logging
    wandb_project: str = "cleanRL"  # W&B project name
    grid_env: bool = False  # Whether the environment uses discrete grid observations (applies OneHot wrapper)
    eval_every: int = 10000  # Frequency of evaluation during training (in updates)
    save_every: int = 10000  # Frequency of saving the model (in updates)
    atari_wrapper: bool = (
        False  # Whether to apply Atari preprocessing and frame stacking
    )
    normalize_obs: bool = False  # Whether to normalize observations
    normalize_reward: bool = False  # Whether to normalize rewards
    log_gradients: bool = False  # Whether to log gradient norms to W&B
    device: str = "cpu"  # Device for training: "auto", "cpu", "cuda", or "cuda:0" etc.
    custom_agent: Optional[nn.Module] = None  # Custom neural network class or instance
    env: Optional[gym.Env] = None  # Optional pre-created environment


# ===== RunningMeanStd for Normalization =====
class RunningMeanStd:
    """Tracks running mean and std of a data stream using Welford's algorithm."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# --- Networks ---
def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNet(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()

        self.fc1 = layer_init(nn.Linear(state_space, 32))
        self.fc2 = layer_init(nn.Linear(32, 32))
        self.fc3 = layer_init(nn.Linear(32, 16))
        self.out = layer_init(nn.Linear(16, action_space))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        logits = self.out(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist

    def get_action(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        dist = self.forward(x)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 32))
        self.fc2 = layer_init(nn.Linear(32, 32))
        self.fc3 = layer_init(nn.Linear(32, 16))
        self.value = layer_init(nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)


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
    def thunk():
        if env is not None:
            # Use provided environment
            env_to_use = env

        else:
            # Create new environment
            env_to_use = gym.make(env_id, render_mode=render_mode)

        env_to_use = gym.wrappers.RecordEpisodeStatistics(env_to_use)
        if Config.normalize_reward:
            env_to_use = gym.wrappers.NormalizeReward(env_to_use)
        if grid_env:
            env_to_use = OneHotWrapper(
                env_to_use, obs_shape=env_to_use.observation_space.n
            )
        if Config.normalize_obs:
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
                obs = torch.tensor(obs, device=device, dtype=torch.float32)
                action, _, _ = model.get_action(obs)
                action = action.cpu().numpy()
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()

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


def train_a2c(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = Config.env,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    gamma: float = Config.gamma,
    n_envs: int = Config.n_envs,
    ENTROPY_COEFF: float = Config.ENTROPY_COEFF,
    capture_video: bool = Config.capture_video,
    use_wandb: bool = Config.use_wandb,
    wandb_project: str = Config.wandb_project,
    exp_name: str = Config.exp_name,
    grid_env: bool = Config.grid_env,
    max_grad_norm: float = Config.max_grad_norm,
    eval_every: int = Config.eval_every,
    save_every: int = Config.save_every,
    atari_wrapper: bool = Config.atari_wrapper,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    actor_class: Any = ActorNet,
    critic_class: Any = CriticNet,
    log_gradients: bool = Config.log_gradients,
    num_eval_episodes: int = Config.num_eval_episodes,
    anneal_lr: bool = Config.anneal_lr,
    normalize_obs: bool = Config.normalize_obs,
    normalize_reward: bool = Config.normalize_reward,
    device: str = Config.device,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id if env is None else env.spec.id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.lr = lr
    Config.gamma = gamma
    Config.n_envs = n_envs
    Config.ENTROPY_COEFF = ENTROPY_COEFF
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.exp_name = exp_name
    Config.grid_env = grid_env
    Config.max_grad_norm = max_grad_norm
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.atari_wrapper = atari_wrapper
    Config.log_gradients = log_gradients
    Config.num_eval_episodes = num_eval_episodes
    Config.anneal_lr = anneal_lr
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.device = device

    if env is not None and Config.env_id != env_id:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for default environments or env for custom environments."
        )

    if Config.capture_video and not Config.use_wandb:
        raise ValueError(
            "Cannot capture video without WandB enabled. Set use_wandb=True to upload videos."
        )

    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            config=vars(Config()),
            name=run_name,
            monitor_gym=True,
        )

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)

    device = torch.device(Config.device)

    if Config.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        torch.backends.cudnn.benchmark = False

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
        obs_space_shape = envs.single_observation_space.shape[0]

    action_space_n = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else envs.single_action_space.shape[0]
    )

    print(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")

    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(actor_class, obs_space_shape, action_space_n)
        actor_network = actor_class.to(device)
    else:
        # Use actor class
        actor_network = actor_class(obs_space_shape, action_space_n).to(device)

    # Create critic network
    if isinstance(critic_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(critic_class, obs_space_shape)
        critic_network = critic_class.to(device)
    else:
        # Use critic class
        critic_network = critic_class(obs_space_shape).to(device)

    print("Actor Network Architecture:")
    print(actor_network)
    print("\nCritic Network Architecture:")
    print(critic_network)

    num_updates = Config.total_timesteps // Config.n_envs

    # Separate optimizers for actor and critic (A2C style)
    actor_optim = optim.Adam(actor_network.parameters(), lr=Config.lr)
    critic_optim = optim.Adam(critic_network.parameters(), lr=Config.lr)

    global_step = 0

    next_obs, _ = envs.reset(seed=Config.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    start_time = time.time()

    for step in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        # Annealing the rate if instructed to do so.
        global_step = step * Config.n_envs
        if Config.anneal_lr:
            frac = 1.0 - (step - 1.0) / num_updates
            lrnow = frac * Config.lr
            actor_optim.param_groups[0]["lr"] = lrnow
            critic_optim.param_groups[0]["lr"] = lrnow

        # Rollout Phase - Collect full episode

        log_probs = []
        rewards = []
        values = []
        entropies = []

        while True:
            # Get action and value
            result = actor_network.get_action(next_obs)
            if len(result) == 2:
                action, logprob = result
                logprob = logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
                dist = None
            elif len(result) == 3:
                action, logprob, dist = result
                logprob = logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
            else:
                raise ValueError(
                    f"Error unpacking result from get_action. Expected 2 or 3 values, got {len(result)}"
                )

            # Track entropy if enabled
            if Config.ENTROPY_COEFF > 0.0 and dist is not None:
                entropies.append(dist.entropy())

            value = critic_network(next_obs)

            # Log distribution statistics
            if Config.use_wandb and step % 100 == 0:
                if dist is not None and hasattr(dist, "probs"):
                    # Discrete
                    probs = dist.probs
                    wandb.log(
                        {
                            "policy/dist_mean": probs.mean().item(),
                            "policy/dist_std": probs.std().item(),
                            "policy/entropy": dist.entropy().mean().item(),
                            "global_step": global_step,
                        }
                    )

            values.append(value.flatten())

            log_probs.append(logprob)

            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            rewards.append(
                torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            )
            next_obs = torch.Tensor(new_obs).to(device)

            if np.all(done):
                break

        # Convert lists to tensors

        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.stack(rewards)

        values_tensor = torch.stack(values)

        # Calculate returns (Monte Carlo)
        num_steps = len(rewards)
        returns = torch.zeros_like(rewards_tensor, dtype=torch.float32).to(device)
        rt = 0.0
        for t in reversed(range(num_steps)):
            rt = rewards_tensor[t] + rt * Config.gamma
            returns[t] = rt

        # Calculate advantages (no normalization in pure A2C)
        advantages = (returns - values_tensor).detach()

        # Flatten for batch processing
        b_logprobs = log_probs_tensor.flatten()
        b_values = values_tensor.flatten()
        b_returns = returns.flatten()
        b_advantages = advantages.flatten()

        # A2C Update - Single pass through data
        # Actor Loss: policy gradient with advantage
        policy_loss = -(b_logprobs * b_advantages).mean()

        # Add entropy bonus if enabled
        if Config.ENTROPY_COEFF > 0.0 and entropies:
            entropy_loss = torch.stack(entropies).mean() * Config.ENTROPY_COEFF
            policy_loss = policy_loss - entropy_loss
            if Config.use_wandb:
                wandb.log(
                    {
                        "losses/entropy_loss": entropy_loss.item(),
                        "global_step": global_step,
                    }
                )

        # Critic Loss: MSE between value predictions and returns
        critic_loss = F.mse_loss(b_values, b_returns)

        # Optimize Actor
        actor_optim.zero_grad()
        policy_loss.backward()

        # Log gradient norms for actor
        if Config.use_wandb and Config.log_gradients:
            total_norm = 0
            for name, param in actor_network.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    wandb.log(
                        {
                            f"gradients/actor_{name}": param_norm.item(),
                            "global_step": global_step,
                        }
                    )
                    total_norm += param_norm.item() ** 2
            wandb.log(
                {
                    "gradients/actor_total_norm": total_norm**0.5,
                    "global_step": global_step,
                }
            )

        if Config.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                actor_network.parameters(), max_norm=Config.max_grad_norm
            )
        actor_optim.step()

        # Optimize Critic
        critic_optim.zero_grad()
        critic_loss.backward()

        # Log gradient norms for critic
        if Config.use_wandb and Config.log_gradients:
            total_norm = 0
            for name, param in critic_network.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    wandb.log(
                        {
                            f"gradients/critic_{name}": param_norm.item(),
                            "global_step": global_step,
                        }
                    )
                    total_norm += param_norm.item() ** 2
            wandb.log(
                {
                    "gradients/critic_total_norm": total_norm**0.5,
                    "global_step": global_step,
                }
            )

        if Config.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                critic_network.parameters(), max_norm=Config.max_grad_norm
            )
        critic_optim.step()

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
                                    "global_step": global_step,
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
                                "global_step": global_step,
                            }
                        )

        # Log losses and metrics
        if Config.use_wandb and step % 10 == 0:
            wandb.log(
                {
                    "losses/policy_loss": policy_loss.item(),
                    "losses/critic_loss": critic_loss.item(),
                    "charts/learning_rate": actor_optim.param_groups[0]["lr"],
                    "charts/rewards_mean": rewards_tensor.mean().item(),
                    "advantages/advantages_mean": advantages.mean().item(),
                    "advantages/advantages_std": advantages.std().item(),
                    "charts/returns_mean": b_returns.mean().item(),
                    "global_step": global_step,
                }
            )
            print(
                f"Step {step}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, SPS: {int(global_step / (time.time() - start_time))}"
            )

            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "charts/global_step": global_step,
                    }
                )
        if step % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                actor_network,
                device,
                Config.env_id,
                env=env,
                seed=Config.seed,
                num_eval_eps=Config.num_eval_episodes,
                record=Config.capture_video,
                render_mode="rgb_array" if Config.capture_video else None,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                use_wandb=Config.use_wandb,
            )
            avg_return = np.mean(episodic_returns)
            if Config.use_wandb:
                wandb.log(
                    {"charts/val_avg_return": avg_return, "global_step": global_step}
                )
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if step % Config.save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/a2c_model_step_{global_step}.pth"
            torch.save(
                {
                    "actor": actor_network.state_dict(),
                    "critic": critic_network.state_dict(),
                },
                model_path,
            )
            print(f"Model saved at step {step} step {global_step} to {model_path}")

    # --- Final Evaluation and Video Saving ---
    if Config.use_wandb:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(
            actor_network,
            device,
            Config.env_id,
            env=env,
            seed=Config.seed,
            num_eval_eps=Config.num_eval_episodes,
            record=Config.capture_video,
            grid_env=Config.grid_env,
            atari_wrapper=Config.atari_wrapper,
            env_wrapper=env_wrapper,
            use_wandb=Config.use_wandb,
            render_mode="rgb_array" if Config.capture_video else None,
        )

        # Save final video to file if frames were captured
        if eval_frames:
            train_video_path = "videos/final.mp4"
            imageio.mimsave(train_video_path, eval_frames, fps=30)
            print(f"Final training video saved to {train_video_path}")

    envs.close()
    if Config.use_wandb:
        wandb.finish()

    return actor_network


def train_a2c_cnn(
    env_id: str = Config.env_id,
    env: Optional[gym.Env] = None,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    gamma: float = Config.gamma,
    n_envs: int = Config.n_envs,
    ENTROPY_COEFF: float = Config.ENTROPY_COEFF,
    VALUE_COEFF: float = Config.VALUE_COEFF,
    capture_video: bool = Config.capture_video,
    use_wandb: bool = Config.use_wandb,
    wandb_project: str = Config.wandb_project,
    exp_name: str = Config.exp_name,
    grid_env: bool = Config.grid_env,
    max_grad_norm: float = Config.max_grad_norm,
    eval_every: int = Config.eval_every,
    save_every: int = Config.save_every,
    atari_wrapper: bool = Config.atari_wrapper,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    actor_class: Any = ActorNet,
    critic_class: Any = CriticNet,
    log_gradients: bool = Config.log_gradients,
    num_eval_episodes: int = Config.num_eval_episodes,
    anneal_lr: bool = Config.anneal_lr,
    normalize_obs: bool = Config.normalize_obs,
    normalize_reward: bool = Config.normalize_reward,
    device: str = Config.device,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id if env is None else env.spec.id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.lr = lr
    Config.gamma = gamma
    Config.n_envs = n_envs
    Config.ENTROPY_COEFF = ENTROPY_COEFF
    Config.VALUE_COEFF = VALUE_COEFF
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.exp_name = exp_name
    Config.grid_env = grid_env
    Config.max_grad_norm = max_grad_norm
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.atari_wrapper = atari_wrapper
    Config.log_gradients = log_gradients
    Config.num_eval_episodes = num_eval_episodes
    Config.anneal_lr = anneal_lr
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.device = device

    if env is not None and Config.env_id != env_id:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for default environments or env for custom environments."
        )

    if Config.capture_video and not Config.use_wandb:
        raise ValueError(
            "Cannot capture video without WandB enabled. Set use_wandb=True to upload videos."
        )

    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            config=vars(Config()),
            name=run_name,
            monitor_gym=True,
        )

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    if Config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
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
        else envs.single_action_space.shape
    )

    print(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")

    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(
            actor_class,
            obs_space_shape[0] if len(obs_space_shape) == 1 else obs_space_shape,
            action_space_n,
        )
        actor_network = actor_class.to(device)
    else:
        # Use actor class
        actor_network = actor_class(obs_space_shape, action_space_n).to(device)

    # Create critic network
    if isinstance(critic_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(
            critic_class,
            obs_space_shape[0] if len(obs_space_shape) == 1 else obs_space_shape,
        )
        critic_network = critic_class.to(device)
    else:
        # Use critic class
        critic_network = critic_class(obs_space_shape).to(device)

    print("Actor Network Architecture:")
    print(actor_network)
    print("\nCritic Network Architecture:")
    print(critic_network)

    # Compute derived values from passed parameters
    batch_size = Config.n_envs * Config.max_steps
    num_updates = Config.total_timesteps // batch_size

    # Separate optimizers for actor and critic (A2C style)
    actor_optim = optim.Adam(actor_network.parameters(), lr=Config.lr)
    critic_optim = optim.Adam(critic_network.parameters(), lr=Config.lr)

    global_step = 0

    next_obs, _ = envs.reset(seed=Config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(Config.n_envs).to(device)

    start_time = time.time()

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        # Annealing the rate if instructed to do so.
        if Config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * Config.lr
            actor_optim.param_groups[0]["lr"] = lrnow
            critic_optim.param_groups[0]["lr"] = lrnow

        # Rollout Phase - Collect full episode

        log_probs = []
        rewards = []
        dones = []
        values = []
        entropies = []

        for step in range(0, Config.max_steps):
            global_step = (update - 1) * batch_size + step * Config.n_envs

            dones.append(next_done)

            # Get action and value
            result = actor_network.get_action(next_obs)
            if len(result) == 2:
                action, logprob = result
                logprob = logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
                dist = None
            elif len(result) == 3:
                action, logprob, dist = result
                logprob = logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
            else:
                raise ValueError(
                    f"Error unpacking result from get_action. Expected 2 or 3 values, got {len(result)}"
                )

            # Track entropy if enabled
            if Config.ENTROPY_COEFF > 0.0 and dist is not None:
                entropies.append(dist.entropy())

            value = critic_network(next_obs)

            # Log distribution statistics
            if Config.use_wandb and step % 100 == 0:
                if dist is not None and hasattr(dist, "probs"):
                    # Discrete
                    probs = dist.probs
                    wandb.log(
                        {
                            "policy/dist_mean": probs.mean().item(),
                            "policy/dist_std": probs.std().item(),
                            "policy/entropy": dist.entropy().mean().item(),
                            "global_step": global_step,
                        }
                    )

            values.append(value.flatten())

            log_probs.append(logprob)

            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            rewards.append(
                torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            )
            next_obs = torch.Tensor(new_obs).to(device)
            next_done = torch.Tensor(done).to(device)

        # Convert lists to tensors

        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)

        # Calculate returns (Monte Carlo)
        num_steps = len(rewards)
        returns = torch.zeros_like(rewards_tensor, dtype=torch.float32).to(device)
        rt = 0.0
        for t in reversed(range(num_steps)):
            rt = rewards_tensor[t] + rt * Config.gamma
            returns[t] = rt

        # Flatten tensors for batch processing
        b_logprobs = log_probs_tensor.reshape(-1)
        b_values = values_tensor.reshape(-1)
        b_returns = returns.reshape(-1)

        # Calculate advantages (no normalization in pure A2C)
        advantages = (b_returns - b_values).detach()

        # A2C Update - Single pass through data
        # Actor Loss: policy gradient with advantage
        policy_loss = -(b_logprobs * advantages).mean()

        # Add entropy bonus if enabled
        if Config.ENTROPY_COEFF > 0.0 and entropies:
            entropy_loss = torch.stack(entropies).mean() * Config.ENTROPY_COEFF
            policy_loss = policy_loss - entropy_loss
            if Config.use_wandb:
                wandb.log(
                    {
                        "losses/entropy_loss": entropy_loss.item(),
                        "global_step": global_step,
                    }
                )

        # Critic Loss: MSE between value predictions and returns
        critic_loss = F.mse_loss(b_values, b_returns)

        # Optimize Actor
        actor_optim.zero_grad()
        policy_loss.backward()

        # Log gradient norms for actor
        if Config.use_wandb and Config.log_gradients:
            total_norm = 0
            for name, param in actor_network.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    wandb.log(
                        {
                            f"gradients/actor_{name}": param_norm.item(),
                            "global_step": global_step,
                        }
                    )
                    total_norm += param_norm.item() ** 2
            wandb.log(
                {
                    "gradients/actor_total_norm": total_norm**0.5,
                    "global_step": global_step,
                }
            )

        if Config.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                actor_network.parameters(), max_norm=Config.max_grad_norm
            )
        actor_optim.step()

        # Optimize Critic
        critic_optim.zero_grad()
        critic_loss.backward()

        # Log gradient norms for critic
        if Config.use_wandb and Config.log_gradients:
            total_norm = 0
            for name, param in critic_network.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    wandb.log(
                        {
                            f"gradients/critic_{name}": param_norm.item(),
                            "global_step": global_step,
                        }
                    )
                    total_norm += param_norm.item() ** 2
            wandb.log(
                {
                    "gradients/critic_total_norm": total_norm**0.5,
                    "global_step": global_step,
                }
            )

        if Config.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                critic_network.parameters(), max_norm=Config.max_grad_norm
            )
        critic_optim.step()

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
                                    "global_step": global_step,
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
                                "global_step": global_step,
                            }
                        )

        # Log losses and metrics
        if Config.use_wandb and update % 10 == 0:
            wandb.log(
                {
                    "losses/policy_loss": policy_loss.item(),
                    "losses/critic_loss": critic_loss.item(),
                    "charts/learning_rate": actor_optim.param_groups[0]["lr"],
                    "charts/rewards_mean": rewards_tensor.mean().item(),
                    "advantages/advantages_mean": advantages.mean().item(),
                    "advantages/advantages_std": advantages.std().item(),
                    "charts/returns_mean": b_returns.mean().item(),
                    "calculated_return": b_returns.mean().item(),
                    "global_step": global_step,
                }
            )
            print(
                f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, SPS: {int(global_step / (time.time() - start_time))}"
            )

            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "charts/global_step": global_step,
                    }
                )
        if update % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                actor_network,
                device,
                Config.env_id,
                env=env,
                seed=Config.seed,
                num_eval_eps=Config.num_eval_episodes,
                record=Config.capture_video,
                render_mode="rgb_array" if Config.capture_video else None,
                grid_env=Config.grid_env,
                atari_wrapper=Config.atari_wrapper,
                env_wrapper=env_wrapper,
                use_wandb=Config.use_wandb,
            )
            avg_return = np.mean(episodic_returns)
            if Config.use_wandb:
                wandb.log(
                    {"charts/val_avg_return": avg_return, "global_step": global_step}
                )
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if update % Config.save_every == 0 and update > 0:
            model_path = f"runs/{run_name}/models/a2c_model_step_{global_step}.pth"
            torch.save(
                {
                    "actor": actor_network.state_dict(),
                    "critic": critic_network.state_dict(),
                },
                model_path,
            )
            print(f"Model saved at update {update} step {global_step} to {model_path}")

    # --- Final Evaluation and Video Saving ---
    if Config.use_wandb:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(
            actor_network,
            device,
            Config.env_id,
            env=env,
            seed=Config.seed,
            num_eval_eps=Config.num_eval_episodes,
            record=Config.capture_video,
            grid_env=Config.grid_env,
            atari_wrapper=Config.atari_wrapper,
            env_wrapper=env_wrapper,
            use_wandb=Config.use_wandb,
            render_mode="rgb_array" if Config.capture_video else None,
        )

        # Save final video to file if frames were captured
        if eval_frames:
            train_video_path = "videos/final.mp4"
            imageio.mimsave(train_video_path, eval_frames, fps=30)
            print(f"Final training video saved to {train_video_path}")

    envs.close()
    if Config.use_wandb:
        wandb.finish()

    return actor_network


# --- Main Execution ---
if __name__ == "__main__":
    train_a2c()
