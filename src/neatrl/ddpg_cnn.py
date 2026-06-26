"""Deep Deterministic Policy Gradient (DDPG) with CNN networks for visual environments."""

import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from .cli.dashboard import Dashboard
from .ddpg_mlp import ActorNet, QNet
from .utils import configure_logging, get_logger, setup_device
from .utils.nn_utils import (
    validate_critic_network_dimensions,
    validate_policy_network_dimensions,
)
from .utils.wrappers import OneHotWrapper

logger = get_logger(__name__)


# ===== CONFIGURATION =====
@dataclass
class Config:
    """Configuration class for DDPG training."""

    # Experiment settings
    exp_name: str = "DDPG-Experiment"
    seed: int = 42
    env_id: Optional[str] = "HalfCheetah-v5"

    low: float = -1.0
    noise_clip: float = 0.5
    high: float = 1.0  # Action space limits for BipedalWalker
    # Training parameters
    total_timesteps: int = 1000000
    lr: float = 3e-4
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter for target networks
    target_network_frequency: int = 50  # How often to update target networks
    batch_size: int = 256

    exploration_fraction: float = 0.1
    learning_starts: int = 25000
    train_frequency: int = 2

    # Logging & Saving
    capture_video: bool = False  # Whether to capture evaluation videos
    use_wandb: bool = True
    use_dashboard: bool = False  # Whether to use Weights & Biases for logging
    wandb_project: str = "cleanRL"  # W&B project name
    wandb_entity: str = ""  # Your WandB username/team
    eval_every: int = 500  # Frequency of evaluation during training (in steps)
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
    max_grad_norm: float = 0.5  # Maximum gradient norm for gradient clipping
    log_gradients: bool = True  # Whether to log gradient norms to W&B
    device: str = "cpu"  # Device for training: "auto", "cpu", "cuda", or "cuda:0" etc.


class ActorNetCNN(nn.Module):
    """CNN-based deterministic actor for pixel observations."""

    def __init__(self, obs_shape, action_space):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, action_space)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        """Forward pass — returns network output(s).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Network output(s).
        """
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.tanh(self.out(x))  # Output between -1 and 1
        x = x * 1.0  # Scale to action limits
        return x


class QNetCNN(nn.Module):
    """CNN-based critic for pixel observations."""

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
        """Forward pass — returns the Q-value for the given state-action pair.

        Args:
            state (torch.Tensor): State observation tensor.
            act (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Scalar Q-value for each state-action pair in the batch.
        """
        x = torch.nn.functional.relu(self.conv1(state))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        state_features = torch.nn.functional.relu(self.state_fc(x))

        # Process action
        action_features = torch.nn.functional.relu(self.action_fc(action))

        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        x = torch.nn.functional.relu(self.combined_fc(combined))
        return self.out(x)


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
    """Return a thunk that constructs and seeds a Gymnasium environment.

    Args:
        env_id (str): Gymnasium environment ID. Ignored when ``env`` is provided.
        seed (int): Base random seed; actual seed is ``seed + idx``.
        idx (int): Worker index added to the seed. Defaults to 0.
        render_mode (str | None): Render mode passed to ``gym.make``. Defaults to None.
        grid_env (bool): Apply one-hot observation encoding for grid environments.
        atari_wrapper (bool): Apply Atari preprocessing wrappers when True.
        env_wrapper (Callable | None): Optional extra wrapper applied after creation.
        env (gym.Env | None): Pre-built environment; skips ``gym.make`` when provided.

    Returns:
        Callable[[], gym.Env]: A zero-argument thunk that creates the environment.
    """
    if env_id and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium environments or env for custom environments."
        )

    def thunk():
        """Environment factory thunk (called by SyncVectorEnv)."""
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
                env_to_use,
                obs_shape=env_to_use.observation_space.n,  # type: ignore[attr-defined]
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


def evaluate(
    model: nn.Module,
    device: Union[str, torch.device],
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
    """Run evaluation episodes and return episodic returns (and optional video frames).

    Args:
        model (nn.Module): The policy network to evaluate.
        device (torch.device | str): Device on which to run the network.
        env_id (str): Gymnasium environment ID for the evaluation environment.
        env (gym.Env | None): Pre-created environment; skips ``gym.make`` when provided.
        seed (int): Random seed for the evaluation environment. Defaults to 42.
        num_eval_eps (int): Number of episodes to run.
        record (bool): Record frames when True.
        render_mode (str | None): Render mode passed to the environment.
        grid_env (bool): Apply one-hot encoding for grid environments.
        atari_wrapper (bool): Apply Atari preprocessing wrappers when True.
        env_wrapper (Callable | None): Optional extra wrapper applied after creation.
        use_wandb (bool): Upload recorded video to Weights & Biases when True.

    Returns:
        tuple[list[float], list]: A list of total rewards per episode and a list of
            recorded RGB frames (empty if capture_video is False).
    """
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
                try:
                    frame: Any = eval_env.render()
                    frames.append(frame)
                except Exception as e:
                    logger.error(
                        "Video capture failed for %s (%s). "
                        "Check the renderer for this environment is installed "
                        "(e.g. pip install pygame-ce for classic-control envs).",
                        type(eval_env).__name__, type(e).__name__,
                    )
                    raise
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    np.array(obs), device=device, dtype=torch.float32
                ).unsqueeze(0)
                action = model(obs_tensor)
                action = torch.clip(
                    action, Config.low, Config.high
                )  # Use args low and high
                action = action.cpu().numpy()
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    # For continuous action spaces, ensure it's 1D
                    action = action.flatten()

            obs, rewards_curr, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            rewards += float(rewards_curr)

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


def train_ddpg_cnn(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = None,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    buffer_size: int = Config.buffer_size,
    gamma: float = Config.gamma,
    tau: float = Config.tau,
    target_network_frequency: int = Config.target_network_frequency,
    batch_size: int = Config.batch_size,
    exploration_fraction: float = Config.exploration_fraction,
    learning_starts: int = Config.learning_starts,
    train_frequency: int = Config.train_frequency,
    exploration_noise: float = Config.exploration_fraction,
    noise_clip: float = Config.noise_clip,
    low: float = Config.low,
    high: float = Config.high,
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
    actor_class: Any = ActorNet,
    q_network_class: Any = QNet,
) -> nn.Module:
    # Update Config with passed arguments
    """Train a CNN-based DDPG agent on pixel observations.

    Args:
        env_id (str | None): Gymnasium environment ID. Mutually exclusive with ``env``.
        env (gym.Env | None): Pre-created ``gym.Env`` instance. Mutually exclusive with ``env_id``.
        total_timesteps (int): Total environment interaction steps to train for.
        seed (int): Global random seed for reproducibility.
        lr (float): Optimiser learning rate.
        buffer_size (int): Replay buffer capacity (number of transitions).
        gamma (float): Discount factor γ (0 < γ ≤ 1).
        tau (float): Soft target-network update coefficient (0 < τ ≤ 1).
        target_network_frequency (int): Sync target networks every this many steps.
        batch_size (int): Mini-batch size sampled from the replay buffer.
        exploration_fraction (float): Fraction of total_timesteps for exploration noise scaling.
        learning_starts (int): Random-action steps before gradient updates begin.
        train_frequency (int): Update the network every this many steps.
        capture_video (bool): Record evaluation episodes to video files when True.
        use_wandb (bool): Log training metrics to Weights & Biases when True.
        wandb_project (str): W&B project name (used when use_wandb=True).
        wandb_entity (str): W&B entity/username (used when use_wandb=True).
        exp_name (str): Run name used for log directories and W&B.
        eval_every (int): Evaluate every this many environment steps.
        save_every (int): Save a model checkpoint every this many steps.
        num_eval_episodes (int): Number of episodes per evaluation.
        n_envs (int): Number of parallel vectorised environments.
        max_grad_norm (float): Maximum gradient-norm for clipping (0 disables clipping).
        log_gradients (bool): Log per-layer gradient norms to W&B when True.
        device (str): PyTorch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        grid_env (bool): Use one-hot observation encoding for grid environments.
        atari_wrapper (bool): Apply Atari preprocessing (grayscale, frame-stack) when True.
        env_wrapper (Callable | None): Optional callable wrapping the environment after creation.
        normalize_reward (bool): Normalise rewards with a running std when True.
        actor_class (type | nn.Module): Custom ``nn.Module`` class or instance replacing the default actor.
        q_network_class (type | nn.Module): Custom class or instance for the Q-network.
        exploration_noise (float): Std-dev of Gaussian exploration noise.
        noise_clip (float): Clipping range for target-policy smoothing noise.
        low (float): Lower bound of the action space.
        high (float): Upper bound of the action space.

    Returns:
        nn.Module: The trained network (actor / policy / Q-network) ready for inference.
    """
    Config.env_id = env_id or env.spec.id  # type: ignore[union-attr]
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.lr = lr
    Config.buffer_size = buffer_size
    Config.gamma = gamma
    Config.tau = tau
    Config.target_network_frequency = target_network_frequency
    Config.batch_size = batch_size
    Config.exploration_fraction = exploration_fraction
    Config.learning_starts = learning_starts
    Config.train_frequency = train_frequency
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
    Config.exploration_fraction = exploration_noise
    Config.noise_clip = noise_clip
    Config.low = low
    Config.high = high
    Config.normalize_obs = False  # Not supported for CNN-based DDPG
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
    configure_logging(log_dir=os.path.join("runs", Config.exp_name))
    device = setup_device(Config.device, Config.seed)  # type: ignore[assignment]

    # Create environment
    if env is not None:
        env_thunk = make_env(
            "",
            Config.seed,
            idx=0,
            render_mode="rgb_array",
            env_wrapper=Config.env_wrapper,
            env=env,
        )
    else:
        env_thunk = make_env(
            Config.env_id,
            Config.seed,
            idx=0,
            render_mode="rgb_array",
            env_wrapper=Config.env_wrapper,
        )

    train_env_cnn = env_thunk()

    # Determine if we're dealing with discrete observation spaces
    is_discrete_obs = isinstance(train_env_cnn.observation_space, gym.spaces.Discrete)
    obs_space = train_env_cnn.observation_space

    # Compute observation dimensions
    if is_discrete_obs:
        obs_shape: Union[int, tuple[int, ...]] = int(obs_space.n)  # type: ignore[attr-defined]
    else:
        obs_shape = tuple(obs_space.shape)  # type: ignore[arg-type]

    # Compute action dimensions
    action_shape = int(
        train_env_cnn.action_space.n  # type: ignore[attr-defined]
        if isinstance(train_env_cnn.action_space, gym.spaces.Discrete)
        else train_env_cnn.action_space.shape[0]  # type: ignore[index]
    )
    env = train_env_cnn

    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(actor_class, obs_shape, action_shape)
        actor_net = actor_class.to(device)
    else:
        # Use actor class
        actor_net = actor_class(obs_shape, action_shape).to(device)

    # Create critic network
    if isinstance(q_network_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(q_network_class, obs_shape, action_shape)
        q_network = q_network_class.to(device)
    else:
        # Use critic class
        q_network = q_network_class(obs_shape, action_shape).to(device)

    target_actor_net = copy.deepcopy(actor_net)
    target_q_network = copy.deepcopy(q_network)

    target_q_network.load_state_dict(q_network.state_dict())
    target_actor_net.load_state_dict(actor_net.state_dict())

    # Print network architecture
    logger.debug("%s\n%s", "Actor Network Architecture:", actor_net)
    logger.debug("%s\n%s", "\nCritic Network Architecture:", q_network)

    # Optimizers
    actor_optim = optim.Adam(actor_net.parameters(), lr=Config.lr)
    q_optim = optim.Adam(q_network.parameters(), lr=Config.lr)

    # Set networks to training mode
    q_network.train()
    actor_net.train()

    # Replay buffer

    replay_buffer = ReplayBuffer(
        Config.buffer_size,
        env.observation_space,
        env.action_space,
        device=device,
        handle_timeout_termination=False,
        n_envs=Config.n_envs,
    )

    obs, _ = env.reset()
    start_time = time.time()

    dashboard = (
        Dashboard("DDPG-CNN", Config.env_id or "custom", Config.total_timesteps)
        if Config.use_dashboard
        else None
    )

    for step in tqdm(range(Config.total_timesteps)):
        # Get action from actor network with exploration noise
        with torch.no_grad():
            action = actor_net(
                torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            )
            action = action + torch.clip(
                torch.randn_like(action) * Config.exploration_fraction,
                -Config.noise_clip,
                Config.noise_clip,
            )
            action = torch.clip(action, Config.low, Config.high)
        action_np = action.cpu().numpy().flatten()

        new_obs, reward, terminated, truncated, info = env.step(action_np)
        done = np.logical_or(terminated, truncated)
        replay_buffer.add(
            obs, new_obs, action_np, np.array(reward), np.array(done), [info]
        )

        # Training step
        if step > Config.learning_starts:
            data = replay_buffer.sample(Config.batch_size)
            with torch.no_grad():
                next_actions = target_actor_net(
                    data.next_observations.to(torch.float32)
                )
                target_max = target_q_network(
                    data.next_observations.to(torch.float32), next_actions
                )
                td_target = data.rewards + Config.gamma * target_max * (1 - data.dones)

            old_val = q_network(
                data.observations.to(torch.float32), data.actions.to(torch.float32)
            )

            q_optim.zero_grad()
            loss = nn.functional.mse_loss(old_val, td_target)
            loss.backward()

            # Log gradient norm per layer for critic
            if Config.use_wandb and Config.log_gradients:
                for name, param in q_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/critic_layer_{name}": grad_norm,
                                "global_step": step,
                            }
                        )

            # Apply gradient clipping for critic
            if Config.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(q_network.parameters()),
                    max_norm=Config.max_grad_norm,
                )

            q_optim.step()

            if step % Config.train_frequency == 0:
                actor_optim.zero_grad()
                actions = actor_net(data.observations.to(torch.float32))
                policy_loss = -q_network(
                    data.observations.to(torch.float32), actions.to(torch.float32)
                ).mean()
                policy_loss.backward()

                # Log gradient norm per layer for actor
                if Config.use_wandb and Config.log_gradients:
                    for name, param in actor_net.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/actor_layer_{name}": grad_norm,
                                    "global_step": step,
                                }
                            )

                # Apply gradient clipping for actor
                if Config.max_grad_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(actor_net.parameters()) + list(q_network.parameters()),
                        max_norm=Config.max_grad_norm,
                    )

                actor_optim.step()

            # Update target networks
            if step % Config.target_network_frequency == 0:
                for q_params, target_params in zip(
                    q_network.parameters(), target_q_network.parameters()
                ):
                    target_params.data.copy_(
                        Config.tau * q_params.data
                        + (1.0 - Config.tau) * target_params.data
                    )

                for actor_params, target_actor_params in zip(
                    actor_net.parameters(), target_actor_net.parameters()
                ):
                    target_actor_params.data.copy_(
                        Config.tau * actor_params.data
                        + (1.0 - Config.tau) * target_actor_params.data
                    )

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
            if step % 100 == 0:
                if Config.use_wandb:
                    wandb.log(
                        {
                            "losses/critic_loss": loss.item(),
                            "losses/actor_loss": policy_loss.item()
                            if step % Config.train_frequency == 0
                            else 0.0,
                            "charts/lr": actor_optim.param_groups[0]["lr"],
                            "rewards/rewards_mean": data.rewards.mean().item(),
                            "rewards/rewards_std": data.rewards.std().item(),
                            "global_step": step,
                        }
                    )
                logger.info(
                    "Step %d Critic Loss: %.4f Actor Loss: %.4f SPS: %d",
                    step,
                    loss.item(),
                    policy_loss.item() if step % Config.train_frequency == 0 else 0.0,
                    int(step / (time.time() - start_time)),
                )
            if dashboard:
                dashboard.update(
                    agent_steps=step,
                    epoch=step,
                    losses={
                        "critic_loss": loss.item(),
                        "actor_loss": policy_loss.item()
                        if policy_loss is not None
                        else 0.0,
                    },
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
                logger.info(
                    "Evaluation returns: %s  Average: %.2f",
                    [float(r) for r in episodic_returns],
                    avg_return,
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
                    "q_network": q_network.state_dict(),
                    "target_actor": target_actor_net.state_dict(),
                    "target_q_network": target_q_network.state_dict(),
                },
                model_path,
            )
            logger.info(f"Model saved at step {step} to {model_path}")

        if done.all():
            obs, _ = env.reset()
        else:
            obs = new_obs

    # Final evaluation and video saving
    if Config.use_wandb:
        logger.info("Capturing final evaluation video...")
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
            imageio.mimsave(train_video_path, eval_frames, fps=30, codec="libx264")  # type: ignore[arg-type]
            logger.info(f"Final training video saved to {train_video_path}")
            wandb.finish()

    env.close()
    if dashboard:
        dashboard.close()
    return actor_net


# --- Main Execution ---
if __name__ == "__main__":
    train_ddpg_cnn()
