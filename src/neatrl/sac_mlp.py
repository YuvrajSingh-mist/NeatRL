"""Soft Actor-Critic (SAC) with MLP networks for continuous-action Gymnasium environments."""

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
from .utils import configure_logging, get_logger, get_space_dims, setup_device
from .utils.nn_utils import (
    validate_critic_network_dimensions,
    validate_policy_network_dimensions,
)
from .utils.wrappers import OneHotWrapper

logger = get_logger(__name__)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ===== CONFIGURATION =====
@dataclass
class Config:
    """Configuration class for SAC training."""

    # Experiment settings
    exp_name: str = "SAC-Experiment"
    seed: int = 42
    env_id: Optional[str] = "HalfCheetah-v5"

    # Training parameters
    total_timesteps: int = 1000000
    lr: float = 3e-4
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter for target networks
    target_network_frequency: int = 1  # How often to update target networks
    batch_size: int = 256

    # SAC-specific parameters
    alpha: float = 0.2  # Entropy regularization coefficient
    autotune_alpha: bool = True  # Whether to automatically tune alpha
    target_entropy_scale: float = (
        -1.0
    )  # Target entropy = target_entropy_scale * action_dim

    learning_starts: int = 5000
    policy_frequency: int = 1  # How often to update the policy (1 = every step)

    # Logging & Saving
    capture_video: bool = False  # Whether to capture evaluation videos
    use_wandb: bool = True
    use_dashboard: bool = False  # Whether to use Weights & Biases for logging
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
    max_grad_norm: float = 0.5  # Maximum gradient norm for gradient clipping
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
        """Forward pass — returns network output(s).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Network output(s).
        """
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action using the reparameterization trick and return it with log-probability.

        Args:
            x (torch.Tensor): Observation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sampled action (tanh-squashed) and
                corresponding log-probability with correction for tanh squashing.
        """
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
    """Twin Q-network (soft critic) taking concatenated state-action pairs."""

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
        """Forward pass — returns the Q-value for the given state-action pair.

        Args:
            state (torch.Tensor): State observation tensor.
            act (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Scalar Q-value for each state-action pair in the batch.
        """
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=-1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


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
                # For SAC, use stochastic action during evaluation
                action, _ = model.get_action(obs_tensor)  # type: ignore[operator]
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


def train_sac(
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
    """Train a SAC agent on a continuous-action environment.

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
        alpha (float): Initial entropy regularisation coefficient α.
        autotune_alpha (bool): Automatically tune α via a dual-variable update when True.
        target_entropy_scale (float): Target entropy = target_entropy_scale × action_dim.
        learning_starts (int): Random-action steps before gradient updates begin.
        policy_frequency (int): Update the actor every this many critic updates.
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
        normalize_obs (bool): Normalise observations with a running mean/std when True.
        normalize_reward (bool): Normalise rewards with a running std when True.
        actor_class (type | nn.Module): Custom ``nn.Module`` class or instance replacing the default actor.
        q_network_class (type | nn.Module): Custom class or instance for the twin Q-networks (SAC).

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
    configure_logging(log_dir=os.path.join("runs", Config.exp_name))
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    device = setup_device(Config.device, Config.seed)  # type: ignore[assignment]

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
    obs_space_shape, action_space_n = get_space_dims(envs)

    logger.info(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")

    # Create actor network
    if isinstance(actor_class, nn.Module):
        # Use custom actor instance
        validate_policy_network_dimensions(actor_class, obs_space_shape, action_space_n)
        actor_net = actor_class.to(device)
    else:
        # Use actor class
        actor_net = actor_class(obs_space_shape, action_space_n)
        actor_net = actor_net.to(device)

    # Create twin critic networks (SAC uses two Q-networks)
    if isinstance(q_network_class, nn.Module):
        # Use custom critic instance
        validate_critic_network_dimensions(
            q_network_class, obs_space_shape, action_space_n
        )
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
    logger.debug("%s\n%s", "Actor Network Architecture:", actor_net)
    logger.debug("%s\n%s", "\nQ1 Network Architecture:", q1_network)
    logger.debug("%s\n%s", "\nQ2 Network Architecture:", q2_network)

    # Optimizers
    actor_optim = optim.Adam(actor_net.parameters(), lr=Config.lr)
    q1_optim = optim.Adam(q1_network.parameters(), lr=Config.lr)
    q2_optim = optim.Adam(q2_network.parameters(), lr=Config.lr)

    # Automatic entropy tuning
    if Config.autotune_alpha:
        target_entropy = Config.target_entropy_scale * float(action_space_n)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = optim.Adam([log_alpha], lr=Config.lr)
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

    obs, _ = envs.reset()  # type: ignore[var-annotated]
    start_time = time.time()

    dashboard = (
        Dashboard("SAC", Config.env_id or "custom", Config.total_timesteps)
        if Config.use_dashboard
        else None
    )

    for step in tqdm(range(Config.total_timesteps)):
        # Sample action from stochastic policy
        with torch.no_grad():
            action, _ = actor_net.get_action(  # type: ignore[operator]
                torch.tensor(obs, device=device, dtype=torch.float32)
            )

        action_np = action.cpu().numpy()
        new_obs, reward, terminated, truncated, info = envs.step(action_np)  # type: ignore[var-annotated]
        done = np.logical_or(terminated, truncated)
        replay_buffer.add(
            obs, new_obs, action_np, np.array(reward), np.array(done), [info]
        )

        # Training step
        if step > Config.learning_starts:
            data = replay_buffer.sample(Config.batch_size)

            # Update Q-networks
            with torch.no_grad():
                next_actions, next_log_probs = actor_net.get_action(  # type: ignore[operator]
                    data.next_observations.to(torch.float32)
                )

                next_log_probs = (
                    next_log_probs.sum(dim=-1, keepdim=True)
                    if len(next_log_probs.shape) > 1
                    else next_log_probs
                )

                target_q1 = target_q1_network(
                    data.next_observations.to(torch.float32), next_actions
                )
                target_q2 = target_q2_network(
                    data.next_observations.to(torch.float32), next_actions
                )
                min_target_q = torch.min(target_q1, target_q2)
                td_target = data.rewards + Config.gamma * (
                    min_target_q - alpha * next_log_probs
                ) * (1 - data.dones)

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
                new_actions, log_probs = actor_net.get_action(  # type: ignore[operator]
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
                    alpha_loss = (
                        -log_alpha.exp() * (log_probs + target_entropy).detach()
                    ).mean()
                    alpha_loss_value = alpha_loss.item()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()

            if step % Config.target_network_frequency == 0:
                # Soft update target networks
                for param, target_param in zip(
                    q1_network.parameters(), target_q1_network.parameters()
                ):
                    target_param.data.copy_(
                        Config.tau * param.data + (1 - Config.tau) * target_param.data
                    )
                for param, target_param in zip(
                    q2_network.parameters(), target_q2_network.parameters()
                ):
                    target_param.data.copy_(
                        Config.tau * param.data + (1 - Config.tau) * target_param.data
                    )

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
                logger.info(
                    "Step %d Q1 Loss: %.4f Q2 Loss: %.4f Alpha: %.4f Actor Loss: %.4f SPS: %d",
                    step,
                    q1_loss.item(),
                    q2_loss.item(),
                    alpha,
                    actor_loss.item(),
                    sps,
                )
                if dashboard:
                    dashboard.update(
                        agent_steps=step,
                        epoch=step,
                        losses={
                            "q1_loss": q1_loss.item(),
                            "q2_loss": q2_loss.item(),
                            "actor_loss": actor_loss.item(),
                            "alpha": float(alpha),
                            "alpha_loss": alpha_loss_value or 0.0,
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
                    "q1_network": q1_network.state_dict(),
                    "q2_network": q2_network.state_dict(),
                    "target_q1_network": target_q1_network.state_dict(),
                    "target_q2_network": target_q2_network.state_dict(),
                },
                model_path,
            )
            logger.info(f"Model saved at step {step} to {model_path}")

        if done.all():
            obs, _ = envs.reset()
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

    envs.close()
    if dashboard:
        dashboard.close()
    return actor_net


# --- Main Execution ---
if __name__ == "__main__":
    train_sac()
