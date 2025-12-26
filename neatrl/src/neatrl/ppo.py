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
    """Configuration class for PPO training."""

    # Experiment settings
    exp_name: str = "PPO-Experiment"  # Experiment name for logging and saving
    seed: int = 42  # Random seed for reproducibility
    env_id: Optional[str] = "CliffWalking-v0"  # Gymnasium environment ID
    total_timesteps: int = 5_000_000  # Total number of timesteps to train

    # PPO & Agent settings
    lr: float = 3e-4  # Learning rate for optimizer
    gamma: float = 0.99  # Discount factor for rewards
    GAE: float = 0.95  # Generalized Advantage Estimation lambda
    n_envs: int = 1  # Number of parallel environments for data collection
    max_steps: int = 128  # Maximum steps per rollout before updating
    num_minibatches: int = 4  # Number of minibatches for PPO updates
    PPO_EPOCHS: int = 4  # Number of PPO epochs per update
    clip_value: float = 0.2  # PPO clipping value for policy ratio
    ENTROPY_COEFF: float = 0.01  # Entropy coefficient for exploration
    VALUE_COEFF: float = 0.5  # Value loss coefficient in total loss
    num_eval_episodes: int = 5  # Number of evaluation episodes
    anneal_lr: bool = False  # Whether to anneal learning rate over time
    max_grad_norm: float = (
        0.0  # Maximum gradient norm for gradient clipping (0.0 to disable)
    )
    value_clip: bool = False  # Whether to apply value function clipping

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
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = (
        None,
    )  # Optional custom environment wrapper
    normalize_obs: bool = False  # Whether to normalize observations
    normalize_reward: bool = False  # Whether to normalize rewards
    log_gradients: bool = True  # Whether to log gradient norms to W&B
    device: str = "cpu"  # Device for training: "auto", "cpu", "cuda", or "cuda:0" etc.
    custom_agent: Optional[nn.Module] = None  # Custom neural network class or instance
    env: Optional[gym.Env] = None  # Optional pre-created environment


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

        self.fc1 = layer_init(nn.Linear(state_space, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, 32))
        self.out = layer_init(nn.Linear(32, action_space))

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

        if grid_env:
            env_to_use = OneHotWrapper(
                env_to_use, obs_shape=env_to_use.observation_space.n
            )

        if Config.normalize_reward:
            env_to_use = gym.wrappers.NormalizeReward(env_to_use)

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
                obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                action, _, _ = model.get_action(obs)
                action = action.cpu().numpy()
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    # For continuous action spaces, squeeze the batch dimension

                    action = action.squeeze(0) if action.shape[0] >= 1 else action

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


def train_ppo(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = Config.env,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    gamma: float = Config.gamma,
    GAE: float = Config.GAE,
    n_envs: int = Config.n_envs,
    max_steps: int = Config.max_steps,
    PPO_EPOCHS: int = Config.PPO_EPOCHS,
    clip_value: float = Config.clip_value,
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
    num_minibatches: int = Config.num_minibatches,
    log_gradients: bool = Config.log_gradients,
    num_eval_episodes: int = Config.num_eval_episodes,
    anneal_lr: bool = Config.anneal_lr,
    normalize_obs: bool = Config.normalize_obs,
    normalize_reward: bool = Config.normalize_reward,
    device: str = Config.device,
    value_clip: bool = False,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.lr = lr
    Config.gamma = gamma
    Config.GAE = GAE
    Config.n_envs = n_envs
    Config.max_steps = max_steps
    Config.PPO_EPOCHS = PPO_EPOCHS
    Config.clip_value = clip_value
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
    Config.num_minibatches = num_minibatches
    Config.log_gradients = log_gradients
    Config.num_eval_episodes = num_eval_episodes
    Config.anneal_lr = anneal_lr
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.value_clip = value_clip
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

    action_shape = (
        ()
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else (action_space_n,)
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

    # Compute derived values from passed parameters
    batch_size = Config.n_envs * Config.max_steps
    minibatch_size = batch_size // Config.num_minibatches
    num_updates = Config.total_timesteps // batch_size

    optimizer = optim.Adam(
        list(actor_network.parameters()) + list(critic_network.parameters()),
        lr=Config.lr,
        eps=1e-5,
    )

    # Tensor Storage

    obs_storage = torch.zeros(
        (Config.max_steps, Config.n_envs) + (obs_space_shape,)
    ).to(device)
    actions_storage = torch.zeros((Config.max_steps, Config.n_envs) + action_shape).to(
        device
    )
    logprobs_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    rewards_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    dones_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    values_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)

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
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout Phase
        for step in range(0, Config.max_steps):
            global_step = (update - 1) * batch_size + step * Config.n_envs
            obs_storage[step] = next_obs

            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, dist = actor_network.get_action(next_obs)
                value = critic_network(next_obs)

                # Log distribution statistics
                if Config.use_wandb:
                    if hasattr(dist, "probs"):
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
                    else:
                        # Continuous
                        wandb.log(
                            {
                                "policy/dist_mean": dist.mean.mean().item(),
                                "policy/dist_std": dist.stddev.mean().item(),
                                "policy/entropy": dist.entropy().mean().item(),
                                "global_step": global_step,
                            }
                        )

            values_storage[step] = value.flatten()
            # print(action_space_n, action.shape, actions_storage[step].shape, action_shape)
            actions_storage[step] = action
            logprobs_storage[step] = (
                logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
            )

            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            extrinsic_reward = torch.tensor(reward).to(device)
            rewards_storage[step] = extrinsic_reward
            next_obs = torch.Tensor(new_obs).to(device)
            next_done = torch.Tensor(done).to(device)

        # Calculate returns after the rollout is complete
        with torch.no_grad():
            # Get the bootstrapped value from the state after the last step
            bootstrap_value = critic_network(next_obs).squeeze()

            # Initialize tensors for advantages
            advantages = torch.zeros_like(rewards_storage).to(device)

            # Set the initial "next state" value. If an env was done, this is 0, otherwise it's the bootstrap value.
            lastgae = 0.0

            # Loop backwards from the last step to the first
            for t in reversed(range(Config.max_steps)):
                if t == Config.max_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_value = bootstrap_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    next_value = values_storage[t + 1] * nextnonterminal

                delta = (
                    rewards_storage[t] + Config.gamma * next_value - values_storage[t]
                )
                advantages[t] = lastgae = (
                    delta + Config.GAE * lastgae * nextnonterminal * Config.gamma
                )

        # Calculate returns
        returns = advantages + values_storage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten the batch
        b_obs = obs_storage.reshape((-1,) + (obs_space_shape,))
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)
        clipfracs = []

        # PPO Update Phase
        b_inds = np.arange(batch_size)
        for _epoch in range(Config.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # --- PPO Policy and Value Loss ---

                _, new_log_probs, dist = actor_network.get_action(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = (
                    new_log_probs.sum(dim=-1)
                    if len(new_log_probs.shape) > 1
                    else new_log_probs
                )
                logratio = new_log_probs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > Config.clip_value).float().mean().item()
                    ]
                    if Config.use_wandb:
                        wandb.log({"charts/approx_kl": approx_kl.item()})

                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(
                    ratio, 1 - Config.clip_value, 1 + Config.clip_value
                )
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                current_values = critic_network(b_obs[mb_inds])

                # Value clipping
                if Config.value_clip:
                    v_loss_unclipped = (current_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        current_values - b_values[mb_inds],
                        -Config.clip_value,
                        Config.clip_value,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)

                    critic_loss = Config.VALUE_COEFF * 0.5 * v_loss_max.mean()

                else:
                    critic_loss = (
                        Config.VALUE_COEFF
                        * 0.5
                        * F.mse_loss(current_values.squeeze(), b_returns[mb_inds])
                    )

                critic_loss = F.mse_loss(current_values.squeeze(), b_returns[mb_inds])
                entropy_loss = dist.entropy().mean()

                # Total Loss
                loss = (
                    policy_loss
                    + Config.VALUE_COEFF * critic_loss
                    - Config.ENTROPY_COEFF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()

                # Log gradient norm per layer
                if Config.use_wandb and Config.log_gradients:
                    # Calculate gradient norm before clipping
                    total_norm_before = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2)
                                for p in list(actor_network.parameters())
                                + list(critic_network.parameters())
                                if p.grad is not None
                            ]
                        ),
                        2,
                    )

                    for name, param in actor_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/actor_layer_{name}": grad_norm,
                                    "global_step": global_step,
                                }
                            )
                    for name, param in critic_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/critic_layer_{name}": grad_norm,
                                    "global_step": global_step,
                                }
                            )

                    wandb.log(
                        {
                            "gradients/norm_before_clip": total_norm_before.item(),
                            "global_step": global_step,
                        }
                    )

                # Apply gradient clipping
                if Config.max_grad_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(actor_network.parameters())
                        + list(critic_network.parameters()),
                        max_norm=Config.max_grad_norm,
                    )

                optimizer.step()

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
                    "losses/total_loss": loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "losses/value_loss": critic_loss.item(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "rewards/rewards_mean": np.mean(rewards_storage.cpu().numpy()),
                    "rewards/rewards_std": np.std(rewards_storage.cpu().numpy()),
                    "advantages/advantages_mean": b_advantages.mean().item(),
                    "advantages/advantages_std": b_advantages.std().item(),
                    "global_step": global_step,
                    "kl/approx_kl": approx_kl.item(),
                    "kl/old_approx_kl": old_approx_kl.item(),
                    "clipfrac/clip_fraction": np.mean(clipfracs),
                }
            )
            print(
                f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}, SPS: {int(global_step / (time.time() - start_time))}"
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
            model_path = f"runs/{run_name}/models/model_step_{global_step}.pth"
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


def train_ppo_cnn(
    env_id: Optional[str] = None,
    env: Optional[gym.Env] = Config.env,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    gamma: float = Config.gamma,
    GAE: float = Config.GAE,
    n_envs: int = Config.n_envs,
    max_steps: int = Config.max_steps,
    PPO_EPOCHS: int = Config.PPO_EPOCHS,
    clip_value: float = Config.clip_value,
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
    num_minibatches: int = Config.num_minibatches,
    log_gradients: bool = Config.log_gradients,
    num_eval_episodes: int = Config.num_eval_episodes,
    anneal_lr: bool = Config.anneal_lr,
    normalize_obs: bool = Config.normalize_obs,
    normalize_reward: bool = Config.normalize_reward,
    device: str = Config.device,
    value_clip: bool = False,
) -> nn.Module:
    # Update Config with passed arguments
    Config.env_id = env_id
    Config.total_timesteps = total_timesteps
    Config.seed = seed
    Config.lr = lr
    Config.gamma = gamma
    Config.GAE = GAE
    Config.n_envs = n_envs
    Config.max_steps = max_steps
    Config.PPO_EPOCHS = PPO_EPOCHS
    Config.clip_value = clip_value
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
    Config.num_minibatches = num_minibatches
    Config.log_gradients = log_gradients
    Config.num_eval_episodes = num_eval_episodes
    Config.anneal_lr = anneal_lr
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.value_clip = value_clip
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
        obs_space_shape = envs.single_observation_space.shape

    action_space_n = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else envs.single_action_space.shape
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

    # Compute derived values from passed parameters
    batch_size = Config.n_envs * Config.max_steps
    minibatch_size = batch_size // Config.num_minibatches
    num_updates = Config.total_timesteps // batch_size

    optimizer = optim.Adam(
        list(actor_network.parameters()) + list(critic_network.parameters()),
        lr=Config.lr,
        eps=1e-5,
    )

    # Tensor Storage

    obs_storage = torch.zeros((Config.max_steps, Config.n_envs) + obs_space_shape).to(
        device
    )
    actions_storage = torch.zeros((Config.max_steps, Config.n_envs) + action_shape).to(
        device
    )
    logprobs_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    rewards_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    dones_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)
    values_storage = torch.zeros((Config.max_steps, Config.n_envs)).to(device)

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
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout Phase
        for step in range(0, Config.max_steps):
            global_step = (update - 1) * batch_size + step * Config.n_envs
            obs_storage[step] = next_obs

            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, dist = actor_network.get_action(next_obs)
                value = critic_network(next_obs)

                # Log distribution statistics
                if Config.use_wandb:
                    if hasattr(dist, "probs"):
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
                    else:
                        # Continuous
                        wandb.log(
                            {
                                "policy/dist_mean": dist.mean.mean().item(),
                                "policy/dist_std": dist.stddev.mean().item(),
                                "policy/entropy": dist.entropy().mean().item(),
                                "global_step": global_step,
                            }
                        )

            values_storage[step] = value.flatten()
            # print(action_space_n, action.shape, actions_storage[step].shape, action_shape)
            actions_storage[step] = action
            logprobs_storage[step] = (
                logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
            )

            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            extrinsic_reward = torch.tensor(reward).to(device)
            rewards_storage[step] = extrinsic_reward
            next_obs = torch.Tensor(new_obs).to(device)
            next_done = torch.Tensor(done).to(device)

        # Calculate returns after the rollout is complete
        with torch.no_grad():
            # Get the bootstrapped value from the state after the last step
            bootstrap_value = critic_network(next_obs).squeeze()

            # Initialize tensors for advantages
            advantages = torch.zeros_like(rewards_storage).to(device)

            # Set the initial "next state" value. If an env was done, this is 0, otherwise it's the bootstrap value.
            lastgae = 0.0

            # Loop backwards from the last step to the first
            for t in reversed(range(Config.max_steps)):
                if t == Config.max_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_value = bootstrap_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    next_value = values_storage[t + 1] * nextnonterminal

                delta = (
                    rewards_storage[t] + Config.gamma * next_value - values_storage[t]
                )
                advantages[t] = lastgae = (
                    delta + Config.GAE * lastgae * nextnonterminal * Config.gamma
                )

        # Calculate returns
        returns = advantages + values_storage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten the batch
        b_obs = obs_storage.reshape((-1,) + obs_space_shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)
        clipfracs = []

        # PPO Update Phase
        b_inds = np.arange(batch_size)
        for _epoch in range(Config.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # --- PPO Policy and Value Loss ---

                _, new_log_probs, dist = actor_network.get_action(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = new_log_probs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > Config.clip_value).float().mean().item()
                    ]
                    if Config.use_wandb:
                        wandb.log({"charts/approx_kl": approx_kl.item()})

                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(
                    ratio, 1 - Config.clip_value, 1 + Config.clip_value
                )
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                current_values = critic_network(b_obs[mb_inds])

                # Value clipping
                if Config.value_clip:
                    v_loss_unclipped = (current_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        current_values - b_values[mb_inds],
                        -Config.clip_value,
                        Config.clip_value,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)

                    critic_loss = Config.VALUE_COEFF * 0.5 * v_loss_max.mean()

                else:
                    critic_loss = (
                        Config.VALUE_COEFF
                        * 0.5
                        * F.mse_loss(current_values.squeeze(), b_returns[mb_inds])
                    )

                critic_loss = F.mse_loss(current_values.squeeze(), b_returns[mb_inds])
                entropy_loss = dist.entropy().mean()

                # Total Loss
                loss = (
                    policy_loss
                    + Config.VALUE_COEFF * critic_loss
                    - Config.ENTROPY_COEFF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()

                # Log gradient norm per layer
                if Config.use_wandb and Config.log_gradients:
                    # Calculate gradient norm before clipping
                    total_norm_before = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2)
                                for p in list(actor_network.parameters())
                                + list(critic_network.parameters())
                                if p.grad is not None
                            ]
                        ),
                        2,
                    )

                    for name, param in actor_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/actor_layer_{name}": grad_norm,
                                    "global_step": global_step,
                                }
                            )
                    for name, param in critic_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/critic_layer_{name}": grad_norm,
                                    "global_step": global_step,
                                }
                            )

                    wandb.log(
                        {
                            "gradients/norm_before_clip": total_norm_before.item(),
                            "global_step": global_step,
                        }
                    )

                # Apply gradient clipping
                if Config.max_grad_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(actor_network.parameters())
                        + list(critic_network.parameters()),
                        max_norm=Config.max_grad_norm,
                    )

                optimizer.step()

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
                    "losses/total_loss": loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "losses/value_loss": critic_loss.item(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "rewards/rewards_mean": np.mean(rewards_storage.cpu().numpy()),
                    "rewards/rewards_std": np.std(rewards_storage.cpu().numpy()),
                    "advantages/advantages_mean": b_advantages.mean().item(),
                    "advantages/advantages_std": b_advantages.std().item(),
                    "global_step": global_step,
                    "kl/approx_kl": approx_kl.item(),
                    "kl/old_approx_kl": old_approx_kl.item(),
                    "clipfrac/clip_fraction": np.mean(clipfracs),
                }
            )
            print(
                f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}, SPS: {int(global_step / (time.time() - start_time))}"
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
            model_path = f"runs/{run_name}/models/model_step_{global_step}.pth"
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
    train_ppo()
