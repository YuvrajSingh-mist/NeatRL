import os
import random
import time
from typing import Callable, Optional, Union

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    NormalizeObservation,
    NormalizeReward,
)
from tqdm import tqdm

import wandb

gym.register_envs(ale_py)


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name: str = "REINFORCE"
    seed: int = 42
    env_id: Optional[str] = "CartPole-v1"

    # Training parameters
    episodes: int = 2000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    max_grad_norm: float = 1.0  # Maximum gradient norm for gradient clipping
    num_eval_eps: int = 10
    grid_env: bool = False
    use_entropy: bool = False
    entropy_coeff: float = 0.01
    anneal_lr: bool = True  # Whether to anneal learning rate over time
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None

    # Evaluation & logging
    eval_every: int = 100
    save_every: int = 1000
    upload_every: int = 100
    atari_wrapper: bool = False
    n_envs: int = 4
    capture_video: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Custom agent
    custom_agent: Optional[Union[nn.Module, type]] = (
        None  # Custom neural network class or instance
    )

    # Normalization
    normalize_obs: bool = False
    normalize_reward: bool = False

    # Logging options
    log_gradients: bool = False

    # Logging & saving
    use_wandb: bool = False
    wandb_project: str = "cleanRL"
    wandb_entity: str = ""
    buffer_size: int = 10000
    tau: float = 1.0
    target_network_frequency: int = 50
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 1000
    train_frequency: int = 10
    env: Optional[gym.Env] = None  # Optional pre-created environment


# For discrete actions
class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x = self.out(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))
        x = torch.nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        return x

    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(
            action_probs
        )  # Create a categorical distribution from the probabilities

        action = dist.sample()  # Sample an action from the distribution
        return action, dist.log_prob(action), dist


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_shape=16):
        super().__init__(env)
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs):
        one_hot = torch.zeros(self.obs_shape, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()


def make_env(
    env_id: Optional[str],
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
        elif env_id is not None:
            # Create new environment
            env_to_use = gym.make(env_id, render_mode=render_mode)
        else:
            raise ValueError("Either env or env_id must be provided")

        env_to_use = gym.wrappers.RecordEpisodeStatistics(env_to_use)
        if Config.normalize_reward:
            env_to_use = NormalizeReward(env_to_use)
        if grid_env:
            env_to_use = OneHotWrapper(
                env_to_use, obs_shape=env_to_use.observation_space.n
            )
        if Config.normalize_obs:
            env_to_use = NormalizeObservation(env_to_use)
        if atari_wrapper:
            env_to_use = AtariPreprocessing(
                env_to_use, grayscale_obs=True, scale_obs=True
            )
            env_to_use = FrameStackObservation(env_to_use, stack_size=4)

        if env_wrapper:
            env_to_use = env_wrapper(env_to_use)

        env_to_use.action_space.seed(seed + idx)
        return env_to_use

    return thunk


def evaluate(
    env_id=None,
    env=None,
    model=None,
    device=None,
    seed=None,
    num_eval_eps=10,
    env_wrapper=None,
    capture_video=False,
    atari_wrapper=False,
    grid_env=False,
):
    eval_env = make_env(
        env_id=env_id,
        env=env,
        seed=seed,
        idx=0,
        render_mode="rgb_array",
        atari_wrapper=atari_wrapper,
        grid_env=grid_env,
        env_wrapper=env_wrapper,
    )()
    eval_env.action_space.seed(seed)

    model = model.to(device)
    model = model.eval()
    returns = []
    frames = []

    for _ in range(num_eval_eps):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if capture_video:
                frame = eval_env.render()
                frames.append(frame)

            with torch.no_grad():
                action = model.get_action(
                    torch.tensor(obs, device=device, dtype=torch.float32)
                )
                if len(action) == 2:
                    action, _ = action
                elif len(action) == 3:
                    action, _, _ = action

                # Handle both discrete and continuous action spaces
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    action = action.detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward

        returns.append(episode_reward)

        # Save video
        if frames:
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
    model.train()
    eval_env.close()
    return returns, frames


def calculate_param_norm(model):
    """Calculate the L2 norm of all parameters in a model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def validate_policy_network_dimensions(policy_network, obs_dim, action_dim):
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


def train_reinforce(
    env_id=None,
    env=Config.env,
    total_steps=Config.episodes,
    seed=Config.seed,
    learning_rate=Config.learning_rate,
    gamma=Config.gamma,
    max_grad_norm=Config.max_grad_norm,
    capture_video=Config.capture_video,
    use_wandb=Config.use_wandb,
    wandb_project=Config.wandb_project,
    wandb_entity=Config.wandb_entity,
    exp_name=Config.exp_name,
    eval_every=Config.eval_every,
    save_every=Config.save_every,
    atari_wrapper=Config.atari_wrapper,
    custom_agent=Config.custom_agent,
    num_eval_eps=Config.num_eval_eps,
    n_envs=Config.n_envs,
    device=Config.device,
    grid_env=Config.grid_env,
    use_entropy=Config.use_entropy,
    entropy_coeff=Config.entropy_coeff,
    normalize_obs=Config.normalize_obs,
    normalize_reward=Config.normalize_reward,
    log_gradients=Config.log_gradients,
    anneal_lr=Config.anneal_lr,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
):
    """
    Train a REINFORCE agent on a Gymnasium environment.

    Args:
        env_id: Optional Gymnasium environment ID (ignored if env is provided)
        env: Optional pre-created Gymnasium environment (if provided, env_id is ignored)
        total_steps: Number of steps to train
        seed: Random seed
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        max_grad_norm: Maximum gradient norm for gradient clipping (0.0 to disable)
        capture_video: Whether to capture training videos
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/username
        exp_name: Experiment name
        eval_every: Frequency of evaluation during training
        save_every: Frequency of saving the model
        atari_wrapper: Whether to apply Atari preprocessing wrappers
        custom_agent: Custom neural network class or instance (nn.Module subclass or instance, optional, defaults to PolicyNet)
        num_eval_eps: Number of evaluation episodes
        n_envs: Number of parallel environments (currently not used for training, kept for compatibility)
        device: Device to use for training (e.g., "cpu", "cuda")
        grid_env: Whether the environment uses discrete grid observations
        use_entropy: Whether to include an entropy bonus in the loss
        entropy_coeff: Coefficient for the entropy bonus (only used if use_entropy=True)
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        log_gradients: Whether to log gradient norms to WandB
        anneal_lr: Whether to anneal learning rate over time
        env_wrapper: Optional environment wrapper function
    Returns:
        Trained Policy-network model
    """
    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"

    # Set Config attributes from function arguments
    if env is not None and env_id is not None:
        raise ValueError(
            "Cannot provide both 'env' and 'env_id'. Provide either 'env' (pre-created environment) or 'env_id' (environment ID), not both."
        )

    Config.env = env
    Config.env_id = env_id if env_id is not None else Config.env_id

    Config.episodes = total_steps
    Config.seed = seed
    Config.learning_rate = learning_rate
    Config.gamma = gamma
    Config.max_grad_norm = max_grad_norm
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.wandb_entity = wandb_entity
    Config.exp_name = exp_name
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.atari_wrapper = atari_wrapper
    Config.custom_agent = custom_agent
    Config.num_eval_eps = num_eval_eps
    Config.n_envs = n_envs
    Config.device = device
    Config.grid_env = grid_env
    Config.use_entropy = use_entropy
    Config.entropy_coeff = entropy_coeff
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.log_gradients = log_gradients
    Config.anneal_lr = anneal_lr
    Config.env_wrapper = env_wrapper

    # Initialize WandB
    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            entity=Config.wandb_entity,
            sync_tensorboard=False,
            config=locals(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Warn if entropy_coeff is set but use_entropy is False
    if not Config.use_entropy and Config.entropy_coeff != 0.0:
        print(
            f"Warning: entropy_coeff={Config.entropy_coeff} is provided but use_entropy=False. Entropy regularization will not be applied."
        )

    if Config.capture_video:
        os.makedirs(f"videos/{run_name}/train", exist_ok=True)
        os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    # Set seeds
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)

    # setting up the device
    Config.device = torch.device(Config.device)

    if Config.device.type == "cuda" and not torch.cuda.is_available():
        Config.device = torch.device("cpu")
        print("CUDA not available, falling back to CPU")

    if Config.device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        torch.backends.cudnn.benchmark = False

    elif Config.device.type == "mps":
        torch.mps.manual_seed(Config.seed)

    if Config.env is not None:
        env = Config.env

    if Config.n_envs > 1:
        env = gym.vector.SyncVectorEnv(
            [
                make_env(
                    env_id=Config.env_id,
                    env=Config.env,
                    seed=Config.seed,
                    idx=i,
                    render_mode="rgb_array",
                    atari_wrapper=Config.atari_wrapper,
                    grid_env=Config.grid_env,
                    env_wrapper=Config.env_wrapper,
                )
                for i in range(Config.n_envs)
            ]
        )
    else:
        env = make_env(
            env_id=Config.env_id,
            env=Config.env,
            seed=Config.seed,
            idx=0,
            render_mode="rgb_array",
            atari_wrapper=Config.atari_wrapper,
            grid_env=Config.grid_env,
            env_wrapper=Config.env_wrapper,
        )()

    # Determine if we're dealing with discrete observation spaces
    if Config.n_envs > 1:
        is_discrete_obs = isinstance(env.single_observation_space, gym.spaces.Discrete)
        obs_space = env.single_observation_space
    else:
        is_discrete_obs = isinstance(env.observation_space, gym.spaces.Discrete)
        obs_space = env.observation_space

    # Compute observation dimensions
    if is_discrete_obs:
        obs_shape = obs_space.n
    else:
        obs_shape = obs_space.shape[0]

    # Compute action dimensions
    if Config.n_envs > 1:
        action_shape = (
            env.single_action_space.n
            if isinstance(env.single_action_space, gym.spaces.Discrete)
            else env.single_action_space.shape[0]
        )
    else:
        action_shape = (
            env.action_space.n
            if isinstance(env.action_space, gym.spaces.Discrete)
            else env.action_space.shape[0]
        )

    # Use custom agent if provided, otherwise use default PolicyNet
    if Config.custom_agent is not None:
        if isinstance(Config.custom_agent, nn.Module):
            # Custom agent is an instance
            validate_policy_network_dimensions(
                Config.custom_agent, obs_shape, action_shape
            )
            policy_network = Config.custom_agent.to(Config.device)
        elif isinstance(Config.custom_agent, type) and issubclass(
            Config.custom_agent, nn.Module
        ):
            # Custom agent is a class
            policy_network = Config.custom_agent(obs_shape, action_shape).to(
                Config.device
            )
        else:
            raise ValueError(
                "custom_agent must be an instance of nn.Module or a subclass of nn.Module"
            )
    else:
        policy_network = PolicyNet(obs_shape, action_shape).to(Config.device)

    optimizer = optim.Adam(policy_network.parameters(), lr=Config.learning_rate)

    # Print network architecture
    print("Policy-Network Architecture:")
    print(policy_network)

    # Log network architecture to WandB
    if use_wandb:
        wandb.config.update({"network_architecture": str(policy_network)})

    policy_network.train()

    start_time = time.time()

    updates = Config.episodes // Config.n_envs

    for step in tqdm(range(updates)):
        global_step = step * Config.n_envs
        obs, _ = env.reset()
        rewards = []
        log_probs = []
        entropies = []
        done = False

        # Annealing the rate if instructed to do so.
        if Config.anneal_lr:
            frac = 1.0 - (step - 1.0) / updates
            lrnow = frac * Config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        while True:
            result = policy_network.get_action(
                torch.tensor(obs, device=Config.device, dtype=torch.float32)
            )
            if len(result) == 2:
                action, log_prob = result
                log_prob = log_prob.sum(dim=-1) if len(log_prob.shape) > 1 else log_prob

                dist = None

            elif len(result) == 3:
                action, log_prob, dist = result
                log_prob = log_prob.sum(dim=-1) if len(log_prob.shape) > 1 else log_prob

            else:
                raise ValueError(
                    f"Error unpacking result from get_action. Expected 3 got {len(result)}"
                )

            if Config.use_entropy and dist is None:
                raise ValueError(
                    "use_entropy is True but get_action did not return dist"
                )

            # Handle both discrete and continuous action spaces
            if Config.n_envs > 1:
                action_space = env.single_action_space
                # For vectorized environments, convert to numpy array of actions

                action = action.detach().cpu().numpy()
            else:
                action_space = env.action_space
                # For single environment, convert to scalar or array
                if isinstance(action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    action = action.detach().cpu().numpy()

            new_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            log_probs.append(log_prob)
            if Config.use_entropy:
                entropies.append(dist.entropy())

            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/action_mean": np.mean(action),
                        "charts/action_std": np.std(action),
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "step": global_step,
                    }
                )

                if Config.n_envs > 1:
                    reward = np.array(reward)
                    wandb.log(
                        {
                            "rewards/reward_mean": reward.mean(),
                            "rewards/reward_std": np.std(reward),
                            "step": global_step,
                        }
                    )
                else:
                    wandb.log({"rewards/reward": reward, "step": global_step})

                if dist is not None:
                    wandb.log(
                        {
                            "charts/dist_mean": dist.mean.mean().item(),
                            "charts/dist_std": dist.stddev.mean().item(),
                            "step": global_step,
                        }
                    )

            done = np.logical_or(terminated, truncated)
            obs = new_obs
            if np.all(done):
                break

        # Calculate returns
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + Config.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns, device=Config.device, dtype=torch.float32
        ).detach()

        if Config.use_wandb:
            wandb.log(
                {"charts/returns_mean": returns.mean().item(), "step": global_step}
            )

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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
                                "step": global_step,
                            }
                        )

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/log_prob": log_prob.mean().item(),
                        "charts/log_probs_std": log_prob.std().item(),
                        "step": global_step,
                    }
                )

            policy_loss.append(-log_prob * R)  # Negative for gradient ascent

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        if Config.use_entropy:
            entropy_loss = torch.stack(entropies).mean() * Config.entropy_coeff

            if Config.use_wandb:
                wandb.log(
                    {"losses/entropy_loss": entropy_loss.item(), "step": global_step}
                )

            loss = loss - entropy_loss
        loss.backward()

        # Calculate gradient norm before clipping
        total_norm_before = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in policy_network.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        )

        # Log gradient norm per layer
        if Config.use_wandb and Config.log_gradients:
            for name, param in policy_network.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.detach(), 2).item()
                    wandb.log(
                        {
                            f"gradients/layer_{name}": grad_norm,
                            "step": global_step,
                        }
                    )

        # Log gradient norm
        if Config.use_wandb and Config.log_gradients:
            wandb.log(
                {
                    "gradients/norm_before_clip": total_norm_before.item(),
                    "step": global_step,
                }
            )
            # Apply gradient clipping
            if Config.max_grad_norm != 0.0:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    policy_network.parameters(), max_norm=Config.max_grad_norm
                )

        optimizer.step()

        # Log loss and metrics every 100 episodes
        if step % 100 == 0:
            if Config.use_wandb:
                wandb.log(
                    {
                        "losses/policy_loss": loss.item(),
                        "step": global_step,
                    }
                )

        # Print progress every 1000 steps
        if step % 10 == 0:
            print(
                f"Step {step}, Policy Loss: {loss.item():.4f}, SPS: {int(step / (time.time() - start_time))}"
            )
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/SPS": int(step / (time.time() - start_time)),
                        "step": global_step,
                    }
                )

        # Model evaluation & saving
        if step % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id=Config.env_id,
                env=Config.env,
                model=policy_network,
                device=Config.device,
                seed=Config.seed,
                env_wrapper=Config.env_wrapper,
                num_eval_eps=Config.num_eval_eps,
                capture_video=Config.capture_video,
                atari_wrapper=Config.atari_wrapper,
                grid_env=Config.grid_env,
            )
            avg_return = np.mean(episodic_returns)

            if Config.use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "step": global_step})
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if Config.use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / (time.time() - start_time + 1e-8)),
                    "step": global_step,
                }
            )

        if step % Config.save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/reinforce_model_episode_{step}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(policy_network.state_dict(), model_path)
            print(f"Model saved at episode {step} to {model_path}")

    # Save final video to WandB
    if Config.use_wandb:
        train_video_path = "videos/final.mp4"
        _, frames = evaluate(
            env_id=Config.env_id,
            env=Config.env,
            model=policy_network,
            device=Config.device,
            seed=Config.seed,
            env_wrapper=Config.env_wrapper,
            num_eval_eps=Config.num_eval_eps,
            capture_video=Config.capture_video,
            atari_wrapper=Config.atari_wrapper,
            grid_env=Config.grid_env,
        )
        imageio.mimsave(train_video_path, frames, fps=30)
        print(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()

    return policy_network


def train_reinforce_cnn(
    env_id=None,
    env=Config.env,
    total_steps=Config.episodes,
    seed=Config.seed,
    learning_rate=Config.learning_rate,
    gamma=Config.gamma,
    max_grad_norm=Config.max_grad_norm,
    capture_video=Config.capture_video,
    use_wandb=Config.use_wandb,
    wandb_project=Config.wandb_project,
    wandb_entity=Config.wandb_entity,
    exp_name=Config.exp_name,
    eval_every=Config.eval_every,
    save_every=Config.save_every,
    atari_wrapper=Config.atari_wrapper,
    custom_agent=Config.custom_agent,
    num_eval_eps=Config.num_eval_eps,
    n_envs=Config.n_envs,
    device=Config.device,
    grid_env=Config.grid_env,
    use_entropy=Config.use_entropy,
    entropy_coeff=Config.entropy_coeff,
    normalize_obs=Config.normalize_obs,
    normalize_reward=Config.normalize_reward,
    log_gradients=Config.log_gradients,
    anneal_lr=Config.anneal_lr,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
):
    """
    Train a REINFORCE agent on a Gymnasium environment.

    Args:
        env_id: Optional Gymnasium environment ID (ignored if env is provided)
        env: Optional pre-created Gymnasium environment (if provided, env_id is ignored)
        total_steps: Number of steps to train
        seed: Random seed
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        max_grad_norm: Maximum gradient norm for gradient clipping (0.0 to disable)
        capture_video: Whether to capture training videos
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/username
        exp_name: Experiment name
        eval_every: Frequency of evaluation during training
        save_every: Frequency of saving the model
        atari_wrapper: Whether to apply Atari preprocessing wrappers
        custom_agent: Custom neural network class or instance (nn.Module subclass or instance, optional, defaults to PolicyNet)
        num_eval_eps: Number of evaluation episodes
        n_envs: Number of parallel environments (currently not used for training, kept for compatibility)
        device: Device to use for training (e.g., "cpu", "cuda")
        grid_env: Whether the environment uses discrete grid observations
        use_entropy: Whether to include an entropy bonus in the loss
        entropy_coeff: Coefficient for the entropy bonus (only used if use_entropy=True)
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        env_wrapper: Optional environment wrapper function
        log_gradients: Whether to log gradient norms to WandB
        anneal_lr: Whether to anneal learning rate over time
    Returns:
        Trained Policy-network model
    """
    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"

    # Set Config attributes from function arguments
    if env is not None and env_id is not None:
        raise ValueError(
            "Cannot provide both 'env' and 'env_id'. Provide either 'env' (pre-created environment) or 'env_id' (environment ID), not both."
        )

    Config.env = env
    Config.env_id = env_id if env_id is not None else Config.env_id

    Config.episodes = total_steps
    Config.seed = seed
    Config.learning_rate = learning_rate
    Config.gamma = gamma
    Config.max_grad_norm = max_grad_norm
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.wandb_entity = wandb_entity
    Config.exp_name = exp_name
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.atari_wrapper = atari_wrapper
    Config.custom_agent = custom_agent
    Config.num_eval_eps = num_eval_eps
    Config.n_envs = n_envs
    Config.device = device
    Config.grid_env = grid_env
    Config.use_entropy = use_entropy
    Config.entropy_coeff = entropy_coeff
    Config.normalize_obs = normalize_obs
    Config.normalize_reward = normalize_reward
    Config.log_gradients = log_gradients
    Config.anneal_lr = anneal_lr
    Config.env_wrapper = None

    if env is not None and Config.env_id != env_id:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for default environments or env for custom environments."
        )

    # Initialize WandB
    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            entity=Config.wandb_entity,
            sync_tensorboard=False,
            config=locals(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Warn if entropy_coeff is set but use_entropy is False
    if not Config.use_entropy and Config.entropy_coeff != 0.0:
        print(
            f"Warning: entropy_coeff={Config.entropy_coeff} is provided but use_entropy=False. Entropy regularization will not be applied."
        )

    if Config.capture_video:
        os.makedirs(f"videos/{run_name}/train", exist_ok=True)
        os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    # Set seeds
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)

    # setting up the device
    Config.device = torch.device(Config.device)

    if Config.device.type == "cuda" and not torch.cuda.is_available():
        Config.device = torch.device("cpu")
        print("CUDA not available, falling back to CPU")

    if Config.device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        torch.backends.cudnn.benchmark = False

    elif Config.device.type == "mps":
        torch.mps.manual_seed(Config.seed)

    if Config.env is not None:
        env = Config.env

    if Config.n_envs > 1:
        env = gym.vector.SyncVectorEnv(
            [
                make_env(
                    env_id=Config.env_id,
                    env=Config.env,
                    seed=Config.seed,
                    idx=i,
                    render_mode="rgb_array",
                    atari_wrapper=Config.atari_wrapper,
                    grid_env=Config.grid_env,
                    env_wrapper=Config.env_wrapper,
                )
                for i in range(Config.n_envs)
            ]
        )
    else:
        env = make_env(
            env_id=Config.env_id,
            env=Config.env,
            seed=Config.seed,
            idx=0,
            render_mode="rgb_array",
            atari_wrapper=Config.atari_wrapper,
            grid_env=Config.grid_env,
            env_wrapper=Config.env_wrapper,
        )()

    # Determine if we're dealing with discrete observation spaces
    if Config.n_envs > 1:
        is_discrete_obs = isinstance(env.single_observation_space, gym.spaces.Discrete)
        obs_space = env.single_observation_space
    else:
        is_discrete_obs = isinstance(env.observation_space, gym.spaces.Discrete)
        obs_space = env.observation_space

    # Compute observation dimensions
    if is_discrete_obs:
        obs_shape = obs_space.n
    else:
        obs_shape = obs_space.shape

    # Compute action dimensions
    if Config.n_envs > 1:
        action_shape = (
            env.single_action_space.n
            if isinstance(env.single_action_space, gym.spaces.Discrete)
            else env.single_action_space.shape
        )
    else:
        action_shape = (
            env.action_space.n
            if isinstance(env.action_space, gym.spaces.Discrete)
            else env.action_space.shape
        )

    # Use custom agent if provided, otherwise use default PolicyNet
    if Config.custom_agent is not None:
        if isinstance(Config.custom_agent, nn.Module):
            # Custom agent is an instance
            validate_policy_network_dimensions(
                Config.custom_agent, obs_shape, action_shape
            )
            policy_network = Config.custom_agent.to(Config.device)
        elif isinstance(Config.custom_agent, type) and issubclass(
            Config.custom_agent, nn.Module
        ):
            # Custom agent is a class
            policy_network = Config.custom_agent(action_shape).to(Config.device)
        else:
            raise ValueError(
                "custom_agent must be an instance of nn.Module or a subclass of nn.Module"
            )
    else:
        policy_network = PolicyNet(obs_shape, action_shape).to(Config.device)

    optimizer = optim.Adam(policy_network.parameters(), lr=Config.learning_rate)

    # Print network architecture

    print("Policy-Network Architecture:")
    print(policy_network)

    # Log network architecture to WandB
    if use_wandb:
        wandb.config.update({"network_architecture": str(policy_network)})

    policy_network.train()

    start_time = time.time()

    updates = Config.episodes // Config.n_envs

    for step in tqdm(range(updates)):
        global_step = step * Config.n_envs
        obs, _ = env.reset()
        rewards = []
        log_probs = []
        entropies = []
        done = False

        # Annealing the rate if instructed to do so.
        if Config.anneal_lr:
            frac = 1.0 - (step - 1.0) / updates
            lrnow = frac * Config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        while True:
            result = policy_network.get_action(
                torch.tensor(obs, device=Config.device, dtype=torch.float32)
            )
            if len(result) == 2:
                action, log_prob = result
                log_prob = log_prob.sum(dim=-1) if len(log_prob.shape) > 1 else log_prob

                dist = None

            elif len(result) == 3:
                action, log_prob, dist = result
                log_prob = log_prob.sum(dim=-1) if len(log_prob.shape) > 1 else log_prob

            else:
                raise ValueError(
                    f"Error unpacking result from get_action. Expected 3 got {len(result)}"
                )

            if Config.use_entropy and dist is None:
                raise ValueError(
                    "use_entropy is True but get_action did not return dist"
                )

            # Handle both discrete and continuous action spaces
            if Config.n_envs > 1:
                action_space = env.single_action_space
                # For vectorized environments, convert to numpy array of actions

                action = action.detach().cpu().numpy()
            else:
                action_space = env.action_space
                # For single environment, convert to scalar or array
                if isinstance(action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    action = action.detach().cpu().numpy()

            new_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            log_probs.append(log_prob)
            if Config.use_entropy:
                entropies.append(dist.entropy())

            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/action_mean": np.mean(action),
                        "charts/action_std": np.std(action),
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "step": global_step,
                    }
                )

                if Config.n_envs > 1:
                    reward = np.array(reward)
                    wandb.log(
                        {
                            "rewards/reward_mean": reward.mean(),
                            "rewards/reward_std": np.std(reward),
                            "step": global_step,
                        }
                    )
                else:
                    wandb.log({"rewards/reward": reward, "step": global_step})

                if dist is not None:
                    wandb.log(
                        {
                            "charts/dist_mean": dist.mean.mean().item(),
                            "charts/dist_std": dist.stddev.mean().item(),
                            "step": global_step,
                        }
                    )

            done = np.logical_or(terminated, truncated)
            obs = new_obs
            if np.all(done):
                break

        # Calculate returns
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + Config.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns, device=Config.device, dtype=torch.float32
        ).detach()

        if Config.use_wandb:
            wandb.log(
                {"charts/returns_mean": returns.mean().item(), "step": global_step}
            )

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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
                                "step": global_step,
                            }
                        )

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/log_prob": log_prob.mean().item(),
                        "charts/log_probs_std": log_prob.std().item(),
                        "step": global_step,
                    }
                )

            policy_loss.append(-log_prob * R)  # Negative for gradient ascent

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        if Config.use_entropy:
            entropy_loss = torch.stack(entropies).mean() * Config.entropy_coeff

            if Config.use_wandb:
                wandb.log(
                    {"losses/entropy_loss": entropy_loss.item(), "step": global_step}
                )

            loss = loss - entropy_loss
        loss.backward()

        # Calculate gradient norm before clipping
        total_norm_before = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in policy_network.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        )

        # Log gradient norm per layer
        if Config.use_wandb and Config.log_gradients:
            for name, param in policy_network.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.detach(), 2).item()
                    wandb.log(
                        {
                            f"gradients/layer_{name}": grad_norm,
                            "step": global_step,
                        }
                    )

        # Log gradient norm
        if Config.use_wandb and Config.log_gradients:
            wandb.log(
                {
                    "gradients/norm_before_clip": total_norm_before.item(),
                    "step": global_step,
                }
            )
            # Apply gradient clipping
            if Config.max_grad_norm != 0.0:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    policy_network.parameters(), max_norm=Config.max_grad_norm
                )

        optimizer.step()

        # Log loss and metrics every 100 episodes
        if step % 100 == 0:
            if Config.use_wandb:
                wandb.log(
                    {
                        "losses/policy_loss": loss.item(),
                        "step": global_step,
                    }
                )

        # Print progress every 1000 steps
        if step % 10 == 0:
            print(
                f"Step {step}, Policy Loss: {loss.item():.4f}, SPS: {int(step / (time.time() - start_time))}"
            )
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/SPS": int(step / (time.time() - start_time)),
                        "step": global_step,
                    }
                )

        # Model evaluation & saving
        if step % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id=Config.env_id,
                env=Config.env,
                model=policy_network,
                device=Config.device,
                seed=Config.seed,
                env_wrapper=Config.env_wrapper,
                num_eval_eps=Config.num_eval_eps,
                capture_video=Config.capture_video,
                atari_wrapper=Config.atari_wrapper,
                grid_env=Config.grid_env,
            )
            avg_return = np.mean(episodic_returns)

            if Config.use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "step": global_step})
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if Config.use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / (time.time() - start_time + 1e-8)),
                    "step": global_step,
                }
            )

        if step % Config.save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/reinforce_model_episode_{step}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(policy_network.state_dict(), model_path)
            print(f"Model saved at episode {step} to {model_path}")

    # Save final video to WandB
    if Config.use_wandb:
        train_video_path = "videos/final.mp4"
        _, frames = evaluate(
            env_id=Config.env_id,
            env=Config.env,
            model=policy_network,
            device=Config.device,
            seed=Config.seed,
            env_wrapper=Config.env_wrapper,
            num_eval_eps=Config.num_eval_eps,
            capture_video=Config.capture_video,
            atari_wrapper=Config.atari_wrapper,
            grid_env=Config.grid_env,
        )
        imageio.mimsave(train_video_path, frames, fps=30)
        print(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()

    return policy_network


if __name__ == "__main__":
    train_reinforce()
