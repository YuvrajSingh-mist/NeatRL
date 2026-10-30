"""Deep Q-Network (DQN) with a CNN for Atari-style pixel observations.

Follows the setup of Mnih et al. (2015): 84x84 grayscale frames, 4-frame
stacks, frame skip 4, no-op starts, reward clipping and a periodically synced
target network. ``estimate_values`` mirrors Figure 3 of van Hasselt et al.
(2016) by reporting the agent's own max-Q next to the return it actually earns,
which is where Q-learning's overestimation shows up. For the Double DQN fix,
see :mod:`neatrl.double_dqn_cnn`.
"""

import copy
import os
import random
import sys
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

from .cli import Dashboard
from .utils import configure_logging, get_logger, setup_device
from .utils.nn_utils import validate_q_network_dimensions

logger = get_logger(__name__)

try:
    import ale_py  # type: ignore[import-untyped]

    gym.register_envs(ale_py)
except ImportError:
    pass


# ===== CONFIGURATION =====
@dataclass
class Config:
    """Hyperparameters and settings for Atari DQN training."""

    # Experiment settings
    exp_name: str = "DQN-CNN"
    seed: int = 42
    env_id: Optional[str] = "ALE/Pong-v5"

    # Training parameters. Defaults follow Mnih et al. (2015) where practical;
    # total_timesteps and buffer_size are reduced so one run fits on a laptop.
    total_timesteps: int = 1_000_000  # paper: 50M agent steps (200M frames)
    lr: float = 1e-4  # paper: RMSprop 2.5e-4, alpha 0.95, eps 0.01
    buffer_size: int = 100_000  # paper: 1M transitions
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 -> hard target copies, as in the paper
    target_network_frequency: int = 10_000  # paper: every 10k agent steps
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.1  # paper anneals 1.0 -> 0.1
    exploration_fraction: float = 0.1
    learning_starts: int = 20_000
    train_frequency: int = 4
    max_grad_norm: float = 10.0  # paper clips the gradient norm at 10

    # Atari preprocessing
    atari_wrapper: bool = True
    frame_stack: int = 4
    screen_size: int = 84
    clip_reward: bool = True  # sign() clipping to {-1, 0, 1}

    # Logging & saving
    capture_video: bool = False
    use_wandb: bool = True
    wandb_project: str = "cleanRL"
    wandb_entity: str = ""
    eval_every: int = 50_000
    save_every: int = 100_000
    num_eval_eps: int = 3
    log_gradients: bool = False
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None
    device: Union[str, torch.device] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class QNetCNN(nn.Module):
    """Convolutional Q-network from Mnih et al. (2015), for 84x84 frame stacks."""

    def __init__(self, obs_shape, action_space):
        super().__init__()
        channels = int(obs_shape[0])
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Infer the flattened conv output so other screen sizes also work.
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            n_flat = self._features(dummy).shape[1]

        self.fc1 = nn.Linear(n_flat, 512)
        self.out = nn.Linear(512, action_space)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional trunk and flatten its output."""
        # Keep observations as uint8 in the replay buffer to save memory and
        # normalise them here instead.
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns one Q-value per discrete action."""
        return self.out(torch.nn.functional.relu(self.fc1(self._features(x))))


def _obs_tensor(obs: Any, device: Union[str, torch.device]) -> torch.Tensor:
    """Add a batch dimension, keeping uint8 pixels compact."""
    array = np.asarray(obs)
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    return torch.as_tensor(array, device=device, dtype=dtype).unsqueeze(0)


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    render_mode: Optional[str] = None,
    atari_wrapper: bool = True,
    frame_stack: int = 4,
    screen_size: int = 84,
    clip_reward: bool = True,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    env: Optional[gym.Env] = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds and seeds an Atari-preprocessed environment."""
    if env_id and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium "
            "environments or env for custom environments."
        )

    def thunk() -> gym.Env:
        if env is not None:
            env_to_use = env
        else:
            make_kwargs: dict[str, Any] = {"render_mode": render_mode}
            if atari_wrapper:
                # AtariPreprocessing does its own frame skipping; the paper also
                # disables sticky actions.
                make_kwargs["frameskip"] = 1
                make_kwargs["repeat_action_probability"] = 0.0
            env_to_use = gym.make(env_id, **make_kwargs)

        # Wrapped innermost so episode returns stay in game-score units.
        if env is None:
            env_to_use = gym.wrappers.RecordEpisodeStatistics(env_to_use)

        if atari_wrapper:
            env_to_use = gym.wrappers.AtariPreprocessing(
                env_to_use,
                noop_max=30,
                frame_skip=4,
                screen_size=screen_size,
                terminal_on_life_loss=False,
                grayscale_obs=True,
                scale_obs=False,  # keep uint8 pixels for a compact replay buffer
            )
            if clip_reward:
                env_to_use = gym.wrappers.TransformReward(
                    env_to_use, lambda reward: float(np.sign(reward))
                )
            env_to_use = gym.wrappers.FrameStackObservation(
                env_to_use, stack_size=frame_stack
            )

        if env_wrapper is not None:
            env_to_use = env_wrapper(env_to_use)

        env_to_use.action_space.seed(seed + idx)
        return env_to_use

    return thunk


def evaluate(
    model: nn.Module,
    device: Union[str, torch.device],
    env_id: str,
    seed: int = 42,
    atari_wrapper: bool = True,
    frame_stack: int = 4,
    screen_size: int = 84,
    clip_reward: bool = True,
    num_eval_eps: int = 3,
    capture_video: bool = False,
    use_wandb: bool = False,
) -> tuple[list[float], list[np.ndarray]]:
    """Run greedy evaluation episodes; return episodic returns and frames."""
    eval_env = make_env(
        env_id,
        seed,
        idx=0,
        render_mode="rgb_array" if capture_video else None,
        atari_wrapper=atari_wrapper,
        frame_stack=frame_stack,
        screen_size=screen_size,
        clip_reward=clip_reward,
    )()

    model = model.to(device)
    model.eval()
    returns: list[float] = []
    frames: list[np.ndarray] = []

    # Some envs are registered without a TimeLimit, so cap the episode length to
    # guarantee evaluation terminates.
    spec = getattr(eval_env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) or 10_000

    for _ in range(num_eval_eps):
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0.0

        for _step in range(max_steps):
            if capture_video:
                frames.append(eval_env.render())

            with torch.no_grad():
                q_values = model(_obs_tensor(obs, device))
            action = int(q_values.argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            if done:
                break

        returns.append(episode_return)

    if frames and use_wandb:
        video = np.transpose(np.stack(frames), (0, 3, 1, 2))
        wandb.log({"videos/eval_policy": wandb.Video(video, fps=30, format="mp4")})

    eval_env.close()
    model.train()
    return returns, frames


def estimate_values(
    model: nn.Module,
    device: Union[str, torch.device],
    env_id: str,
    seed: int = 42,
    gamma: float = 0.99,
    num_episodes: int = 3,
    atari_wrapper: bool = True,
    frame_stack: int = 4,
    screen_size: int = 84,
    clip_reward: bool = True,
) -> tuple[float, float]:
    """Compare the agent's own value estimates against realised returns.

    Runs greedy evaluation episodes and, for every state the policy visits,
    pairs the network's estimate with what actually happened from that state on:

        predicted: max_a Q(s_t, a)
        realised:  sum_{k >= t} gamma^(k-t) r_k

    Both are averaged over the same states, so with an unbiased critic the two
    numbers agree. This is the measurement of Figure 3 in van Hasselt et al.
    (2016): plain Q-learning reports more than the policy earns, because the
    same noisy estimate both picks and scores the bootstrap action.

    Comparing per-state values matters: averaging the estimate over every
    visited state and comparing it to a single per-episode ``G_0`` measures two
    different quantities and hides the bias.

    Returns ``(mean predicted value, mean realised discounted return)``.
    """
    eval_env = make_env(
        env_id,
        seed,
        idx=0,
        atari_wrapper=atari_wrapper,
        frame_stack=frame_stack,
        screen_size=screen_size,
        clip_reward=clip_reward,
    )()

    model = model.to(device)
    model.eval()
    predictions: list[float] = []
    realised: list[float] = []

    # Some envs are registered without a TimeLimit, so cap the episode length.
    spec = getattr(eval_env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) or 10_000

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_predictions: list[float] = []
        episode_rewards: list[float] = []

        for _step in range(max_steps):
            with torch.no_grad():
                q_values = model(_obs_tensor(obs, device))
            episode_predictions.append(float(q_values.max().item()))

            action = int(q_values.argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_rewards.append(float(reward))
            if done:
                break

        # Discounted return from every state, computed backwards.
        returns: list[float] = []
        running = 0.0
        for reward in reversed(episode_rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()

        predictions.extend(episode_predictions)
        realised.extend(returns)

    eval_env.close()
    model.train()

    if not predictions:
        return 0.0, 0.0
    return float(np.mean(predictions)), float(np.mean(realised))


def train_dqn_cnn(
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
    start_e: float = Config.start_e,
    end_e: float = Config.end_e,
    exploration_fraction: float = Config.exploration_fraction,
    learning_starts: int = Config.learning_starts,
    train_frequency: int = Config.train_frequency,
    max_grad_norm: float = Config.max_grad_norm,
    capture_video: bool = Config.capture_video,
    use_wandb: bool = Config.use_wandb,
    wandb_project: str = Config.wandb_project,
    wandb_entity: str = Config.wandb_entity,
    exp_name: str = Config.exp_name,
    eval_every: int = Config.eval_every,
    save_every: int = Config.save_every,
    num_eval_eps: int = Config.num_eval_eps,
    log_gradients: bool = Config.log_gradients,
    device: Union[str, torch.device] = Config.device,
    atari_wrapper: bool = Config.atari_wrapper,
    frame_stack: int = Config.frame_stack,
    screen_size: int = Config.screen_size,
    clip_reward: bool = Config.clip_reward,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    q_network_class: Any = QNetCNN,
    eval_callback: Optional[Callable[[int, dict], None]] = None,
) -> nn.Module:
    """Train a CNN-based DQN agent on Atari-style pixel observations.

    Pass either ``env_id`` or ``env``. Hyperparameters match the signature.
    ``eval_callback`` receives ``(step, metrics)`` after every evaluation, which
    is handy for plotting the overestimation gap. Returns the trained network.
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
    Config.start_e = start_e
    Config.end_e = end_e
    Config.exploration_fraction = exploration_fraction
    Config.learning_starts = learning_starts
    Config.train_frequency = train_frequency
    Config.max_grad_norm = max_grad_norm
    Config.capture_video = capture_video
    Config.use_wandb = use_wandb
    Config.wandb_project = wandb_project
    Config.wandb_entity = wandb_entity
    Config.exp_name = exp_name
    Config.eval_every = eval_every
    Config.save_every = save_every
    Config.num_eval_eps = num_eval_eps
    Config.log_gradients = log_gradients
    Config.device = device
    Config.atari_wrapper = atari_wrapper
    Config.frame_stack = frame_stack
    Config.screen_size = screen_size
    Config.clip_reward = clip_reward
    Config.env_wrapper = env_wrapper

    if env_id is not None and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium "
            "environments or env for custom environments."
        )
    if Config.capture_video and not Config.use_wandb:
        raise ValueError(
            "Cannot capture video without WandB enabled. Set use_wandb=True."
        )

    run_name = f"{Config.env_id}__{Config.exp_name}__{Config.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    os.makedirs(f"videos/{run_name}/eval", exist_ok=True)

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            entity=Config.wandb_entity,
            sync_tensorboard=False,
            config=vars(Config()),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    configure_logging(log_dir=os.path.join("runs", Config.exp_name))
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    device = setup_device(Config.device, Config.seed)  # type: ignore[assignment]

    env_thunk = make_env(
        Config.env_id or "",
        Config.seed,
        idx=0,
        render_mode="rgb_array" if Config.capture_video else None,
        atari_wrapper=Config.atari_wrapper,
        frame_stack=Config.frame_stack,
        screen_size=Config.screen_size,
        clip_reward=Config.clip_reward,
        env_wrapper=Config.env_wrapper,
        env=env,
    )
    train_env = env_thunk()

    obs_space = train_env.observation_space
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 3:
        raise ValueError(
            "DQN-CNN expects an image observation space (channels, height, "
            f"width); got {obs_space}. Use neatrl.dqn_mlp for vector states."
        )
    obs_shape = tuple(obs_space.shape)
    action_shape = int(train_env.action_space.n)  # type: ignore[attr-defined]

    if isinstance(q_network_class, nn.Module):
        validate_q_network_dimensions(q_network_class, obs_shape, action_shape)
        q_network = q_network_class.to(device)
        target_net = copy.deepcopy(q_network)
    else:
        q_network = q_network_class(obs_shape, action_shape).to(device)
        target_net = q_network_class(obs_shape, action_shape).to(device)
    target_net.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=Config.lr)
    model_params = sum(p.numel() for p in q_network.parameters())
    logger.debug("%s\n%s", "Q-Network Architecture:", q_network)
    if Config.use_wandb:
        wandb.config.update(
            {
                "network_architecture": str(q_network),
                "obs_shape": obs_shape,
                "action_shape": action_shape,
            }
        )

    q_network.train()
    target_net.train()

    replay_buffer = ReplayBuffer(
        Config.buffer_size,
        train_env.observation_space,
        train_env.action_space,
        device=device,
        handle_timeout_termination=False,
        n_envs=1,
        optimize_memory_usage=True,  # share the observation array to halve RAM
    )

    def epsilon_at(step: int) -> float:
        """Linearly anneal epsilon from start_e to end_e."""
        decay_steps = max(Config.exploration_fraction * Config.total_timesteps, 1.0)
        fraction = min(max(step / decay_steps, 0.0), 1.0)
        return Config.start_e + fraction * (Config.end_e - Config.start_e)

    obs, _ = train_env.reset()
    start_time = time.time()
    latest_loss = 0.0
    latest_avg_return = 0.0
    latest_ep_return = 0.0
    latest_gap = 0.0

    # Only show the interactive dashboard on a TTY; otherwise let logs through.
    dashboard = None
    if Dashboard is not None and sys.stdout.isatty():
        dashboard = Dashboard(
            "DQN-CNN",
            Config.env_id or "custom",
            Config.total_timesteps,
            config=Config,
        )

    for step in tqdm(range(Config.total_timesteps)):
        eps = epsilon_at(step)

        if random.random() < eps:
            action = int(train_env.action_space.sample())
        else:
            with torch.no_grad():
                q_values = q_network(_obs_tensor(obs, device))
            action = int(q_values.argmax(dim=1).item())

        new_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            obs,
            new_obs,
            np.array(action),
            np.array(reward),
            np.array(done),
            [info],
        )
        obs = new_obs if not done else train_env.reset()[0]

        if "episode" in info:
            latest_ep_return = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
            ep_len = int(np.asarray(info["episode"]["l"]).reshape(-1)[0])
            logger.info(
                "Step=%d Return=%.2f Length=%d Eps=%.3f",
                step,
                latest_ep_return,
                ep_len,
                eps,
            )
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/episodic_return": latest_ep_return,
                        "charts/episodic_length": ep_len,
                        "charts/epsilon": eps,
                        "global_step": step,
                    }
                )

        if step > Config.learning_starts and step % Config.train_frequency == 0:
            data = replay_buffer.sample(Config.batch_size)

            with torch.no_grad():
                # Plain Q-learning target: the same max is used to select and
                # to evaluate the next action.
                next_q = target_net(data.next_observations).max(dim=1)[0]
                td_target = data.rewards.flatten() + Config.gamma * next_q * (
                    1 - data.dones.flatten()
                )

            old_val = (
                q_network(data.observations).gather(1, data.actions.long()).squeeze(1)
            )
            loss = nn.functional.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()

            if Config.use_wandb and Config.log_gradients:
                for name, param in q_network.named_parameters():
                    if param.grad is not None:
                        wandb.log(
                            {
                                f"gradients/layer_{name}": float(
                                    torch.norm(param.grad.detach(), 2)
                                ),
                                "global_step": step,
                            }
                        )

            if Config.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    q_network.parameters(), max_norm=Config.max_grad_norm
                )
            optimizer.step()
            latest_loss = float(loss.item())

            if Config.use_wandb and step % 100 == 0:
                wandb.log(
                    {
                        "losses/td_loss": latest_loss,
                        "q_values/mean": float(old_val.mean()),
                        "q_values/max": float(old_val.max()),
                        "global_step": step,
                    }
                )

        if step % Config.target_network_frequency == 0:
            for q_params, target_params in zip(
                q_network.parameters(), target_net.parameters()
            ):
                target_params.data.copy_(
                    Config.tau * q_params.data + (1.0 - Config.tau) * target_params.data
                )

        if Config.eval_every > 0 and step > 0 and step % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                q_network,
                device,
                Config.env_id or "",
                seed=Config.seed,
                atari_wrapper=Config.atari_wrapper,
                frame_stack=Config.frame_stack,
                screen_size=Config.screen_size,
                clip_reward=Config.clip_reward,
                num_eval_eps=Config.num_eval_eps,
                capture_video=Config.capture_video,
                use_wandb=Config.use_wandb,
            )
            avg_return = float(np.mean(episodic_returns))
            latest_avg_return = avg_return

            mean_max_q, true_value = estimate_values(
                q_network,
                device,
                Config.env_id or "",
                seed=Config.seed,
                gamma=Config.gamma,
                num_episodes=Config.num_eval_eps,
                atari_wrapper=Config.atari_wrapper,
                frame_stack=Config.frame_stack,
                screen_size=Config.screen_size,
                clip_reward=Config.clip_reward,
            )
            latest_gap = mean_max_q - true_value

            logger.info(
                "Evaluation returns: %s Average: %.2f | mean max-Q: %.2f | "
                "realised discounted return: %.2f | gap: %.2f",
                [float(r) for r in episodic_returns],
                avg_return,
                mean_max_q,
                true_value,
                latest_gap,
            )
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/val_avg_return": avg_return,
                        "values/mean_max_q": mean_max_q,
                        "values/true_discounted_return": true_value,
                        "values/overestimation_gap": latest_gap,
                        "val_step": step,
                    }
                )
            if eval_callback is not None:
                eval_callback(
                    step,
                    {
                        "avg_return": avg_return,
                        "mean_max_q": mean_max_q,
                        "true_discounted_return": true_value,
                        "overestimation_gap": latest_gap,
                        "model": q_network,
                    },
                )

        if Config.save_every > 0 and step > 0 and step % Config.save_every == 0:
            model_path = f"runs/{run_name}/models/dqn_cnn_step_{step}.pth"
            torch.save(
                {
                    "q_network": q_network.state_dict(),
                    "target_network": target_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_path,
            )
            logger.info("Model saved at step %d to %s", step, model_path)

        if Config.use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / max(time.time() - start_time, 1e-9)),
                    "global_step": step,
                }
            )

        if dashboard:
            dashboard.push_metrics(
                epsilon=eps,
                total_loss=latest_loss,
                ep_return=latest_ep_return,
                q_gap=latest_gap,
            )
            dashboard.update(
                agent_steps=step,
                epoch=step // max(Config.train_frequency, 1),
                losses={"td_loss": latest_loss},
                eval_stats={
                    "avg_return": latest_avg_return,
                    "last_return": latest_ep_return,
                },
                message=f"Q-Net: {model_params:,} params",
            )

    if Config.use_wandb and Config.capture_video:
        _, frames = evaluate(
            q_network,
            device,
            Config.env_id or "",
            seed=Config.seed,
            atari_wrapper=Config.atari_wrapper,
            frame_stack=Config.frame_stack,
            screen_size=Config.screen_size,
            clip_reward=Config.clip_reward,
            num_eval_eps=Config.num_eval_eps,
            capture_video=True,
            use_wandb=Config.use_wandb,
        )
        if frames:
            train_video_path = f"videos/{run_name}/eval/final.mp4"
            imageio.mimsave(train_video_path, frames, fps=30)  # type: ignore[arg-type]
            logger.info("Final evaluation video saved to %s", train_video_path)

    final_path = f"runs/{run_name}/models/dqn_cnn_final.pth"
    torch.save(q_network.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    train_env.close()
    if Config.use_wandb:
        wandb.finish()
    if dashboard:
        dashboard.close(
            message=f"Q-Net: {model_params:,} params  |  "
            f"Final avg_return: {latest_avg_return:.1f}  |  "
            f"Final value gap: {latest_gap:.2f}"
        )

    return q_network


# --- Main Execution ---
if __name__ == "__main__":
    train_dqn_cnn()
