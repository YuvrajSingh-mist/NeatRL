"""Double DQN with a CNN for Atari-style pixel observations.

Same environment preprocessing, network and hyperparameters as
:mod:`neatrl.dqn_cnn`; only the TD target changes. Van Hasselt et al. (2016)
decompose the ``max`` into action *selection* and action *evaluation*:

    Y = R + gamma * Q_target(s', argmax_a Q_online(s', a))

Selection still follows the online network (so we keep learning the value of
the greedy policy), while the target network supplies the value, which removes
the upward bias a plain ``max`` inherits from noisy estimates.
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
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

import wandb

from .cli.dashboard import Dashboard
from .dqn_cnn import (
    QNetCNN,
    _obs_tensor,
    estimate_values,
    evaluate,
    make_env,
)
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
    """Hyperparameters and settings for Atari Double DQN training.

    Mirrors :class:`neatrl.dqn_cnn.Config` on purpose: the two algorithms must
    differ only in the TD target for a fair overestimation comparison.
    """

    # Experiment settings
    exp_name: str = "DoubleDQN-CNN"
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


def double_dqn_target(
    q_network: nn.Module,
    target_network: nn.Module,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Bootstrap target R + gamma * Q_target(s', argmax_a Q_online(s', a)).

    Selection uses the online network, evaluation uses the target network, so a
    single noisy estimate can no longer both pick and score the next action.
    Terminal transitions are left with ``rewards`` only.
    """
    with torch.no_grad():
        next_actions = q_network(next_observations).argmax(dim=1, keepdim=True)
        next_q = target_network(next_observations).gather(1, next_actions).squeeze(1)
    return rewards.flatten() + gamma * next_q * (1 - dones.flatten())


def train_double_dqn_cnn(
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
    """Train a CNN-based Double DQN agent on Atari-style pixel observations.

    Pass either ``env_id`` or ``env``. Hyperparameters match the signature.
    ``eval_callback`` receives ``(step, metrics)`` after every evaluation.
    Returns the trained network ready for inference.
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
            "Double DQN-CNN expects an image observation space (channels, "
            f"height, width); got {obs_space}. Use "
            "neatrl.double_dqn_mlp for vector states."
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
                "algorithm": "double_dqn",
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
    if sys.stdout.isatty():
        dashboard = Dashboard(
            "DoubleDQN-CNN",
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

            # Double DQN target: select with the online net, evaluate with the
            # target net.
            td_target = double_dqn_target(
                q_network,
                target_net,
                data.next_observations,
                data.rewards,
                data.dones,
                Config.gamma,
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
            model_path = f"runs/{run_name}/models/double_dqn_cnn_step_{step}.pth"
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
                message=f"Q-Net: {model_params:,} params | Double DQN",
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

    final_path = f"runs/{run_name}/models/double_dqn_cnn_final.pth"
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
    train_double_dqn_cnn()
