"""Double DQN with MLP architecture for discrete-action Gymnasium environments.

Identical to :mod:`neatrl.dqn_mlp` except for the TD target. Van Hasselt et al.
(2016) split the ``max`` into action selection and evaluation so a single noisy
estimate cannot both pick the action and score it:

    Y = R + gamma * Q_target(s', argmax_a Q_online(s', a))

This removes the systematic upward bias of the plain Q-learning target.
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

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
from .dqn_mlp import LinearEpsilonDecay, QNet, evaluate, make_env
from .utils import configure_logging, get_logger, get_space_dims, setup_device
from .utils.nn_utils import calculate_param_norm, validate_q_network_dimensions

logger = get_logger(__name__)

try:
    import ale_py  # type: ignore[import-untyped]

    gym.register_envs(ale_py)
except ImportError:
    pass


# ===== CONFIGURATION =====
@dataclass
class Config:
    """Hyperparameters and settings for Double DQN training."""

    # Experiment settings
    exp_name: str = "DoubleDQN"
    seed: int = 42
    env_id: str = "CartPole-v1"

    # Training parameters
    total_timesteps: int = 50_000
    lr: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 -> hard target copies
    target_network_frequency: int = 50
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 1_000
    train_frequency: int = 10
    max_grad_norm: float = 0.5
    num_eval_eps: int = 10
    grid_env: bool = False

    eval_every: int = 1_000
    save_every: int = 10_000
    atari_wrapper: bool = False
    n_envs: int = 1
    capture_video: bool = False
    custom_agent: Optional[Any] = None  # Custom Q-network class or instance

    # Logging & saving
    use_wandb: bool = True
    wandb_project: str = "cleanRL"
    wandb_entity: str = ""
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


def train_double_dqn(
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
    n_envs: int = Config.n_envs,
    grid_env: bool = Config.grid_env,
    atari_wrapper: bool = Config.atari_wrapper,
    custom_agent: Optional[Any] = Config.custom_agent,
    device: Union[str, torch.device] = Config.device,
    eval_callback: Optional[Any] = None,
) -> nn.Module:
    """Train a Double DQN agent on a discrete-action Gymnasium environment.

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
    Config.n_envs = n_envs
    Config.grid_env = grid_env
    Config.atari_wrapper = atari_wrapper
    Config.custom_agent = custom_agent
    Config.device = device

    if env_id is not None and env is not None:
        raise ValueError(
            "Cannot provide both env_id and env. Use env_id for Gymnasium "
            "environments or env for custom environments."
        )
    if Config.n_envs != 1:
        raise ValueError(
            "train_double_dqn currently supports a single environment "
            f"(n_envs=1); got n_envs={Config.n_envs}."
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

    if env is not None:
        train_env = env
    else:
        train_env = make_env(
            Config.env_id,
            Config.seed,
            idx=0,
            atari_wrapper=Config.atari_wrapper,
            grid_env=Config.grid_env,
        )()

    if not isinstance(train_env.action_space, gym.spaces.Discrete):
        raise ValueError(
            "Double DQN expects a discrete action space; got "
            f"{train_env.action_space}. Use neatrl.td3 / neatrl.sac for "
            "continuous control."
        )

    obs_shape, action_shape = get_space_dims(train_env)

    if custom_agent is not None:
        if not isinstance(custom_agent, nn.Module):
            raise ValueError("custom_agent must be an instance of nn.Module")
        validate_q_network_dimensions(custom_agent, obs_shape, action_shape)
        q_network = custom_agent.to(device)
        target_net = copy.deepcopy(q_network)
    else:
        q_network = QNet(obs_shape, action_shape).to(device)
        target_net = QNet(obs_shape, action_shape).to(device)

    target_net.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=Config.lr)
    eps_decay = LinearEpsilonDecay(Config.start_e, Config.end_e, Config.total_timesteps)

    model_params = sum(p.numel() for p in q_network.parameters())
    logger.debug("%s\n%s", "Q-Network Architecture:", q_network)
    if Config.use_wandb:
        wandb.config.update(
            {
                "network_architecture": str(q_network),
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
    )

    obs, _ = train_env.reset(seed=Config.seed)
    start_time = time.time()
    latest_loss = 0.0
    latest_avg_return = 0.0
    latest_ep_return = 0.0
    update_count = 0

    # Only show the interactive dashboard on a TTY; otherwise let logs through.
    dashboard = None
    if sys.stdout.isatty():
        dashboard = Dashboard(
            "DoubleDQN",
            Config.env_id or "custom",
            Config.total_timesteps,
            config=Config,
        )

    for step in tqdm(range(Config.total_timesteps)):
        eps = float(eps_decay(step, Config.exploration_fraction))

        if random.random() < eps:
            action = int(train_env.action_space.sample())
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(
                        0
                    )
                )
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

        if "episode" in info and done:
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

            if Config.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    q_network.parameters(), max_norm=Config.max_grad_norm
                )
            optimizer.step()
            update_count += 1
            latest_loss = float(loss.item())

        if step % Config.target_network_frequency == 0:
            norm_before = calculate_param_norm(target_net)
            for q_params, target_params in zip(
                q_network.parameters(), target_net.parameters()
            ):
                target_params.data.copy_(
                    Config.tau * q_params.data + (1.0 - Config.tau) * target_params.data
                )
            if Config.use_wandb:
                wandb.log(
                    {
                        "target_network/norm_before_update": norm_before,
                        "target_network/norm_after_update": calculate_param_norm(
                            target_net
                        ),
                        "global_step": step,
                    }
                )

        if Config.eval_every > 0 and step > 0 and step % Config.eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id=Config.env_id,
                model=q_network,
                device=device,
                seed=Config.seed,
                atari_wrapper=Config.atari_wrapper,
                num_eval_eps=Config.num_eval_eps,
                capture_video=Config.capture_video,
                grid_env=Config.grid_env,
            )
            avg_return = float(np.mean(episodic_returns))
            latest_avg_return = avg_return
            logger.info(
                "Evaluation returns: %s Average: %.2f",
                [float(r) for r in episodic_returns],
                avg_return,
            )
            if Config.use_wandb:
                wandb.log(
                    {
                        "charts/val_avg_return": avg_return,
                        "val_step": step,
                    }
                )
            if eval_callback is not None:
                eval_callback(
                    step, {"avg_return": float(avg_return), "model": q_network}
                )

        if Config.save_every > 0 and step > 0 and step % Config.save_every == 0:
            model_path = f"runs/{run_name}/models/double_dqn_step_{step}.pth"
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
            )
            dashboard.update(
                agent_steps=step,
                epoch=update_count,
                losses={"td_loss": latest_loss},
                eval_stats={
                    "avg_return": latest_avg_return,
                    "last_return": latest_ep_return,
                },
                message=f"Q-Net: {model_params:,} params | Double DQN",
            )

    if Config.use_wandb and Config.capture_video:
        _, frames = evaluate(
            env_id=Config.env_id,
            model=q_network,
            device=device,
            seed=Config.seed,
            atari_wrapper=Config.atari_wrapper,
            num_eval_eps=Config.num_eval_eps,
            capture_video=True,
            grid_env=Config.grid_env,
        )
        if frames:
            video_path = f"videos/{run_name}/eval/final.mp4"
            imageio.mimsave(video_path, frames, fps=30)  # type: ignore[arg-type]
            logger.info("Final evaluation video saved to %s", video_path)

    final_path = f"runs/{run_name}/models/double_dqn_final.pth"
    torch.save(q_network.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    train_env.close()
    if Config.use_wandb:
        wandb.finish()
    if dashboard:
        dashboard.close(
            message=f"Q-Net: {model_params:,} params  |  "
            f"Final avg_return: {latest_avg_return:.1f}"
        )

    return q_network


# --- Main Execution ---
if __name__ == "__main__":
    train_double_dqn()
