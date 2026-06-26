"""Deep Q-Network (DQN) with MLP architecture for discrete-action Gymnasium environments."""

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

import wandb

from .cli.dashboard import Dashboard
from .utils import configure_logging, get_logger, get_space_dims, setup_device
from .utils.nn_utils import (
    calculate_param_norm,
    validate_q_network_dimensions,
)
from .utils.wrappers import OneHotWrapper

logger = get_logger(__name__)

try:
    import ale_py  # type: ignore[import-untyped]

    gym.register_envs(ale_py)
except ImportError:
    pass


# ===== CONFIGURATION =====
@dataclass
class Config:
    # Experiment settings
    """Hyperparameters and settings for DQN training."""

    exp_name: str = "DQN"
    seed: int = 42
    env_id: str = "DQN-Experiment"

    # Training parameters
    total_timesteps: int = 20000
    lr: float = 2.5e-4
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 50
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 1000
    train_frequency: int = 10
    max_grad_norm: float = 0.5  # Maximum gradient norm for gradient clipping
    num_eval_eps: int = 10
    grid_env: bool = False

    eval_every: int = 1000
    save_every: int = 1000
    upload_every: int = 100
    atari_wrapper: bool = False
    n_envs: int = 1
    capture_video: bool = False
    device: Union[str, torch.device] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    custom_agent: Optional[Any] = None  # Custom neural network class or instance

    # Logging & saving
    use_wandb: bool = True
    wandb_project: str = "cleanRL"
    wandb_entity: str = ""


class QNet(nn.Module):
    """Fully-connected Q-network mapping observations to action values."""

    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.q_value = nn.Linear(512, action_space)

    def forward(self, x):
        """Forward pass — returns network output(s).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Network output(s).
        """
        return self.q_value(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class LinearEpsilonDecay(nn.Module):
    """Linearly decays epsilon from start to end over duration steps."""

    def __init__(self, initial_eps, end_eps, total_timesteps):
        super().__init__()
        self.initial_eps = initial_eps
        # self.decay_factor = decay_factor
        self.total_timesteps = total_timesteps
        self.end_eps = end_eps

    def forward(self, current_timestep, decay_factor):
        """Forward pass — returns network output(s).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Network output(s).
        """
        slope = (self.end_eps - self.initial_eps) / (
            self.total_timesteps * decay_factor
        )
        return max(slope * current_timestep + self.initial_eps, self.end_eps)


def make_env(env_id, seed, idx, atari_wrapper=False, grid_env=False):
    """Return a thunk that constructs and seeds a Gymnasium environment.

    Args:
        env_id (str): Gymnasium environment ID.
        seed (int): Base random seed; actual seed is ``seed + idx``.
        idx (int): Worker index added to the seed.
        atari_wrapper (bool): Apply Atari preprocessing wrappers when True.
        grid_env (bool): Apply one-hot observation encoding for grid environments.

    Returns:
        Callable[[], gym.Env]: A zero-argument thunk that creates the environment.
    """

    def thunk():
        """Create environment with video recording"""
        env = gym.make(env_id, render_mode="rgb_array")

        # Special handling for FrozenLake discrete states
        if grid_env:
            env = OneHotWrapper(env, obs_shape=env.observation_space.n)  # type: ignore[attr-defined]

        if atari_wrapper:
            env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
            env = FrameStackObservation(env, stack_size=4)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)

        return env

    return thunk


def evaluate(
    env_id,
    model,
    device,
    seed,
    atari_wrapper=False,
    num_eval_eps=10,
    capture_video=False,
    grid_env=False,
):
    """Run evaluation episodes using a greedy policy and return rewards and optional frames.

    Args:
        env_id (str): Gymnasium environment ID for the evaluation environment.
        model (nn.Module): The Q-network to evaluate (greedy action selection).
        device (torch.device | str): Device on which to run the network.
        seed (int): Random seed for the evaluation environment.
        atari_wrapper (bool): Apply Atari preprocessing wrappers when True.
        num_eval_eps (int): Number of episodes to run.
        capture_video (bool): Record frames when True.
        grid_env (bool): Apply one-hot encoding for grid environments.

    Returns:
        tuple[list[float], list]: A list of total rewards per episode and a list of
            recorded RGB frames (empty if capture_video is False).
    """
    eval_env = make_env(
        idx=0, env_id=env_id, seed=seed, atari_wrapper=atari_wrapper, grid_env=grid_env
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
                try:
                    frame = eval_env.render()
                    frames.append(frame)
                except Exception as e:
                    logger.error(
                        "Video capture failed for %s (%s). "
                        "Check the renderer for this environment is installed "
                        "(e.g. pip install pygame-ce for classic-control envs).",
                        type(eval_env).__name__,
                        type(e).__name__,
                    )
                    raise

            action = (
                model(
                    torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                )
                .argmax()
                .item()
            )
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)

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


def train_dqn(
    env_id=Config.env_id,
    total_timesteps=Config.total_timesteps,
    seed=Config.seed,
    lr=Config.lr,
    buffer_size=Config.buffer_size,
    gamma=Config.gamma,
    tau=Config.tau,
    target_network_frequency=Config.target_network_frequency,
    batch_size=Config.batch_size,
    start_e=Config.start_e,
    end_e=Config.end_e,
    exploration_fraction=Config.exploration_fraction,
    learning_starts=Config.learning_starts,
    train_frequency=Config.train_frequency,
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
):
    """Train a DQN agent on a Gymnasium environment.

    Args:
        env_id (str): Gymnasium environment ID (e.g. ``'CartPole-v1'``).
        total_timesteps (int): Total environment interaction steps to train for.
        seed (int): Global random seed for reproducibility.
        lr (float): Optimiser learning rate.
        buffer_size (int): Replay buffer capacity (number of transitions).
        gamma (float): Discount factor γ (0 < γ ≤ 1).
        tau (float): Soft target-network update coefficient (0 < τ ≤ 1).
        target_network_frequency (int): Sync target networks every this many steps.
        batch_size (int): Mini-batch size sampled from the replay buffer.
        start_e (float): Initial epsilon for epsilon-greedy exploration.
        end_e (float): Final epsilon after decay.
        exploration_fraction (float): Fraction of total_timesteps over which epsilon decays.
        learning_starts (int): Random-action steps before gradient updates begin.
        train_frequency (int): Update the network every this many steps.
        max_grad_norm (float): Maximum gradient-norm for clipping (0 disables clipping).
        capture_video (bool): Record evaluation episodes to video files when True.
        use_wandb (bool): Log training metrics to Weights & Biases when True.
        wandb_project (str): W&B project name (used when use_wandb=True).
        wandb_entity (str): W&B entity/username (used when use_wandb=True).
        exp_name (str): Run name used for log directories and W&B.
        eval_every (int): Evaluate every this many environment steps.
        save_every (int): Save a model checkpoint every this many steps.
        atari_wrapper (bool): Apply Atari preprocessing (grayscale, frame-stack) when True.
        custom_agent (nn.Module | None): Custom ``nn.Module`` instance replacing the default Q-network.
        num_eval_eps (int): Number of episodes per evaluation.
        n_envs (int): Number of parallel vectorised environments.
        device (str | torch.device): PyTorch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        grid_env (bool): Use one-hot observation encoding for grid environments.

    Returns:
        nn.Module: The trained network (actor / policy / Q-network) ready for inference.
    """
    Config.env_id = env_id
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

    # Initialize WandB
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            sync_tensorboard=False,
            config=locals(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if capture_video:
        os.makedirs(f"videos/{run_name}/train", exist_ok=True)
        os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    # Set seeds
    configure_logging(log_dir=os.path.join("runs", exp_name))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = setup_device(device, seed)

    if n_envs > 1:
        logger.info(f"Using {n_envs} parallel environments for experience collection.")
        env = gym.vector.SyncVectorEnv(
            [
                make_env(
                    env_id, seed, idx=i, atari_wrapper=atari_wrapper, grid_env=grid_env
                )
                for i in range(n_envs)
            ]
        )
    else:
        env = make_env(
            env_id, seed, idx=0, atari_wrapper=atari_wrapper, grid_env=grid_env
        )()

    obs_shape, action_shape = get_space_dims(env)

    # Use custom agent if provided, otherwise use default QNet
    if custom_agent is not None:
        if isinstance(custom_agent, nn.Module):
            # Validate custom agent's dimensions first
            validate_q_network_dimensions(custom_agent, obs_shape, action_shape)

            q_network = custom_agent.to(device)
            target_net = custom_agent.to(device)
        else:
            raise ValueError("agent must be an instance of nn.Module")
    else:
        q_network = QNet(obs_shape, action_shape).to(device)
        target_net = QNet(obs_shape, action_shape).to(device)

    target_net.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    eps_decay = LinearEpsilonDecay(start_e, end_e, total_timesteps)

    # Print network architecture
    logger.debug("%s\n%s", "Q-Network Architecture:", q_network)

    # Log network architecture to WandB
    if use_wandb:
        wandb.config.update({"network_architecture": str(q_network)})

    q_network.train()
    target_net.train()

    model_params = sum(p.numel() for p in q_network.parameters())

    replay_buffer = ReplayBuffer(
        buffer_size,
        env.single_observation_space if n_envs > 1 else env.observation_space,  # type: ignore[attr-defined]
        env.single_action_space if n_envs > 1 else env.action_space,  # type: ignore[attr-defined]
        device=device,
        handle_timeout_termination=False,
        n_envs=n_envs,
    )

    obs, _ = env.reset()
    start_time = time.time()
    frames = []
    latest_avg_return = 0.0
    latest_ep_return = 0.0
    update_count = 0
    latest_loss = 0.0

    dashboard = Dashboard("DQN", Config.env_id or "custom", total_timesteps, config=Config)

    for step in tqdm(range(total_timesteps), disable=True):
        step = step * n_envs
        eps = eps_decay(step, exploration_fraction)
        rnd = random.random()

        if rnd < eps:
            if n_envs > 1:
                # Sample one action per environment
                action = np.array(
                    [env.single_action_space.sample() for _ in range(n_envs)]  # type: ignore[attr-defined]
                )
            else:
                action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.tensor(obs, device=device, dtype=torch.float32)
                )

                if use_wandb:
                    # Log Q-value statistics
                    wandb.log(
                        {
                            "q_values/max": q_values.max().item(),
                            "q_values/min": q_values.min().item(),
                            "q_values/mean": q_values.mean().item(),
                            "step": step,
                        }
                    )

                action = (
                    q_values.argmax(dim=-1).cpu().numpy()
                    if n_envs > 1
                    else int(q_values.argmax(dim=-1).item())
                )

        new_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)

        replay_buffer.add(
            obs, new_obs, np.array(action), np.array(reward), np.array(done), [info]
        )

        # Log episode returns
        if "episode" in info:
            if n_envs > 1:
                for i in range(n_envs):
                    if done[i]:
                        ep_ret = info["episode"]["r"][i]
                        ep_len = info["episode"]["l"][i]
                        latest_ep_return = ep_ret

                        logger.info(
                            "Step=%d Env=%d Return=%.2f Length=%d",
                            step,
                            i,
                            ep_ret,
                            ep_len,
                        )

                        if use_wandb:
                            wandb.log(
                                {
                                    "charts/episodic_return": ep_ret,
                                    "charts/episodic_length": ep_len,
                                    "charts/global_step": step,
                                }
                            )
            else:
                if done:
                    ep_ret = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    latest_ep_return = ep_ret

                    logger.info(f"Step={step}, Return={ep_ret:.2f}, Length={ep_len}")

                    if use_wandb:
                        wandb.log(
                            {
                                "charts/episodic_return": ep_ret,
                                "charts/episodic_length": ep_len,
                                "charts/global_step": step,
                            }
                        )

        if step > learning_starts and step % train_frequency == 0:
            data = replay_buffer.sample(batch_size)

            target_max = target_net(data.next_observations).max(1)[0]
            td_target = data.rewards.flatten() + gamma * target_max * (
                1 - data.dones.flatten()
            )

            old_val = q_network(data.observations).gather(1, data.actions).squeeze()

            optimizer.zero_grad()
            loss = nn.functional.mse_loss(old_val, td_target)
            loss.backward()

            # Log gradient norm per layer
            if use_wandb:
                for name, param in q_network.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/layer_{name}": grad_norm,
                                "step": step,
                            }
                        )

            # Calculate gradient norm before clipping
            total_norm_before = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), 2)
                        for p in q_network.parameters()
                        if p.grad is not None
                    ]
                ),
                2,
            )
            # Log gradient norm
            if use_wandb:
                wandb.log(
                    {
                        "gradients/norm_before_clip": total_norm_before.item(),
                        "step": step,
                    }
                )
            # Calculate gradient norm before clipping
            if max_grad_norm != 0.0:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    q_network.parameters(), max_norm=max_grad_norm
                )

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2)
                            for p in q_network.parameters()
                            if p.grad is not None
                        ]
                    ),
                    2,
                )

                if use_wandb:
                    wandb.log(
                        {
                            "gradients/norm_after_clip": total_norm_after.item(),
                            "step": step,
                        }
                    )
                    wandb.log(
                        {
                            "gradients/clip_ratio": total_norm_after.item()
                            / (total_norm_before.item() + 1e-10),
                            "step": step,
                        }
                    )

            optimizer.step()
            update_count += 1
            latest_loss = loss.item()

            # Log loss and metrics every 100 steps
            if step % 100 == 0:
                if use_wandb:
                    wandb.log(
                        {
                            "losses/td_loss": loss.item(),
                        }
                    )

        # Update target network
        if step % target_network_frequency == 0:
            # Calculate norm of the target network parameters before update
            target_norm_before = calculate_param_norm(target_net)

            # Perform soft update of target network
            for q_params, target_params in zip(
                q_network.parameters(), target_net.parameters()
            ):
                target_params.data.copy_(
                    tau * q_params.data + (1.0 - tau) * target_params.data
                )

            # Calculate norm of the target network parameters after update
            target_norm_after = calculate_param_norm(target_net)

            # Calculate change in target network parameters
            target_norm_delta = abs(target_norm_after - target_norm_before)

            # Log target network update statistics
            if use_wandb:
                wandb.log(
                    {
                        "target_network/norm_before_update": target_norm_before,
                        "step": step,
                    }
                )
                wandb.log(
                    {
                        "target_network/norm_after_update": target_norm_after,
                        "step": step,
                    }
                )
                wandb.log(
                    {"target_network/norm_delta": target_norm_delta, "step": step}
                )
                wandb.log(
                    {
                        "target_network/update_ratio": target_norm_delta
                        / (target_norm_before + 1e-10),
                        "step": step,
                    }
                )

        # Model evaluation & saving
        if step % eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id,
                q_network,
                device,
                seed,
                num_eval_eps=num_eval_eps,
                atari_wrapper=atari_wrapper,
                capture_video=capture_video,
                grid_env=grid_env,
            )
            avg_return = np.mean(episodic_returns)
            latest_avg_return = avg_return

            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": step})
            logger.info(
                f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}"
            )

        if done.all():
            obs, _ = env.reset()
        else:
            obs = new_obs

        # ── Dashboard update (every env step) ───────────────────────
        if dashboard:
            dashboard.update(
                agent_steps=step,
                epoch=update_count,
                losses={"td_loss": latest_loss},
                eval_stats={
                    "avg_return": latest_avg_return,
                    "last_return": latest_ep_return,
                },
                message=f"Q-Net: {model_params:,} params",
            )

        if use_wandb:
            wandb.log({"step": step})

        if use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / (time.time() - start_time)),
                    "charts/step": step,
                }
            )

        if step % save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/dqn_model_step_{step}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(q_network.state_dict(), model_path)
            logger.info(f"Model saved at step {step} to {model_path}")

    # Save final video to WandB
    if use_wandb and capture_video:
        train_video_path = "videos/final.mp4"
        _, frames = evaluate(
            env_id,
            q_network,
            device,
            seed,
            atari_wrapper=atari_wrapper,
            num_eval_eps=num_eval_eps,
            capture_video=True,
            grid_env=grid_env,
        )
        if frames:
            imageio.mimsave(train_video_path, frames, fps=30)  # type: ignore[arg-type]
            logger.info(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()
    if dashboard:
        dashboard.close(
            message=f"Q-Net: {model_params:,} params  |  "
            f"Updates: {update_count}  |  "
            f"Final avg_return: {latest_avg_return:.1f}"
        )

    return q_network


if __name__ == "__main__":
    train_dqn()
