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
import wandb
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from .utils import configure_logging, get_logger, get_space_dims, setup_device
from .utils.nn_utils import (
    calculate_param_norm,
    validate_q_network_dimensions,
)

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
    learning_rate: float = 2.5e-4
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
    max_grad_norm: float = 1.0  # Maximum gradient norm for gradient clipping
    num_eval_eps: int = 10
    grid_env: bool = False

    eval_every: int = 1000
    save_every: int = 1000
    upload_every: int = 100
    atari_wrapper: bool = False
    n_envs: int = 4
    capture_video: bool = False
    device: Union[str, torch.device] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    custom_agent: Optional[Any] = None  # Custom neural network class or instance

    # Logging & saving
    use_wandb: bool = False
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
        """Forward pass — returns network output(s)."""
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
        """Forward pass — returns network output(s)."""
        slope = (self.end_eps - self.initial_eps) / (
            self.total_timesteps * decay_factor
        )
        return max(slope * current_timestep + self.initial_eps, self.end_eps)


class OneHotWrapper(gym.ObservationWrapper):
    """Wraps a discrete observation space into a one-hot float vector."""
    def __init__(self, env, obs_shape=16):
        super().__init__(env)
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs):
        """Convert discrete integer observation to one-hot float tensor."""
        one_hot = torch.zeros(self.obs_shape, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()


def make_env(env_id, seed, idx, atari_wrapper=False, grid_env=False):
    """Return a thunk that creates and seeds a gymnasium environment."""
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
    """Run eval_episodes episodes and return total rewards and any recorded frames."""
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
        # episode_frames = []

        while not done:
            if capture_video:
                frame = eval_env.render()
                frames.append(frame)

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
    learning_rate=Config.learning_rate,
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
    """
    Train a DQN agent on a Gymnasium environment.

    Args:
        env_id: Gymnasium environment ID
        total_timesteps: Total training timesteps
        seed: Random seed
        learning_rate: Learning rate for optimizer
        buffer_size: Replay buffer size
        gamma: Discount factor
        tau: Target network update rate
        target_network_frequency: How often to update target network
        batch_size: Batch size for training
        start_e: Initial epsilon for exploration
        end_e: Final epsilon for exploration
        exploration_fraction: Fraction of timesteps for epsilon decay
        learning_starts: When to start learning
        train_frequency: How often to train
        max_grad_norm: Maximum gradient norm for gradient clipping (0.0 to disable)
        capture_video: Whether to capture training videos
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/username
        exp_name: Experiment name
        eval_every: Frequency of evaluation during training
        save_every: Frequency of saving the model
        atari_wrapper: Whether to apply Atari preprocessing wrappers
        agent: Custom neural network class or instance (nn.Module subclass or instance, optional, defaults to QNet)
        num_eval_eps: Number of evaluation episodes
        n_envs : Number of parallel environments for the replay buffer
        capture_video: Whether to record evaluation videos
        device: Device to use for training (e.g., "cpu", "cuda")
        grid_env: Whether the environment uses discrete grid observations
    Returns:
        Trained Q-network model
    """
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
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    eps_decay = LinearEpsilonDecay(start_e, end_e, total_timesteps)

    # Print network architecture
    logger.debug("%s\n%s", "Q-Network Architecture:", q_network)

    # Log network architecture to WandB
    if use_wandb:
        wandb.config.update({"network_architecture": str(q_network)})

    q_network.train()
    target_net.train()

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

    for step in tqdm(range(total_timesteps)):
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

                        logger.info("Step=%d Env=%d Return=%.2f Length=%d", step, i, ep_ret, ep_len)

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

            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": step})
            logger.info(
                f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}"
            )

        if done.all():
            obs, _ = env.reset()
        else:
            obs = new_obs

        # Print progress every 1000 steps
        if step % 10 == 0:
            logger.debug("Step %d TD Loss: %.4f SPS: %d", step, loss.item(), int(step / (time.time() - start_time)))

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
    if use_wandb:
        train_video_path = "videos/final.mp4"
        _, frames = evaluate(
            env_id,
            q_network,
            device,
            seed,
            atari_wrapper=atari_wrapper,
            num_eval_eps=num_eval_eps,
            capture_video=capture_video,
            grid_env=grid_env,
        )
        imageio.mimsave(train_video_path, frames, fps=30)  # type: ignore[arg-type]
        logger.info(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()

    return q_network


if __name__ == "__main__":
    train_dqn()
