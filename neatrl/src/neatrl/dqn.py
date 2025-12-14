import os
import random
import time

import ale_py
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

gym.register_envs(ale_py)


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "DQN"
    seed = 42
    env_id = "BreakoutNoFrameskip-v4"

    # Training parameters
    total_timesteps = 20000
    learning_rate = 2.5e-4
    buffer_size = 10000
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 50
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 10
    num_eval_eps = 10
    grid_env = False

    eval_every = 1000
    save_every = 1000
    upload_every = 100
    atari_wrapper = False
    n_envs = 4
    capture_video = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Custom agent
    custom_agent = None  # Custom neural network class or instance

    # Logging & saving
    use_wandb = False
    wandb_project = "cleanRL"
    wandb_entity = ""


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.q_value = nn.Linear(512, action_space)

    def forward(self, x):
        return self.q_value(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class LinearEpsilonDecay(nn.Module):
    def __init__(self, initial_eps, end_eps, total_timesteps):
        super().__init__()
        self.initial_eps = initial_eps
        # self.decay_factor = decay_factor
        self.total_timesteps = total_timesteps
        self.end_eps = end_eps

    def forward(self, current_timestep, decay_factor):
        slope = (self.end_eps - self.initial_eps) / (
            self.total_timesteps * decay_factor
        )
        return max(slope * current_timestep + self.initial_eps, self.end_eps)


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_shape=16):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs):
        one_hot = torch.zeros(16, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()


def make_env(env_id, seed, idx, atari_wrapper=False, grid_env=False):
    def thunk():
        """Create environment with video recording"""
        env = gym.make(env_id, render_mode="rgb_array")

        # Special handling for FrozenLake discrete states
        if grid_env:
            env = OneHotWrapper(env, obs_shape=env.observation_space.n)

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
    model.train()
    eval_env.close()
    return returns, frames


def validate_q_network_dimensions(q_network, obs_dim, action_dim):
    """
    Validate that the Q-network's input and output dimensions match the environment.

    Args:
        q_network: The neural network model (nn.Module)
        obs_dim: Expected observation dimension
        action_dim: Expected action dimension
    """
    # Find first Linear layer for input dimension
    first_layer = None
    for module in q_network.modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
    if first_layer is None:
        raise ValueError(
            "Q-network must have at least one Linear layer for dimension validation."
        )
    if first_layer.in_features != obs_dim:
        raise ValueError(
            f"Q-network input dimension {first_layer.in_features} does not match observation dimension {obs_dim}."
        )

    # Find last Linear layer for output dimension
    last_layer = None
    for module in reversed(list(q_network.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise ValueError(
            "Q-network must have at least one Linear layer for dimension validation."
        )
    if last_layer.out_features != action_dim:
        raise ValueError(
            f"Q-network output dimension {last_layer.out_features} does not match action dimension {action_dim}."
        )


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setting up the device
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, falling back to CPU")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    elif device.type == "mps":
        torch.mps.manual_seed(seed)

    if n_envs > 1:
        print(f"Using {n_envs} parallel environments for experience collection.")
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

    # Compute observation and action dimensions
    obs_shape = (
        env.single_observation_space.shape[0]
        if n_envs > 1
        else env.observation_space.shape[0]
    )
    action_shape = env.single_action_space.n if n_envs > 1 else env.action_space.n

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
    print("Q-Network Architecture:")
    print(q_network)

    q_network.train()
    target_net.train()

    replay_buffer = ReplayBuffer(
        buffer_size,
        env.single_observation_space if n_envs > 1 else env.observation_space,
        env.single_action_space if n_envs > 1 else env.action_space,
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
                    [env.single_action_space.sample() for _ in range(n_envs)]
                )
            else:
                action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.tensor(obs, device=device, dtype=torch.float32)
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

                        print(
                            f"Step={step}, Env={i}, Return={ep_ret:.2f}, Length={ep_len}"
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

                    print(f"Step={step}, Return={ep_ret:.2f}, Length={ep_len}")

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
            optimizer.step()

            # Log loss and metrics every 100 steps
            if step % 100 == 0:
                if use_wandb:
                    wandb.log(
                        {
                            "losses/td_loss": loss.item(),
                        }
                    )

                    # Upload video to wandb if video recording is enabled

        # Update target network
        if step % target_network_frequency == 0:
            for q_params, target_params in zip(
                q_network.parameters(), target_net.parameters()
            ):
                target_params.data.copy_(
                    tau * q_params.data + (1.0 - tau) * target_params.data
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
                record=capture_video,
                grid_env=grid_env,
            )
            avg_return = np.mean(episodic_returns)

            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": step})
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if done.all():
            obs, _ = env.reset()
        else:
            obs = new_obs

        print("SPS: ", int(step / (time.time() - start_time)), end="\r")

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
            print(f"Model saved at step {step} to {model_path}")

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
            capture_video=True,
            grid_env=grid_env,
        )
        imageio.mimsave(train_video_path, frames, fps=30)
        print(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()

    return q_network


if __name__ == "__main__":
    train_dqn()
