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
from tqdm import tqdm

import wandb

gym.register_envs(ale_py)


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "REINFORCE"
    seed = 42
    env_id = "CartPole-v1"

    # Training parameters
    episodes = 2000
    learning_rate = 2.5e-4
    gamma = 0.99
    max_grad_norm = 1.0  # Maximum gradient norm for gradient clipping
    num_eval_eps = 10
    grid_env = False
    use_entropy = False
    entropy_coeff = 0.01
    # Evaluation & logging
    eval_every = 100
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
    buffer_size = 10000
    tau = 1.0
    target_network_frequency = 50
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 10


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

    def get_action(self, x, iseval=False):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(
            action_probs
        )  # Create a categorical distribution from the probabilities
        if iseval:
            action = torch.argmax(action_probs, dim=-1)
            return action
        else:
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
    num_eval_eps=10,
    capture_video=False,
    atari_wrapper=False,
    grid_env=False,
):
    eval_env = make_env(
        env_id, seed, idx=0, atari_wrapper=atari_wrapper, grid_env=grid_env
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
                    torch.tensor(obs, device=device, dtype=torch.float32), iseval=True
                )
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
    env_id=Config.env_id,
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
):
    """
    Train a REINFORCE agent on a Gymnasium environment.

    Args:
        env_id: Gymnasium environment ID
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
    Returns:
        Trained Policy-network model
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

    # Warn if entropy_coeff is set but use_entropy is False
    if not use_entropy and entropy_coeff != 0.0:
        print(
            f"Warning: entropy_coeff={entropy_coeff} is provided but use_entropy=False. Entropy regularization will not be applied."
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

    # Determine if we're dealing with discrete observation spaces
    if n_envs > 1:
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
    if n_envs > 1:
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
    if custom_agent is not None:
        if isinstance(custom_agent, nn.Module):
            # Validate custom agent's dimensions first
            validate_policy_network_dimensions(custom_agent, obs_shape, action_shape)

            policy_network = custom_agent.to(device)
        else:
            raise ValueError("agent must be an instance of nn.Module")
    else:
        policy_network = PolicyNet(obs_shape, action_shape).to(device)

    optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

    # Print network architecture
    print("Policy-Network Architecture:")
    print(policy_network)

    # Log network architecture to WandB
    if use_wandb:
        wandb.config.update({"network_architecture": str(policy_network)})

    policy_network.train()

    start_time = time.time()

    for step in tqdm(range(total_steps)):
        step = step * n_envs
        obs, _ = env.reset()
        rewards = []
        log_probs = []
        entropies = []
        done = False

        while True:
            result = policy_network.get_action(
                torch.tensor(obs, device=device, dtype=torch.float32)
            )
            if len(result) == 2:
                action, log_prob = result
                dist = None
            elif len(result) == 3:
                action, log_prob, dist = result
            else:
                raise ValueError(
                    f"Error unpacking result from get_action. Expected 3 got {len(result)}"
                )

            if use_entropy and dist is None:
                raise ValueError(
                    "use_entropy is True but get_action did not return dist"
                )

            # Handle both discrete and continuous action spaces
            if n_envs > 1:
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

            log_probs = log_probs.sum(dim=-1) if n_envs > 1 else log_probs

            log_probs.append(log_prob)
            if use_entropy:
                entropies.append(dist.entropy())

            if use_wandb:
                wandb.log(
                    {
                        "charts/action_mean": np.mean(action),
                        "charts/action_std": np.std(action),
                        "step": step,
                    }
                )

                if n_envs > 1:
                    reward = np.array(reward)
                    wandb.log(
                        {
                            "rewards/reward_mean": reward.mean(),
                            "rewards/reward_std": np.std(reward),
                            "step": step,
                        }
                    )
                else:
                    wandb.log({"rewards/reward": reward, "step": step})

                if dist is not None:
                    wandb.log(
                        {
                            "charts/dist_mean": dist.mean.mean().item(),
                            "charts/dist_std": dist.stddev.mean().item(),
                            "step": step,
                        }
                    )

            done = np.logical_or(terminated, truncated)
            obs = new_obs
            if done.all():
                break

        # Calculate returns
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=device, dtype=torch.float32).detach()

        if use_wandb:
            wandb.log({"charts/returns_mean": returns.mean().item(), "step": step})

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            if use_wandb:
                wandb.log(
                    {
                        "charts/log_prob": log_prob.mean().item(),
                        "charts/log_probs_std": log_prob.std().item(),
                        "step": step,
                    }
                )

            policy_loss.append(-log_prob * R)  # Negative for gradient ascent

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        if use_entropy:
            entropy_loss = torch.stack(entropies).mean() * entropy_coeff

            if use_wandb:
                wandb.log({"charts/entropy_loss": entropy_loss.item(), "step": step})

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
        if use_wandb:
            for name, param in policy_network.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.detach(), 2).item()
                    wandb.log(
                        {
                            f"gradients/layer_{name}": grad_norm,
                            "step": step,
                        }
                    )

        # Log gradient norm
        if use_wandb:
            wandb.log(
                {
                    "gradients/norm_before_clip": total_norm_before.item(),
                    "step": step,
                }
            )
            # Apply gradient clipping
            if max_grad_norm != 0.0:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    policy_network.parameters(), max_norm=max_grad_norm
                )

        optimizer.step()

        # Log loss and metrics every 100 episodes
        if step % 100 == 0:
            if use_wandb:
                wandb.log(
                    {
                        "losses/policy_loss": loss.item(),
                        "step": step,
                    }
                )

        # Print progress every 1000 steps
        if step % 10 == 0:
            print(
                f"Step {step}, Policy Loss: {loss.item():.4f}, SPS: {int(step / (time.time() - start_time))}"
            )
            if use_wandb:
                wandb.log(
                    {"charts/SPS": int(step / (time.time() - start_time)), "step": step}
                )

        # Model evaluation & saving
        if step % eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id,
                policy_network,
                device,
                seed,
                num_eval_eps=num_eval_eps,
                capture_video=capture_video,
                atari_wrapper=atari_wrapper,
                grid_env=grid_env,
            )
            avg_return = np.mean(episodic_returns)

            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": step})
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        print("SPS: ", int(step / (time.time() - start_time + 1e-8)), end="\r")

        if use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / (time.time() - start_time + 1e-8)),
                    "charts/episode": step,
                }
            )

        if step % save_every == 0 and step > 0:
            model_path = f"runs/{run_name}/models/reinforce_model_episode_{step}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(policy_network.state_dict(), model_path)
            print(f"Model saved at episode {step} to {model_path}")
    # Save final video to WandB
    if use_wandb:
        train_video_path = "videos/final.mp4"
        _, frames = evaluate(
            env_id,
            policy_network,
            device,
            seed,
            num_eval_eps=num_eval_eps,
            capture_video=capture_video,
            atari_wrapper=atari_wrapper,
            grid_env=grid_env,
        )
        imageio.mimsave(train_video_path, frames, fps=30)
        print(f"Final training video saved to {train_video_path}")
        wandb.finish()

    env.close()

    return policy_network


if __name__ == "__main__":
    train_reinforce()
