from typing import Any, Optional, Callable, Tuple, List, Union
import os
import random
import time
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.nn.functional as F
import imageio
# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-RND-Vectorized-ClipWalking"
    seed = 42
    env_id = "CliffWalking-v0"
    total_timesteps = 5_000_000

    # PPO & Agent settings
    lr = 3e-4
    ext_gamma = 0.99
    int_gamma = 0.95
    n_envs = 1
    max_steps = 128
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.2
    ENTROPY_COEFF = 0.01
    VALUE_COEFF = 0.5
    EXT_COEFF = 2.0  # Weight for extrinsic advantage
    INT_COEFF = 1.0  # Weight for intrinsic advantage
    num_eval_episodes = 5
    anneal_lr = True  # Whether to anneal learning rate over time
    max_grad_norm = 0.0  # Maximum gradient norm for gradient clipping (0.0 to disable)
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"
    grid_env = True # Whether the environment uses discrete grid observations
    eval_every = 10000  # Frequency of evaluation during training
    save_every = 10000  # Frequency of saving the model
    atari_wrapper = False   
    log_gradients = False  # Whether to log gradient norms to W&B
   
    
# ===== RunningMeanStd for Normalization =====
class RunningMeanStd:
    """Tracks running mean and std of a data stream using Welford's algorithm."""
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count



# --- Networks ---
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNet(nn.Module):
    def __init__(self, state_space: Union[int, Tuple[int, ...]], action_space: int, discrete: bool = True) -> None:
        super(ActorNet, self).__init__()
        self.discrete = discrete
        self.fc1 = layer_init(nn.Linear(state_space, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 128))
        
        self.out = layer_init(nn.Linear(128, action_space))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
       
        logits = self.out(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist

    def get_action(self, x: torch.Tensor, action: Optional[torch.Tensor] = None, iseval: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(x)
        if action is None:
            action = dist.sample()
        
        if iseval:
            action = torch.argmax(dist.probs, dim=-1)
            
        log_prob = dist.log_prob(action)
        
        entropy = dist.entropy()
        return action, log_prob, entropy

class CriticNet(nn.Module):
    def __init__(self, state_space: Union[int, Tuple[int, ...]]) -> None:
        super(CriticNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 128))
        self.value_ext = layer_init(nn.Linear(128, 1))
        self.value_int = layer_init(nn.Linear(128, 1))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_ext(x), self.value_int(x)

class PredictorNet(nn.Module):
    def __init__(self, state_space: Union[int, Tuple[int, ...]]) -> None:
        super(PredictorNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.out = layer_init(nn.Linear(256, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class TargetNet(nn.Module):
    def __init__(self, state_space: Union[int, Tuple[int, ...]]) -> None:
        super(TargetNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.out = layer_init(nn.Linear(256, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_shape: int = 16) -> None:
        super().__init__(env)
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 1, (obs_shape,), dtype=np.float32)

    def observation(self, obs: Any) -> np.ndarray:
        one_hot = torch.zeros(self.obs_shape, dtype=torch.float32)
        one_hot[obs] = 1.0
        return one_hot.numpy()


def make_env(env_id: str, seed: int, idx: int, render_mode: Optional[str] = None, grid_env: bool = False, atari_wrapper: bool = False, env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        if grid_env:
            env = OneHotWrapper(env, obs_shape=env.observation_space.n)
        if atari_wrapper:
            env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        if env_wrapper:
            env = env_wrapper(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk



def calculate_param_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of all parameters in a model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def validate_policy_network_dimensions(policy_network: nn.Module, obs_dim: Union[int, Tuple[int, ...]], action_dim: int) -> None:
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


def evaluate(model: nn.Module, device: torch.device, env_id: str, seed: int, run_name: str, num_eval_eps: int = 5, record: bool = False, render_mode: Optional[str] = None, display: bool = False, grid_env: bool = False, atari_wrapper: bool = False, env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None, use_wandb: bool = False) -> Tuple[List[float], List[np.ndarray]]:
    eval_env = make_env(env_id, seed, idx=0, render_mode=render_mode, grid_env=grid_env, atari_wrapper=atari_wrapper, env_wrapper=env_wrapper)()
    model.eval()
    returns = []
    frames = []

    for _ in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        rewards = 0.0
        while not done:
            if record:
                frames.append(eval_env.render())
            if display:
                safe_display(obs)
            with torch.no_grad():
                obs = torch.tensor(obs, device=device, dtype=torch.float32)
                action, _, _ = model.get_action(obs.unsqueeze(0), iseval=True)
                action = action.cpu().numpy()
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action = action.item()
                else:
                    action = action.squeeze()
            obs, rewards_curr, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            rewards += rewards_curr
       
            
        returns.append(rewards)
    eval_env.close()
    model.train()
    
    if record and len(frames) > 0:
        video_path = f"videos/eval_{run_name}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30, codec='libx264')
        print(f"Evaluation video saved to {video_path}")
        if use_wandb:
            wandb.log({"eval/video": wandb.Video(video_path, fps=30, format="mp4")})
    
    return returns, frames

def train_rnd(
    env_id: str = Config.env_id,
    total_timesteps: int = Config.total_timesteps,
    seed: int = Config.seed,
    lr: float = Config.lr,
    ext_gamma: float = Config.ext_gamma,
    int_gamma: float = Config.int_gamma,
    n_envs: int = Config.n_envs,
    max_steps: int = Config.max_steps,
    PPO_EPOCHS: int = Config.PPO_EPOCHS,
    clip_value: float = Config.clip_value,
    ENTROPY_COEFF: float = Config.ENTROPY_COEFF,
    VALUE_COEFF: float = Config.VALUE_COEFF,
    EXT_COEFF: float = Config.EXT_COEFF,
    INT_COEFF: float = Config.INT_COEFF,
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
    predictor_class: Any = PredictorNet,
    target_class: Any = TargetNet,
    num_minibatches: int = Config.num_minibatches,
    log_gradients: bool = Config.log_gradients,
    num_eval_episodes: int = Config.num_eval_episodes,
    anneal_lr: bool = Config.anneal_lr
) -> nn.Module:
    """
    Train a PPO agent with RND intrinsic rewards on a Gymnasium environment.

    Args:
        env_id: Gymnasium environment ID
        total_timesteps: Total number of timesteps to train
        seed: Random seed
        lr: Learning rate
        ext_gamma: Discount factor for extrinsic rewards
        int_gamma: Discount factor for intrinsic rewards
        n_envs: Number of parallel environments
        max_steps: Maximum steps per rollout
        num_minibatches: Number of minibatches
        PPO_EPOCHS: Number of PPO epochs
        clip_value: PPO clipping value
        ENTROPY_COEFF: Entropy coefficient
        VALUE_COEFF: Value loss coefficient
        EXT_COEFF: Extrinsic advantage coefficient
        INT_COEFF: Intrinsic advantage coefficient
        capture_video: Whether to capture video
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        exp_name: Experiment name
        grid_env: Whether the environment uses discrete grid observations
        max_grad_norm: Maximum gradient norm for gradient clipping (0.0 to disable)
        eval_every: Frequency of evaluation during training
        save_every: Frequency of saving the model
        atari_wrapper: Whether to apply Atari preprocessing and frame stacking
        env_wrapper: Custom environment wrapper function (takes env, returns wrapped env)
        actor_class: Custom actor network class
        critic_class: Custom critic network class
        predictor_class: Custom predictor network class
        target_class: Custom target network class
        log_gradients: Whether to log gradient norms to W&B
        num_eval_episodes: Number of evaluation episodes
        anneal_lr: Whether to anneal learning rate over time
        
    Returns:
        Trained actor network
    """
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)

    if use_wandb:
        wandb.init(project=wandb_project, config=vars(Config()), name=run_name, monitor_gym=True)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # env = gym.make(args.env_id, render_mode=None)
    # print("Obs: ", env.observation_space)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, grid_env=grid_env, atari_wrapper=atari_wrapper, env_wrapper=env_wrapper) for i in range(n_envs)]
    )
    if isinstance(envs.single_observation_space, gym.spaces.Discrete):
        obs_space_shape = envs.single_observation_space.n
    else:
        obs_space_shape = envs.single_observation_space.shape[0]
    
    
    action_space_n = envs.single_action_space.n if isinstance(envs.single_action_space, gym.spaces.Discrete) else envs.single_action_space.shape[0]
    
    print(f"Observation Space: {obs_space_shape}, Action Space: {action_space_n}")
    
    actor_network = actor_class(obs_space_shape, action_space_n).to(device)
    critic_network = critic_class(obs_space_shape).to(device)
    predictor_network = predictor_class(obs_space_shape).to(device)
    target_network = target_class(obs_space_shape).to(device)
    
    print("Actor Network Architecture:")
    print(actor_network)
    print("\nCritic Network Architecture:")
    print(critic_network)
    print("\nPredictor Network Architecture:")
    print(predictor_network)
    print("\nTarget Network Architecture:")
    print(target_network)
    
    # Compute derived values from passed parameters
    batch_size = n_envs * max_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size
    
    optimizer = optim.Adam(
        list(actor_network.parameters()) + list(critic_network.parameters()) + list(predictor_network.parameters()), 
        lr=lr, eps=1e-5
    )

    for param in target_network.parameters():
        param.requires_grad = False

    # Initialize running mean/std for intrinsic reward normalization
    intrinsic_reward_rms = RunningMeanStd(shape=())

    # Tensor Storage
    # action_space_naction_shape = action_space_n
    obs_storage = torch.zeros((max_steps, n_envs) + (obs_space_shape,)).to(device)
    actions_storage = torch.zeros((max_steps, n_envs) + (action_space_n,)).to(device)
    logprobs_storage = torch.zeros((max_steps, n_envs)).to(device)
    rewards_storage = torch.zeros((max_steps, n_envs)).to(device)
    intrinsic_rewards_storage = torch.zeros((max_steps, n_envs)).to(device)
    dones_storage = torch.zeros((max_steps, n_envs)).to(device)
    ext_values_storage = torch.zeros((max_steps, n_envs)).to(device)
    int_values_storage = torch.zeros((max_steps, n_envs)).to(device)
    
    global_step = 0
    
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(n_envs).to(device)

    start_time = time.time()

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow
            
        # Rollout Phase
        for step in range(0, max_steps):
            global_step = (update - 1) * batch_size + step * n_envs
            obs_storage[step] = next_obs
         
            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_obs)
                ext_value, int_value = critic_network(next_obs)
            
     
            ext_values_storage[step] = ext_value.flatten()
            int_values_storage[step] = int_value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob.sum(dim=-1) if len(logprob.shape) > 1 else logprob
            
            # --- RND Intrinsic Reward Calculation ---
            with torch.no_grad():
                pred_features = predictor_network(next_obs)
                target_features = target_network(next_obs)
                intrinsic_reward = torch.nn.functional.mse_loss(pred_features, target_features, reduction='none').mean(dim=1)
            
            # Update running statistics and normalize intrinsic rewards
            intrinsic_reward_np = intrinsic_reward.cpu().numpy()
            intrinsic_reward_rms.update(intrinsic_reward_np)
            normalized_intrinsic_reward = intrinsic_reward / np.sqrt(intrinsic_reward_rms.var + 1e-8)
            # print("intrinsic_reward, normalized_intrinsic_reward", intrinsic_reward, normalized_intrinsic_reward)
            intrinsic_rewards_storage[step] = normalized_intrinsic_reward
            
            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
         
            next_obs = torch.Tensor(new_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            wandb.log({
                "train/global_step": global_step,
                "rewards/intrinsic_reward_raw": intrinsic_reward.mean().item(),
                "rewards/intrinsic_reward_normalized": normalized_intrinsic_reward.mean().item(),
                "rewards/intrinsic_reward_std": np.sqrt(intrinsic_reward_rms.var),
                "rewards/extrinsic_reward": rewards_storage[step].mean().item(),
                "rewards/total_reward_ext_int": normalized_intrinsic_reward.mean().item() + rewards_storage[step].mean().item(),
            })
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        wandb.log({"charts/episodic_return": item['episode']['r'], "charts/episodic_length": item['episode']['l']})

        # Calculate returns after the rollout is complete
        with torch.no_grad():
            # Get the bootstrapped value from the state after the last step
            bootstrap_ext_value, bootstrap_int_value = critic_network(next_obs)
            
            # Initialize tensors for returns
            ext_returns = torch.zeros_like(rewards_storage).to(device)
            int_returns = torch.zeros_like(intrinsic_rewards_storage).to(device)
            
            # Set the initial "next state" return. If an env was done, this is 0, otherwise it's the bootstrap value.
            # (1.0 - next_done) is a trick to multiply by 0 if done, and 1 if not done.
            ext_gt_next_state = bootstrap_ext_value.squeeze() * (1.0 - next_done)
            int_gt_next_state = bootstrap_int_value.squeeze() * (1.0 - next_done)

            # Loop backwards from the last step to the first
            for t in reversed(range(max_steps)):
                # Calculate return at step t
                rt_ext = rewards_storage[t] + ext_gamma * ext_gt_next_state
                rt_int = intrinsic_rewards_storage[t] + int_gamma * int_gt_next_state
                
                # Store the calculated returns
                ext_returns[t] = rt_ext
                int_returns[t] = rt_int
                
                # Update the "next state" for the previous step (t-1).
                # If an episode was done at step t, the return propagation is cut off (multiplied by 0).
                ext_gt_next_state = rt_ext * (1.0 - dones_storage[t])
                int_gt_next_state = rt_int * (1.0 - dones_storage[t])
            
        # Calculate advantages as the difference between returns and value function estimates
        ext_advantages = ext_returns - ext_values_storage
        int_advantages = int_returns - int_values_storage

        advantages = (INT_COEFF * int_advantages) + (EXT_COEFF * ext_advantages)
        wandb.log({
            "advantages/ext_advantages": ext_advantages.mean().item(), "advantages/int_advantages": int_advantages.mean().item()})
        
        # Normalize the final combined advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize extrinsic returns
        ext_returns = (ext_returns - ext_returns.mean()) / (ext_returns.std() + 1e-8)
        
        # Normalize intrinsic returns
        int_returns = (int_returns - int_returns.mean()) / (int_returns.std() + 1e-8)
        
        action_shape = () if isinstance(envs.single_action_space, gym.spaces.Discrete) else (action_space_n,)
        # Flatten the batch
        b_obs = obs_storage.reshape((-1, obs_space_shape))
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)

        # PPO & RND Update Phase
        b_inds = np.arange(batch_size)
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # --- PPO Policy and Value Loss ---
               
                _, new_log_probs, entropy = actor_network.get_action(b_obs[mb_inds], b_actions[mb_inds])
                ratio = torch.exp(new_log_probs - b_logprobs[mb_inds])
                
                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(ratio, 1 - clip_value, 1 + clip_value)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
               
                ext_current_values, int_current_values = critic_network(b_obs[mb_inds])
                # print("ext returns, int b returns ", b_ext_returns[mb_inds], b_int_returns[mb_inds])
                ext_critic_loss = F.mse_loss(ext_current_values.squeeze(), b_ext_returns[mb_inds])
                int_critic_loss = F.mse_loss(int_current_values.squeeze(), b_int_returns[mb_inds])
                critic_loss = EXT_COEFF * ext_critic_loss + INT_COEFF * int_critic_loss
                entropy_loss = entropy.mean()

                # --- RND Predictor Loss ---
                pred_features = predictor_network(b_obs[mb_inds])
                with torch.no_grad():
                    target_features = target_network(b_obs[mb_inds])
                intrinsic_loss = F.mse_loss(pred_features, target_features)

                # Total Loss
                loss = policy_loss + VALUE_COEFF * critic_loss + intrinsic_loss - ENTROPY_COEFF * entropy_loss
              
                optimizer.zero_grad()
                loss.backward()
                
                

                # Log gradient norm per layer
                if use_wandb and log_gradients:
                    
                    # Calculate gradient norm before clipping
                    total_norm_before = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2)
                                for p in list(actor_network.parameters()) + list(critic_network.parameters()) + list(predictor_network.parameters())
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
                                    f"gradients/actor_layer_{name}": grad_norm
                                    
                                }
                            )
                    for name, param in critic_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/critic_layer_{name}": grad_norm
                                    
                                }
                            )
                    for name, param in predictor_network.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/predictor_layer_{name}": grad_norm
                                    
                                }
                            )
                    wandb.log(
                        {
                            "gradients/norm_before_clip": total_norm_before.item()
                        }
                    )
                    
                # Apply gradient clipping
                if max_grad_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(actor_network.parameters()) + list(critic_network.parameters()) + list(predictor_network.parameters()), 
                        max_norm=max_grad_norm
                    )
                
                optimizer.step()
        
         # Log episode returns
        if "episode" in info:
            if n_envs > 1:
                for i in range(n_envs):
                    if done[i]:
                        ep_ret = info["episode"]["r"][i]
                        ep_len = info["episode"]["l"][i]

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
                    if use_wandb:
                        wandb.log(
                            {
                                "charts/episodic_return": ep_ret,
                                "charts/episodic_length": ep_len,
                                "charts/global_step": step,
                            }
                        )
        
                      
        # Log losses and metrics                       
        if use_wandb and update % 10 == 0:
            
            wandb.log({
                "losses/total_loss": loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": critic_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/intrinsic_loss": intrinsic_loss.item(),
                "charts/learning_rate": optimizer.param_groups[0]['lr'],
                "charts/rewards_mean": np.mean(rewards_storage.cpu().numpy()),
                "advantages/advantages_mean": b_advantages.mean().item(),
                "advantages/advantages_std": b_advantages.std().item(),
                "charts/ext_returns_mean": b_ext_returns.mean().item(),
                "charts/int_returns_mean": b_int_returns.mean().item(),
                "advantages/ext_advantages_mean": ext_advantages.mean().item(),
                "advantages/int_advantages_mean": int_advantages.mean().item(),
                # "global_step": global_step,
            })
            print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}, SPS: {int(global_step / (time.time() - start_time))}")
    
            if use_wandb:
                wandb.log(
                    {
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "charts/step": global_step,
                    }
                )
        if update % eval_every == 0:
            episodic_returns, _ = evaluate(actor_network, device, env_id, seed, run_name, num_eval_eps=num_eval_episodes, record=capture_video, render_mode='rgb_array', grid_env=grid_env, atari_wrapper=atari_wrapper, env_wrapper=env_wrapper, use_wandb=use_wandb)
            avg_return = np.mean(episodic_returns)
            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": global_step})
            print(f"Evaluation returns: {episodic_returns}, Average: {avg_return:.2f}")

        if update % save_every == 0 and update > 0:
            model_path = f"runs/{run_name}/models/rnd_model_step_{global_step}.pth"
            torch.save({"actor": actor_network.state_dict(), "critic": critic_network.state_dict(), "predictor": predictor_network.state_dict(), "target": target_network.state_dict()}, model_path)
            print(f"Model saved at update {update} step {global_step} to {model_path}")
    
        
         # --- Final Evaluation and Video Saving ---
    if capture_video:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(actor_network, device, env_id, seed, run_name, num_eval_eps=num_eval_episodes, record=True, render_mode='rgb_array', grid_env=grid_env, atari_wrapper=atari_wrapper, env_wrapper=env_wrapper, use_wandb=use_wandb)
                
    envs.close()
    if use_wandb:
        wandb.finish()

    return actor_network


# --- Main Execution ---
if __name__ == "__main__":
    train_rnd()