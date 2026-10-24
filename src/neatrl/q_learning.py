"""Tabular Q-learning for discrete-state, discrete-action Gymnasium environments.

Simple Gym envs only (e.g. FrozenLake, CliffWalking, Taxi). No neural net -
a Q-table is enough when |S| and |A| are small.
"""

import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from .utils import configure_logging, get_logger

logger = get_logger(__name__)

try:
    from .cli.dashboard import Dashboard
except ImportError:  # pragma: no cover - optional rich/psutil CLI extra
    Dashboard = None  # type: ignore[misc, assignment]


# ===== CONFIGURATION =====
@dataclass
class Config:
    """Hyperparameters and settings for tabular Q-learning."""

    exp_name: str = "Q-learning"
    seed: int = 42
    env_id: str = "FrozenLake-v1"

    total_timesteps: int = 50000
    lr: float = 0.1  # alpha - tabular step size
    gamma: float = 0.99
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    num_eval_eps: int = 10

    eval_every: int = 2000
    save_every: int = 5000
    capture_video: bool = False

    use_wandb: bool = True
    wandb_project: str = "cleanRL"
    wandb_entity: str = ""


def _require_discrete(env: gym.Env) -> tuple[int, int]:
    """Return (n_states, n_actions) or raise if spaces are not Discrete."""
    obs_space = env.observation_space
    act_space = env.action_space
    if not isinstance(obs_space, gym.spaces.Discrete):
        raise ValueError(
            "Q-learning is tabular and only supports Discrete observation spaces "
            f"(got {type(obs_space).__name__}). Use FrozenLake, CliffWalking, Taxi, "
            "or similar simple Gym envs - or use DQN for continuous / Box observations."
        )
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(
            "Q-learning only supports Discrete action spaces "
            f"(got {type(act_space).__name__})."
        )
    return int(obs_space.n), int(act_space.n)


def make_env(env_id: str, seed: int, idx: int = 0):
    """Return a thunk that constructs and seeds a Gymnasium environment.

    Args:
        env_id: Gymnasium environment ID.
        seed: Base random seed; actual seed is ``seed + idx``.
        idx: Worker index added to the seed.

    Returns:
        Callable[[], gym.Env]: Zero-argument thunk that creates the environment.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


def epsilon_by_step(
    step: int,
    total_timesteps: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
) -> float:
    """Linearly decay epsilon from start_e to end_e over a fraction of training."""
    decay_steps = max(1, int(total_timesteps * exploration_fraction))
    if step >= decay_steps:
        return end_e
    slope = (end_e - start_e) / decay_steps
    return float(start_e + slope * step)


def select_action(q_table: np.ndarray, state: int, eps: float, n_actions: int) -> int:
    """Epsilon-greedy action from the Q-table."""
    if random.random() < eps:
        return int(random.randrange(n_actions))
    return int(np.argmax(q_table[state]))


def evaluate(
    env_id: str,
    q_table: np.ndarray,
    seed: int,
    num_eval_eps: int = 10,
    capture_video: bool = False,
):
    """Run greedy evaluation episodes; return episode returns and optional frames.

    Args:
        env_id: Gymnasium environment ID.
        q_table: Q-table of shape ``(n_states, n_actions)``.
        seed: Random seed for the evaluation environment.
        num_eval_eps: Number of episodes to run.
        capture_video: Record frames when True.

    Returns:
        tuple[list[float], list]: Episode returns and recorded RGB frames
            (empty if ``capture_video`` is False).
    """
    eval_env = make_env(env_id=env_id, seed=seed, idx=0)()
    eval_env.action_space.seed(seed)
    _require_discrete(eval_env)

    returns = []
    frames: list = []

    for _ in range(num_eval_eps):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        ep_frames: list = []

        while not done:
            if capture_video:
                try:
                    frame = eval_env.render()
                    ep_frames.append(frame)
                except Exception as e:
                    logger.error(
                        "Video capture failed (%s). "
                        "Check the renderer for this environment is installed.",
                        type(e).__name__,
                    )
                    raise

            action = int(np.argmax(q_table[int(obs)]))
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)

        returns.append(episode_reward)

        if ep_frames and capture_video:
            video = np.stack(ep_frames)
            video = np.transpose(video, (0, 3, 1, 2))
            if wandb.run is not None:
                wandb.log(
                    {"videos/eval_policy": wandb.Video(video, fps=5, format="mp4")}
                )
            frames.extend(ep_frames)

    eval_env.close()
    return returns, frames


def train_q_learning(
    env_id=Config.env_id,
    total_timesteps=Config.total_timesteps,
    seed=Config.seed,
    lr=Config.lr,
    gamma=Config.gamma,
    start_e=Config.start_e,
    end_e=Config.end_e,
    exploration_fraction=Config.exploration_fraction,
    capture_video=Config.capture_video,
    use_wandb=Config.use_wandb,
    wandb_project=Config.wandb_project,
    wandb_entity=Config.wandb_entity,
    exp_name=Config.exp_name,
    eval_every=Config.eval_every,
    save_every=Config.save_every,
    num_eval_eps=Config.num_eval_eps,
):
    """Train tabular Q-learning on a discrete Gymnasium environment.

    Hyperparameters match the signature.
    Returns the trained Q-table.
    """
    Config.env_id = env_id
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            sync_tensorboard=False,
            config=locals(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    configure_logging(log_dir=os.path.join("runs", exp_name))
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(env_id, seed, idx=0)()
    n_states, n_actions = _require_discrete(env)

    # Q(s, a) table - zeros; true terminals bootstrap with 0
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    logger.debug(
        "Q-table shape=%s (states=%d, actions=%d)", q_table.shape, n_states, n_actions
    )
    if use_wandb:
        wandb.config.update({"n_states": n_states, "n_actions": n_actions})

    obs, _ = env.reset(seed=seed)
    start_time = time.time()
    latest_avg_return = 0.0
    latest_ep_return = 0.0
    update_count = 0
    latest_td_error = 0.0

    # Live dashboard only on an interactive TTY (keeps CI / pytest quiet)
    dashboard = None
    if Dashboard is not None and sys.stdout.isatty():
        dashboard = Dashboard(
            "Q-learning", Config.env_id or "custom", total_timesteps, config=Config
        )

    for step in tqdm(range(total_timesteps), disable=True):
        eps = epsilon_by_step(
            step, total_timesteps, start_e, end_e, exploration_fraction
        )
        state = int(obs)
        action = select_action(q_table, state, eps, n_actions)

        if dashboard:
            dashboard.push_metrics(
                epsilon=eps,
                q_mean=float(q_table[state].mean()),
            )

        new_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = int(new_obs)

        # Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
        # Bootstrap only when not truly terminated (truncation still bootstraps).
        best_next = 0.0 if terminated else float(np.max(q_table[next_state]))
        td_target = float(reward) + gamma * best_next
        td_error = td_target - q_table[state, action]
        q_table[state, action] += lr * td_error

        update_count += 1
        latest_td_error = float(td_error)

        if dashboard:
            dashboard.push_metrics(total_loss=abs(latest_td_error))

        if use_wandb and step % 100 == 0:
            wandb.log(
                {
                    "losses/td_error": latest_td_error,
                    "charts/epsilon": eps,
                    "q_values/mean": float(q_table.mean()),
                    "q_values/max": float(q_table.max()),
                    "step": step,
                }
            )

        if "episode" in info and done:
            ep_ret = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
            ep_len = int(np.asarray(info["episode"]["l"]).reshape(-1)[0])
            latest_ep_return = ep_ret
            logger.info("Step=%d Return=%.2f Length=%d", step, ep_ret, ep_len)
            if use_wandb:
                wandb.log(
                    {
                        "charts/episodic_return": ep_ret,
                        "charts/episodic_length": ep_len,
                        "charts/global_step": step,
                    }
                )
            if dashboard:
                dashboard.push_metrics(ep_return=ep_ret, ep_length=float(ep_len))

        if step > 0 and step % eval_every == 0:
            episodic_returns, _ = evaluate(
                env_id,
                q_table,
                seed,
                num_eval_eps=num_eval_eps,
                capture_video=capture_video,
            )
            avg_return = float(np.mean(episodic_returns))
            latest_avg_return = avg_return
            if use_wandb:
                wandb.log({"charts/val_avg_return": avg_return, "val_step": step})
            logger.info(
                "Evaluation returns: %s, Average: %.2f",
                episodic_returns,
                avg_return,
            )

        if done:
            obs, _ = env.reset()
        else:
            obs = new_obs

        if dashboard:
            dashboard.update(
                agent_steps=step,
                epoch=update_count,
                losses={"td_error": abs(latest_td_error)},
                eval_stats={
                    "avg_return": latest_avg_return,
                    "last_return": latest_ep_return,
                },
                message=f"Q-table: {n_states}x{n_actions}",
            )

        if use_wandb:
            wandb.log(
                {
                    "charts/SPS": int(step / max(time.time() - start_time, 1e-6)),
                    "charts/step": step,
                    "step": step,
                }
            )

        if step > 0 and step % save_every == 0:
            model_path = f"runs/{run_name}/models/q_table_step_{step}.npy"
            np.save(model_path, q_table)
            logger.info("Q-table saved at step %d to %s", step, model_path)

    final_path = f"runs/{run_name}/models/q_table_final.npy"
    np.save(final_path, q_table)
    logger.info("Final Q-table saved to %s", final_path)

    if use_wandb:
        wandb.finish()

    env.close()
    if dashboard:
        dashboard.close(
            message=(
                f"Q-table: {n_states}x{n_actions}  |  "
                f"Updates: {update_count}  |  "
                f"Final avg_return: {latest_avg_return:.1f}"
            )
        )

    return q_table


if __name__ == "__main__":
    train_q_learning(use_wandb=False)
