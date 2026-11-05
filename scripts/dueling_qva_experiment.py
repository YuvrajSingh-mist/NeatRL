#!/usr/bin/env python3
"""Dueling DQN ablation: mean-subtracted advantage vs plain Q = V + A.

Trains two otherwise identical Dueling DQN agents on the given environment:

  * ``standard``  -- the paper's aggregation, Q = V + A - mean_a A(s,a)
  * ``no_mean``   -- the requested ablation,       Q = V + A

It then records the value stream V(s), the advantage stream A(s,a) and the
resulting Q(s,a) on a shared set of states and on one greedy trajectory, and
writes the comparison plots / metrics to ``--out-dir``.

Works on vector-observation envs (CliffWalking, CartPole) and on Atari, where
the observation becomes the ``(4, 84, 84)`` frame stack the NeatRL ``-cnn``
alias builds. Atari is auto-detected from an ``ALE/`` env id, or forced with
``--atari``.

Usage:
    .venv/bin/python scripts/dueling_qva_experiment.py --steps 60000
    .venv/bin/python scripts/dueling_qva_experiment.py \\
        --env-id ALE/Pong-v5 --steps 40000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation  # noqa: E402
from stable_baselines3.common.buffers import ReplayBuffer  # noqa: E402

from neatrl.dueling_dqn_mlp import DuelingQNet  # noqa: E402
from neatrl.utils.wrappers import OneHotWrapper  # noqa: E402

try:  # Atari is an optional extra in NeatRL.
    import ale_py  # type: ignore[import-untyped]

    gym.register_envs(ale_py)
except ImportError:  # pragma: no cover - only hit without the atari extra
    ale_py = None

# NeatRL's Dueling-DQN defaults, kept identical for both variants.
HP = {
    "lr": 2e-4,
    "buffer_size": 30_000,
    "gamma": 0.99,
    "tau": 1.0,
    "target_network_frequency": 50,
    "batch_size": 128,
    "start_e": 1.0,
    "end_e": 0.05,
    "exploration_fraction": 0.4,
    "learning_starts": 1_000,
    "train_frequency": 4,
    "max_grad_norm": 0.5,
}

# Hand-picked states for the decomposition figure.
# CartPole observation = [x, x_dot, theta, theta_dot].
CARTPOLE_DEMO_STATES = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],  # balanced
        [0.0, 0.0, 0.10, 0.0],  # pole leaning right
        [0.0, 0.0, -0.10, 0.0],  # pole leaning left
        [1.0, 0.5, 0.05, 0.20],  # off-centre, moving right, slightly tilted
    ],
    dtype=np.float32,
)
CARTPOLE_DEMO_LABELS = [
    "balanced\nx=0, x'=0, θ=0, θ'=0",
    "pole leaning right\nθ=+0.10",
    "pole leaning left\nθ=-0.10",
    "off-centre\nx=1.0, x'=0.5, θ=0.05, θ'=0.2",
]

# CliffWalking is a 4x12 grid; state index is row-major (top-left = 0).
# Row 3 is the cliff (37-46), so only row 2 and the start/goal are shown.
CLIFF_DEMO_IDS = [36, 24, 30, 35]
CLIFF_DEMO_LABELS = [
    "start S\nstate 36",
    "above cliff\nstate 24",
    "above cliff middle\nstate 30",
    "above goal\nstate 35",
]


def get_demo_states(
    env_id: str, atari: bool = False, pool: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Return demo states and labels for the chosen env.

    Grid/vector envs use hand-picked states. Atari has no meaningful
    hand-written states, so it takes evenly spaced frames from the shared
    random-policy pool, labelled by their index in that pool.
    """
    if atari:
        if pool is None:
            raise ValueError("Atari demo states need the shared state pool")
        picks = np.linspace(0, len(pool) - 1, 4).astype(int)
        labels = [f"sampled frame\npool index {int(i)}" for i in picks]
        return pool[picks], labels
    if env_id.startswith("CliffWalking"):
        return np.eye(48, dtype=np.float32)[CLIFF_DEMO_IDS], CLIFF_DEMO_LABELS
    return CARTPOLE_DEMO_STATES, CARTPOLE_DEMO_LABELS


def get_action_labels(env_id: str, n_actions: int, atari: bool = False) -> list[str]:
    """Human-readable action names for the decomposition figure."""
    if atari:
        if "Pong" in env_id:
            return ["NOOP", "FIRE", "RIGHT", "LEFT", "UP", "DOWN"][:n_actions]
        if "Breakout" in env_id:
            return ["NOOP", "FIRE", "RIGHT", "LEFT"][:n_actions]
        return [f"a{i}" for i in range(n_actions)]
    if env_id.startswith("CliffWalking"):
        return ["Up", "Right", "Down", "Left"][:n_actions]
    if n_actions == 2:
        return ["Left", "Right"]
    return [f"a{i}" for i in range(n_actions)]


def frame_thumbnail(state: np.ndarray) -> np.ndarray | None:
    """Last frame of a stacked Atari observation, or None for vector states."""
    arr = np.asarray(state)
    if arr.ndim != 3:
        return None
    return arr[-1]


COLOR_STD = "#1f77b4"
COLOR_NM = "#d62728"
COLOR_V = "#2ca02c"
COLOR_A = "#9467bd"
COLOR_Q = "#1f77b4"


def linear_epsilon(
    step: int, start: float, end: float, total: int, frac: float
) -> float:
    """NeatRL's linear epsilon schedule."""
    decay_steps = max(1, int(total * frac))
    slope = (end - start) / decay_steps
    return max(slope * step + start, end)


def make_env(
    env_id: str, seed: int, record_stats: bool = False, atari: bool = False
) -> gym.Env:
    """Build a seeded env, matching NeatRL's wrappers for each family.

    Grid worlds such as CliffWalking expose ``Discrete(n)`` observations; the
    MLP trunk needs the one-hot ``Box(n,)`` form. Atari goes through the same
    ``AtariPreprocessing`` + 4-frame stack that NeatRL's ``-cnn`` alias builds.
    Environments without a spec time limit get a cap so episodes always end.
    """
    make_kwargs: dict = {"render_mode": "rgb_array"}
    if atari:
        # AtariPreprocessing does the frame skipping itself, and the paper's
        # setup disables sticky actions.
        make_kwargs["frameskip"] = 1
        make_kwargs["repeat_action_probability"] = 0.0
    env = gym.make(env_id, **make_kwargs)

    if atari:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
        env = FrameStackObservation(env, stack_size=4)
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHotWrapper(env, obs_shape=env.observation_space.n)

    spec = getattr(env, "spec", None)
    if getattr(spec, "max_episode_steps", None) is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=2_700 if atari else 500)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def obs_shape_of(env: gym.Env) -> int | tuple[int, ...]:
    """Flat size for vector observations, full shape for images."""
    shape = tuple(env.observation_space.shape)
    return shape if len(shape) > 1 else int(shape[0])


def resolve_device(requested: str | None) -> torch.device:
    """Pick the compute device, matching NeatRL's ``setup_device`` semantics.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. NeatRL's default config
    only ever picks ``cuda`` or ``cpu``, so on Apple silicon it silently stays
    on the CPU; passing ``--device mps`` (or ``auto`` here) opts in.
    """
    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available")
    return device


def train_variant(
    env_id: str,
    subtract_mean: bool,
    total_timesteps: int,
    seed: int,
    hp: dict,
    probe_states: np.ndarray | None = None,
    probe_every: int = 10_000,
    atari: bool = False,
    device: torch.device | None = None,
) -> tuple[DuelingQNet, list[tuple[int, float]], dict]:
    """Train one Dueling DQN variant exactly like NeatRL's inner loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or torch.device("cpu")

    env = make_env(env_id, seed, record_stats=True, atari=atari)
    obs, _ = env.reset(seed=seed)

    obs_shape = obs_shape_of(env)
    n_actions = int(env.action_space.n)

    online = DuelingQNet(obs_shape, n_actions, subtract_mean=subtract_mean).to(device)
    target = DuelingQNet(obs_shape, n_actions, subtract_mean=subtract_mean).to(device)
    target.load_state_dict(online.state_dict())

    optimizer = torch.optim.Adam(online.parameters(), lr=hp["lr"])
    buffer = ReplayBuffer(
        hp["buffer_size"],
        env.observation_space,
        env.action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    returns: list[tuple[int, float]] = []
    diag: dict[str, list[float]] = {
        "step": [],
        "mean_V": [],
        "mean_abs_mean_adv": [],
        "mean_raw_mean_adv": [],
        "mean_maxQ": [],
        "mean_action_spread": [],
    }
    online.train()
    target.train()
    t0 = time.time()

    for step in range(total_timesteps):
        eps = linear_epsilon(
            step,
            hp["start_e"],
            hp["end_e"],
            total_timesteps,
            hp["exploration_fraction"],
        )

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                q_values, _, _, _ = online(obs_t.to(device))
                action = int(q_values.argmax(dim=-1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        buffer.add(
            obs,
            next_obs,
            np.array([action]),
            np.array([reward]),
            np.array([done]),
            [info],
        )

        if "episode" in info:
            returns.append((step, float(info["episode"]["r"])))

        if step > hp["learning_starts"] and step % hp["train_frequency"] == 0:
            data = buffer.sample(hp["batch_size"])

            with torch.no_grad():
                target_q_values, _, _, _ = target(data.next_observations.to(device))
                target_max = target_q_values.max(1)[0]
                td_target = data.rewards.flatten().to(device) + hp["gamma"] * (
                    target_max
                ) * (1 - data.dones.flatten().to(device))

            q_values, _, _, _ = online(data.observations.to(device))
            old_val = q_values.gather(1, data.actions.to(device)).squeeze()
            loss = torch.nn.functional.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                online.parameters(), max_norm=hp["max_grad_norm"]
            )
            optimizer.step()

        if step % hp["target_network_frequency"] == 0:
            for q_params, target_params in zip(
                online.parameters(), target.parameters()
            ):
                target_params.data.copy_(
                    hp["tau"] * q_params.data + (1.0 - hp["tau"]) * target_params.data
                )

        if probe_states is not None and step % probe_every == 0:
            q_probe, v_probe, a_probe = qva_on(online, probe_states)
            eff = effective_adv(a_probe, subtract_mean)
            diag["step"].append(float(step))
            diag["mean_V"].append(float(v_probe.mean()))
            diag["mean_abs_mean_adv"].append(float(np.abs(eff.mean(axis=1)).mean()))
            diag["mean_raw_mean_adv"].append(float(a_probe.mean(axis=1).mean()))
            diag["mean_maxQ"].append(float(q_probe.max(axis=1).mean()))
            diag["mean_action_spread"].append(
                float((a_probe.max(axis=1) - a_probe.min(axis=1)).mean())
            )

        obs = next_obs
        if done:
            obs, _ = env.reset()

        if step and step % 10_000 == 0:
            recent = [r for _, r in returns[-20:]]
            avg = float(np.mean(recent)) if recent else float("nan")
            print(
                f"  step {step:6d}/{total_timesteps}  "
                f"eps={eps:.3f}  recent_return={avg:6.1f}  "
                f"({time.time() - t0:5.1f}s)",
                flush=True,
            )

    env.close()
    online.eval()
    print(
        f"  done in {time.time() - t0:.1f}s, "
        f"{len(returns)} episodes, last-20 return="
        f"{np.mean([r for _, r in returns[-20:]]):.1f}",
        flush=True,
    )
    return online, returns, diag


def greedy_rollout(
    net: DuelingQNet,
    env_id: str,
    seed: int,
    episodes: int = 10,
    record: bool = False,
    atari: bool = False,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Run greedy evaluation episodes and optionally record V/A/Q per step."""
    env = make_env(env_id, seed, atari=atari)
    net_device = next(net.parameters()).device
    all_returns: list[float] = []
    all_records: list[list[dict]] = []
    spec = getattr(env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) or (2_700 if atari else 500)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        recs: list[dict] = []
        n_steps = 0

        while not done and n_steps < max_steps:
            n_steps += 1
            with torch.no_grad():
                q, v, adv, _ = net(
                    torch.as_tensor(np.asarray(obs, dtype=np.float32))
                    .unsqueeze(0)
                    .to(net_device)
                )
            action = int(q.argmax(dim=-1).item())
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            if record:
                recs.append(
                    {
                        "obs": obs_before,
                        "action": action,
                        "reward": float(reward),
                        "q": q.squeeze(0).cpu().numpy(),
                        "v": float(v.squeeze().item()),
                        "adv": adv.squeeze(0).cpu().numpy(),
                    }
                )
            ep_return += float(reward)
            done = bool(terminated or truncated)

        all_returns.append(ep_return)
        all_records.append(recs)

    env.close()
    return np.asarray(all_returns), all_records


def collect_shared_states(
    env_id: str, n_states: int, seed: int, atari: bool = False
) -> np.ndarray:
    """Collect one fixed state set with a random policy, shared by both nets."""
    env = make_env(env_id, seed, atari=atari)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    states: list[np.ndarray] = []

    while len(states) < n_states:
        states.append(np.asarray(obs, dtype=np.float32).copy())
        obs, _, terminated, truncated, _ = env.step(
            int(rng.integers(env.action_space.n))
        )
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 10**9)))

    env.close()
    return np.stack(states)


def qva_on(
    net: DuelingQNet,
    states: np.ndarray,
    chunk: int = 256,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Q(s,:), V(s), A(s,:) for a batch of states.

    Chunked so a few thousand Atari frame stacks do not blow up memory.
    """
    device = device or next(net.parameters()).device
    qs, vs, advs = [], [], []
    with torch.no_grad():
        for start in range(0, len(states), chunk):
            batch = torch.as_tensor(
                np.asarray(states[start : start + chunk], dtype=np.float32)
            ).to(device)
            q, v, adv, _ = net(batch)
            qs.append(q.cpu().numpy())
            vs.append(v.squeeze(-1).cpu().numpy())
            advs.append(adv.cpu().numpy())
    return np.concatenate(qs), np.concatenate(vs), np.concatenate(advs)


def effective_adv(adv: np.ndarray, subtract_mean: bool) -> np.ndarray:
    """The advantage that actually enters Q: A - mean(A), or plain A."""
    if subtract_mean:
        return adv - adv.mean(axis=1, keepdims=True)
    return adv


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_decomposition(
    q_std: np.ndarray,
    v_std: np.ndarray,
    a_std: np.ndarray,
    q_nm: np.ndarray,
    v_nm: np.ndarray,
    a_nm: np.ndarray,
    labels: list[str],
    action_labels: list[str],
    env_id: str,
    out_path: str,
    demo_states: np.ndarray | None = None,
) -> None:
    """Grouped A/Q bars plus a V line for each demo state, one row per variant."""
    n_states = len(labels)
    n_actions = a_std.shape[1]
    fig, axes = plt.subplots(2, n_states, figsize=(4.8 * n_states, 9.6))
    if n_states == 1:
        axes = np.asarray(axes).reshape(2, 1)

    x = np.arange(n_actions)
    width = 0.38

    rows = [
        ("Standard:  Q = V + A − mean(A)", q_std, v_std, a_std, True),
        ("No mean subtraction:  Q = V + A", q_nm, v_nm, a_nm, False),
    ]

    for row, (row_title, q, v, a, sub) in enumerate(rows):
        for col in range(n_states):
            ax = axes[row, col]
            bars_a = ax.bar(
                x - width / 2,
                a[col],
                width=width,
                color=COLOR_A,
                edgecolor="white",
                linewidth=0.5,
                label="A(s,a)",
            )
            bars_q = ax.bar(
                x + width / 2,
                q[col],
                width=width,
                color=COLOR_Q,
                edgecolor="white",
                linewidth=0.5,
                label="Q(s,a)",
            )
            v_line = ax.axhline(
                v[col], color=COLOR_V, linestyle="--", linewidth=2.0, label="V(s)"
            )

            for bars in (bars_a, bars_q):
                for bar in bars:
                    val = float(bar.get_height())
                    ax.annotate(
                        f"{val:+.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3 if val >= 0 else -3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7,
                    )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(action_labels[:n_actions], fontsize=8)
            ax.grid(axis="y", alpha=0.25)

            mean_adv = float(a[col].mean())
            note = (
                f"mean(A)={mean_adv:+.2f}\nQ = V + A − {mean_adv:+.2f}"
                if sub
                else f"mean(A)={mean_adv:+.2f}\nQ = V + A"
            )
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
            ax.set_title(labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel("value", fontsize=10)
                ax.legend(handles=[v_line, bars_a, bars_q], fontsize=7, loc="best")

            # Atari: show the frame the numbers were computed from.
            if demo_states is not None:
                frame = frame_thumbnail(demo_states[col])
                if frame is not None:
                    inset = ax.inset_axes([0.60, 0.62, 0.36, 0.34])
                    inset.imshow(frame, cmap="gray" if frame.ndim == 2 else None)
                    inset.set_xticks([])
                    inset.set_yticks([])
        axes[row, 0].annotate(
            row_title,
            xy=(-0.34, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    subtitle = (
        "trained agents, frames sampled from random-policy states"
        if demo_states is not None and frame_thumbnail(demo_states[0]) is not None
        else "trained agents, hand-picked states"
    )
    fig.suptitle(
        f"Dueling DQN decomposition on {env_id}:  Q(s,a), V(s) and A(s,a)\n{subtitle}",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_gauge(
    diag_std: dict,
    diag_nm: dict,
    out_path: str,
) -> None:
    """V / mean-advantage / Q drift during training for both variants."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    series = [
        ("mean_V", "mean V(s) over probe states", "(a) Value stream drift"),
        (
            "mean_abs_mean_adv",
            r"mean$_s$ |mean$_a$ A$_{eff}$(s,a)|",
            "(b) Effective mean advantage",
        ),
        ("mean_maxQ", r"mean$_s$ max$_a$ Q(s,a)", "(c) Greedy Q estimate"),
    ]

    for ax, (key, ylabel, title) in zip(axes, series):
        ax.plot(
            diag_std["step"],
            diag_std[key],
            color=COLOR_STD,
            linewidth=2,
            label="standard: A − mean(A)",
        )
        ax.plot(
            diag_nm["step"],
            diag_nm[key],
            color=COLOR_NM,
            linewidth=2,
            label="no mean subtraction: A",
        )
        ax.set_xlabel("environment step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "Training-time values on a fixed probe set: the mean subtraction pins the gauge",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(
    returns_std: list[tuple[int, float]],
    returns_nm: list[tuple[int, float]],
    q_std: np.ndarray,
    v_std: np.ndarray,
    a_std: np.ndarray,
    q_nm: np.ndarray,
    v_nm: np.ndarray,
    a_nm: np.ndarray,
    eval_std: np.ndarray,
    eval_nm: np.ndarray,
    out_path: str,
) -> None:
    """Learning curves, mean-advantage gauge, and Q/V agreement."""
    eff_std = effective_adv(a_std, True)
    eff_nm = effective_adv(a_nm, False)
    mean_eff_std = eff_std.mean(axis=1)
    mean_eff_nm = eff_nm.mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) learning curves
    ax = axes[0, 0]
    for returns, label, color in (
        (returns_std, "Standard (A − mean A)", COLOR_STD),
        (returns_nm, "No mean subtraction (A)", COLOR_NM),
    ):
        if not returns:
            continue
        steps = np.array([s for s, _ in returns], dtype=float)
        rets = np.array([r for _, r in returns], dtype=float)
        ax.scatter(steps, rets, s=8, alpha=0.20, color=color)
        window = max(3, len(rets) // 60)
        if len(rets) >= window:
            smooth = np.convolve(rets, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], smooth, color=color, linewidth=2, label=label)
        else:
            ax.plot(steps, rets, color=color, linewidth=2, label=label)
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return")
    ax.set_title("(a) Learning curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"greedy eval:  standard {eval_std.mean():.0f}±{eval_std.std():.0f}\n"
        f"greedy eval:  no-mean {eval_nm.mean():.0f}±{eval_nm.std():.0f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    # (b) mean advantage per state (the gauge that the subtraction fixes)
    ax = axes[0, 1]
    order = np.argsort(mean_eff_nm)
    state_idx = np.arange(len(order))
    ax.plot(
        state_idx,
        mean_eff_std[order],
        color=COLOR_STD,
        linewidth=2.5,
        label="Standard: mean(A − mean A) ≡ 0",
    )
    ax.plot(
        state_idx,
        mean_eff_nm[order],
        color=COLOR_NM,
        linewidth=2,
        label="No mean subtraction: mean(A)",
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("shared state (sorted by no-mean value)")
    ax.set_ylabel(r"$\frac{1}{|\mathcal{A}|}\sum_a A_{\mathrm{eff}}(s,a)$")
    ax.set_title("(b) Mean advantage per state")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.04,
        f"std of mean A:  standard {mean_eff_std.std():.2e}  |  "
        f"no-mean {mean_eff_nm.std():.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    # (c) do the learned Q functions agree?
    ax = axes[1, 0]
    ax.scatter(q_std.ravel(), q_nm.ravel(), s=10, alpha=0.35, color="#555555")
    lo = float(min(q_std.min(), q_nm.min()))
    hi = float(max(q_std.max(), q_nm.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Q standard")
    ax.set_ylabel("Q no mean subtraction")
    corr_q = safe_corr(q_std.ravel(), q_nm.ravel())
    policy_agree = float((q_std.argmax(1) == q_nm.argmax(1)).mean())
    ax.set_title(
        f"(c) Q functions on shared states  (r={corr_q:.3f}, "
        f"mean|ΔQ|={np.abs(q_std - q_nm).mean():.3f})"
    )
    ax.text(
        0.02,
        0.98,
        f"greedy action agreement: {100 * policy_agree:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(alpha=0.25)

    # (d) V shifts by the unconstrained mean advantage
    ax = axes[1, 1]
    delta_v = v_nm - v_std
    ax.scatter(mean_eff_nm, delta_v, s=12, alpha=0.45, color="#7f7f7f")
    lim = float(np.max(np.abs(np.concatenate([mean_eff_nm, delta_v])))) * 1.1
    ax.plot([-lim, lim], [lim, -lim], "k--", linewidth=1, label="V shift = −mean(A)")
    ax.set_xlabel(r"mean$_a$ A$_{no\_mean}$(s,a)")
    ax.set_ylabel("V$_{no\\_mean}$(s) − V$_{standard}$(s)")
    corr_shift = safe_corr(mean_eff_nm, delta_v)
    ax.set_title(f"(d) The value stream absorbs the offset  (r={corr_shift:.3f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Dueling DQN with vs without the mean-advantage subtraction on CartPole-v1",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    records: list[dict],
    net_std: DuelingQNet,
    net_nm: DuelingQNet,
    out_path: str,
) -> None:
    """V, max Q and chosen-action advantage along one greedy episode."""
    states = np.stack([r["obs"] for r in records])
    actions = np.array([r["action"] for r in records], dtype=int)

    q_std, v_std, a_std = qva_on(net_std, states)
    q_nm, v_nm, a_nm = qva_on(net_nm, states)
    eff_std = effective_adv(a_std, True)
    eff_nm = a_nm

    t = np.arange(len(states))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, v_std, color=COLOR_STD, linewidth=1.8, label="standard")
    axes[0].plot(t, v_nm, color=COLOR_NM, linewidth=1.8, label="no mean subtraction")
    axes[0].set_ylabel("V(s_t)")
    axes[0].set_title("Value stream along a greedy CartPole episode")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, q_std.max(axis=1), color=COLOR_STD, linewidth=1.8, label="standard")
    axes[1].plot(
        t, q_nm.max(axis=1), color=COLOR_NM, linewidth=1.8, label="no mean subtraction"
    )
    axes[1].set_ylabel(r"$\max_a Q(s_t,a)$")
    axes[1].set_title("Greedy Q value")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        t,
        eff_std[t, actions],
        color=COLOR_STD,
        linewidth=1.8,
        label="standard: A − mean(A)",
    )
    axes[2].plot(
        t,
        eff_nm[t, actions],
        color=COLOR_NM,
        linewidth=1.8,
        label="no mean subtraction: A",
    )
    axes[2].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[2].set_ylabel(r"$A_{\mathrm{eff}}(s_t,a_t)$")
    axes[2].set_xlabel("timestep in episode")
    axes[2].set_title("Advantage of the action actually taken")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Same greedy trajectory (length {len(states)}, return {sum(r['reward'] for r in records):.0f})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CliffWalking-v1")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--states", type=int, default=1_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-dir", default="runs/dueling_qva")
    parser.add_argument(
        "--atari",
        action="store_true",
        help="Force the Atari preprocessing + 4-frame stack (auto for ALE/*).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help=(
            "Replay buffer capacity. Defaults to NeatRL's 30000, but Atari "
            "defaults to 10000 because 30000 (4,84,84) float stacks is ~3.4 GB."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Compute device. 'auto' prefers cuda, then Apple mps, then cpu. "
            "NeatRL's own Config only ever picks cuda or cpu, so MPS needs this."
        ),
    )
    args = parser.parse_args()

    atari = args.atari or args.env_id.startswith("ALE/")
    if atari and ale_py is None:
        raise SystemExit(
            "Atari needs the optional extra: uv pip install 'gymnasium[atari]' ale-py"
        )

    hp = dict(HP)
    if args.buffer_size is not None:
        hp["buffer_size"] = args.buffer_size
    elif atari:
        hp["buffer_size"] = 10_000

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(
        f"Environment: {args.env_id} (atari={atari}), "
        f"steps per variant: {args.steps}, device: {device.type}",
        flush=True,
    )

    # One fixed probe/state set, collected before training and shared by both
    # variants, so every V/A/Q comparison is on identical states.
    shared = collect_shared_states(
        args.env_id, args.states, seed=args.seed + 7, atari=atari
    )

    print("\n[1/3] Standard Dueling DQN  (Q = V + A - mean A)", flush=True)
    net_std, returns_std, diag_std = train_variant(
        args.env_id,
        subtract_mean=True,
        total_timesteps=args.steps,
        seed=args.seed,
        hp=hp,
        probe_states=shared,
        atari=atari,
        device=device,
    )

    print("\n[2/3] Ablation              (Q = V + A)", flush=True)
    net_nm, returns_nm, diag_nm = train_variant(
        args.env_id,
        subtract_mean=False,
        total_timesteps=args.steps,
        seed=args.seed,
        hp=hp,
        probe_states=shared,
        atari=atari,
        device=device,
    )

    print("\n[3/3] Collecting V / A / Q and plotting", flush=True)

    q_std, v_std, a_std = qva_on(net_std, shared)
    q_nm, v_nm, a_nm = qva_on(net_nm, shared)

    # Greedy evaluation (also records one trajectory from the standard policy).
    eval_std, rec_std = greedy_rollout(
        net_std,
        args.env_id,
        seed=args.seed + 100,
        episodes=args.episodes,
        record=True,
        atari=atari,
    )
    eval_nm, _ = greedy_rollout(
        net_nm,
        args.env_id,
        seed=args.seed + 100,
        episodes=args.episodes,
        record=True,
        atari=atari,
    )

    # Demo states for the decomposition bars.
    demo_states, demo_labels = get_demo_states(args.env_id, atari=atari, pool=shared)
    demo_q_std, demo_v_std, demo_a_std = qva_on(net_std, demo_states)
    demo_q_nm, demo_v_nm, demo_a_nm = qva_on(net_nm, demo_states)
    action_labels = get_action_labels(args.env_id, a_std.shape[1], atari=atari)

    eff_std = effective_adv(a_std, True)
    eff_nm = effective_adv(a_nm, False)
    mean_eff_std = eff_std.mean(axis=1)
    mean_eff_nm = eff_nm.mean(axis=1)

    metrics = {
        "env_id": args.env_id,
        "steps": args.steps,
        "seed": args.seed,
        "device": device.type,
        "hyperparameters": hp,
        "standard": {
            "train_episodes": len(returns_std),
            "final_train_return": float(np.mean([r for _, r in returns_std[-20:]])),
            "eval_return_mean": float(eval_std.mean()),
            "eval_return_std": float(eval_std.std()),
            "mean_abs_mean_effective_adv": float(np.abs(mean_eff_std).mean()),
            "std_of_mean_effective_adv": float(mean_eff_std.std()),
            "mean_action_spread": float((a_std.max(1) - a_std.min(1)).mean()),
            "mean_V": float(v_std.mean()),
            "std_V": float(v_std.std()),
            "mean_Q": float(q_std.mean()),
            "std_Q": float(q_std.std()),
            "mean_raw_mean_adv": float(a_std.mean(1).mean()),
        },
        "no_mean": {
            "train_episodes": len(returns_nm),
            "final_train_return": float(np.mean([r for _, r in returns_nm[-20:]])),
            "eval_return_mean": float(eval_nm.mean()),
            "eval_return_std": float(eval_nm.std()),
            "mean_abs_mean_effective_adv": float(np.abs(mean_eff_nm).mean()),
            "std_of_mean_effective_adv": float(mean_eff_nm.std()),
            "mean_action_spread": float((a_nm.max(1) - a_nm.min(1)).mean()),
            "mean_V": float(v_nm.mean()),
            "std_V": float(v_nm.std()),
            "mean_Q": float(q_nm.mean()),
            "std_Q": float(q_nm.std()),
            "mean_raw_mean_adv": float(a_nm.mean(1).mean()),
        },
        "comparison": {
            "policy_agreement": float((q_std.argmax(1) == q_nm.argmax(1)).mean()),
            "corr_Q": safe_corr(q_std.ravel(), q_nm.ravel()),
            "mean_abs_Q_diff": float(np.abs(q_std - q_nm).mean()),
            "corr_V": safe_corr(v_std, v_nm),
            "mean_abs_V_diff": float(np.abs(v_std - v_nm).mean()),
            "corr_of_V_shift_with_mean_adv": safe_corr(v_nm - v_std, mean_eff_nm),
        },
        "training_diagnostics": {"standard": diag_std, "no_mean": diag_nm},
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, "qva_shared_states.npz"),
        states=shared,
        q_std=q_std,
        v_std=v_std,
        a_std=a_std,
        q_nm=q_nm,
        v_nm=v_nm,
        a_nm=a_nm,
    )

    plot_decomposition(
        demo_q_std,
        demo_v_std,
        demo_a_std,
        demo_q_nm,
        demo_v_nm,
        demo_a_nm,
        demo_labels,
        action_labels,
        args.env_id,
        os.path.join(args.out_dir, "01_qva_decomposition.png"),
        demo_states=demo_states,
    )

    plot_training_gauge(
        diag_std,
        diag_nm,
        os.path.join(args.out_dir, "02_training_gauge.png"),
    )

    plot_diagnostics(
        returns_std,
        returns_nm,
        q_std,
        v_std,
        a_std,
        q_nm,
        v_nm,
        a_nm,
        eval_std,
        eval_nm,
        os.path.join(args.out_dir, "03_diagnostics.png"),
    )

    # Trajectory: median-return episode of the standard agent.
    if rec_std:
        order = np.argsort(eval_std)
        median_idx = int(order[len(order) // 2])
        plot_trajectory(
            rec_std[median_idx],
            net_std,
            net_nm,
            os.path.join(args.out_dir, "04_trajectory.png"),
        )

    # Save CPU copies so the checkpoints load anywhere.
    torch.save(
        {k: v.cpu() for k, v in net_std.state_dict().items()},
        os.path.join(args.out_dir, "dueling_standard.pt"),
    )
    torch.save(
        {k: v.cpu() for k, v in net_nm.state_dict().items()},
        os.path.join(args.out_dir, "dueling_no_mean.pt"),
    )

    print("\n=== Summary ===")
    print(f"{'metric':<38}{'standard':>14}{'no mean':>14}")
    for key in (
        "eval_return_mean",
        "mean_abs_mean_effective_adv",
        "std_of_mean_effective_adv",
        "mean_action_spread",
        "mean_V",
        "std_V",
        "mean_Q",
        "std_Q",
    ):
        s = metrics["standard"][key]
        n = metrics["no_mean"][key]
        print(f"{key:<38}{s:>14.4f}{n:>14.4f}")
    comp = metrics["comparison"]
    print("\nComparison on shared states:")
    print(f"  policy agreement          : {100 * comp['policy_agreement']:.1f}%")
    print(f"  corr(Q_std, Q_no_mean)    : {comp['corr_Q']:.4f}")
    print(f"  mean |Q_std - Q_no_mean|  : {comp['mean_abs_Q_diff']:.4f}")
    print(f"  corr(V_std, V_no_mean)    : {comp['corr_V']:.4f}")
    print(f"  mean |V_std - V_no_mean|  : {comp['mean_abs_V_diff']:.4f}")
    print(f"  corr(V shift, mean A)     : {comp['corr_of_V_shift_with_mean_adv']:.4f}")
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
