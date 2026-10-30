"""Watch DQN overestimate, then watch Double DQN fix it.

What is measured (Figure 3 of van Hasselt et al. 2016)
------------------------------------------------------
Run *greedy* evaluation episodes and, for each state the policy visits, pair
the network's estimate with what actually happened from that state onward:

    predicted : max_a Q(s_t, a)
    realised  : sum_{k >= t} gamma^(k-t) r_k

Both are averaged over the same states. An unbiased critic makes the two curves
coincide; overestimation shows up as the solid line riding above the dashed one.

Where the bias comes from
-------------------------
The DQN target is ``r + gamma * max_a Q_target(s', a)``. Noise in ``Q_target``
passes through a ``max``, and ``max`` is convex, so

    E[max_a Q_hat(s', a)]  >=  max_a E[Q_hat(s', a)]

The target is therefore biased upward, and because the agent bootstraps from
its own inflated values the error feeds back into itself. Double DQN splits the
``max`` into action *selection* (online net) and action *evaluation* (target
net), so the noise is no longer maximised over the same estimate.

Choosing an environment
-----------------------
The phenomenon needs the agent to actually learn. Overestimation is a bias in
the *target*, and while the network is still under-fitting, "predicted minus
realised" mostly measures fit error, which swamps it.

* ``CartPole-v1`` (MLP) learns to near-optimal in well under a minute and its
  values are large (up to ~1/(1-gamma) = 100), so the bias is easy to see.
  This is the one to start with.
* ``ALE/Pong-v5`` (CNN) reproduces the paper's setting but needs the paper's
  budget: 50M frames is ~12.5M agent steps, i.e. many hours. At 1M steps the
  agent is still at random play and the measurement says nothing.

Usage
-----
    python docs/non-cli/DQN/overestimation_demo.py                  # CartPole, both
    python docs/non-cli/DQN/overestimation_demo.py --algorithms dqn # DQN only first
    python docs/non-cli/DQN/overestimation_demo.py --steps 120000
    python docs/non-cli/DQN/overestimation_demo.py --env-id ALE/Pong-v5 --steps 5000

Output: one printed line per evaluation, a CSV, and a three-panel PNG (DQN,
Double DQN, both gaps) that is rewritten after every evaluation so you can
watch it while the run is going.
"""

import argparse
import csv
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import torch

from neatrl.double_dqn_cnn import train_double_dqn_cnn
from neatrl.double_dqn_mlp import train_double_dqn
from neatrl.dqn_cnn import make_env as cnn_make_env
from neatrl.dqn_cnn import train_dqn_cnn
from neatrl.dqn_mlp import make_env as mlp_make_env
from neatrl.dqn_mlp import train_dqn

ALGORITHMS = ("dqn", "double_dqn")
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}
COLOURS = {"dqn": "tab:red", "double_dqn": "tab:blue"}


@dataclass
class Point:
    """One (algorithm, seed, eval step) measurement from the paper's protocol."""

    algorithm: str
    seed: int
    step: int
    mean_max_q: float
    true_value: float
    gap: float
    avg_return: float


def is_atari(env_id: str) -> bool:
    """Atari ids are the ones that need the CNN front end."""
    return "ALE/" in env_id or "NoFrameskip" in env_id


def resolve_device(name: Optional[str]) -> str:
    """Pick the fastest available device unless the caller forced one."""
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line for the overestimation demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=60_000, help="agent steps per run")
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds")
    parser.add_argument("--eval-every", type=int, default=3_000)
    parser.add_argument("--num-eval-eps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-network-frequency", type=int, default=100)
    parser.add_argument("--train-frequency", type=int, default=10)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument(
        "--grid-env",
        action="store_true",
        help="one-hot encode discrete observations (FrozenLake, CliffWalking, Taxi)",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda or mps")
    parser.add_argument("--out-dir", default=os.path.join("runs", "overestimation"))
    return parser.parse_args(argv)


def measure_values(
    model: Any,
    env_id: str,
    device: str,
    gamma: float,
    num_episodes: int,
    atari: bool,
    seed: int = 0,
    grid_env: bool = False,
) -> tuple[float, float, float]:
    """Return (mean predicted value, mean realised return, mean episode return).

    Every state visited by the greedy policy contributes one prediction and the
    discounted return that followed it, so the two averages are over the same
    states.
    """
    # MLP envs (vector observations, incl. one-hot grid envs) use the MLP helper;
    # image envs use the Atari preprocessing helper (which has no grid_env).
    if atari:
        env = cnn_make_env(env_id, seed, idx=0, atari_wrapper=True)()
    else:
        env = mlp_make_env(
            env_id, seed, idx=0, atari_wrapper=False, grid_env=grid_env
        )()
    model = model.to(device)
    model.eval()

    predictions: list[float] = []
    realised: list[float] = []
    episode_returns: list[float] = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_predictions: list[float] = []
        episode_rewards: list[float] = []

        # Envs without a TimeLimit (e.g. CliffWalking-v1) need a hard cap, or a
        # circling greedy policy would never end the episode.
        spec = getattr(env, "spec", None)
        max_steps = getattr(spec, "max_episode_steps", None) or 10_000

        for _step in range(max_steps):
            array = np.asarray(obs)
            dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
            batch = torch.as_tensor(array, device=device, dtype=dtype).unsqueeze(0)
            with torch.no_grad():
                q_values = model(batch)
            episode_predictions.append(float(q_values.max().item()))

            action = int(q_values.argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
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
        episode_returns.append(float(np.sum(episode_rewards)))

    env.close()
    model.train()
    if not predictions:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(predictions)),
        float(np.mean(realised)),
        float(np.mean(episode_returns)),
    )


def output_paths(args: argparse.Namespace) -> tuple[str, str]:
    """CSV and PNG paths, tagged with env and budget so runs cannot collide."""
    name = args.env_id.split("/")[-1].lower()
    name = re.sub(r"-v\d+$", "", name)  # LunarLander-v3 -> lunarlander
    tag = f"{name}_{args.steps}steps"
    return (
        os.path.join(args.out_dir, f"overestimation_{tag}.csv"),
        os.path.join(args.out_dir, f"overestimation_{tag}.png"),
    )


def run_agent(
    algorithm: str,
    args: argparse.Namespace,
    seed: int,
    atari: bool,
    points: list[Point],
    csv_path: str,
    png_path: str,
) -> None:
    """Train one agent, measuring the paper's two quantities at every eval."""
    if atari:
        train_fn = train_dqn_cnn if algorithm == "dqn" else train_double_dqn_cnn
    else:
        train_fn = train_dqn if algorithm == "dqn" else train_double_dqn

    def on_eval(step: int, metrics: dict) -> None:
        # Re-measure here so both algorithms use the identical protocol and
        # episode count regardless of what the trainer itself logs.
        predicted, realised, episode_return = measure_values(
            metrics["model"],
            args.env_id,
            args.device,
            args.gamma,
            args.num_eval_eps,
            atari=atari,
            seed=0,
            grid_env=args.grid_env,
        )
        point = Point(
            algorithm=algorithm,
            seed=seed,
            step=step,
            mean_max_q=predicted,
            true_value=realised,
            gap=predicted - realised,
            avg_return=episode_return,
        )
        points.append(point)
        print(
            f"  {LABELS[algorithm]:>10s} seed={seed} step={step:>7d} | "
            f"predicted={point.mean_max_q:+8.3f} | "
            f"realised={point.true_value:+8.3f} | "
            f"gap={point.gap:+8.3f} | "
            f"episode return={point.avg_return:+8.1f}",
            flush=True,
        )
        save_csv(points, csv_path)
        plot(points, png_path, atari)

    kwargs: dict[str, Any] = {
        "env_id": args.env_id,
        "total_timesteps": args.steps,
        "seed": seed,
        "lr": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "target_network_frequency": args.target_network_frequency,
        "exploration_fraction": args.exploration_fraction,
        "eval_every": args.eval_every,
        "save_every": 0,
        "num_eval_eps": args.num_eval_eps,
        "use_wandb": False,
        "capture_video": False,
        "device": args.device,
        "exp_name": f"{algorithm.upper()}-overestimation",
        "eval_callback": on_eval,
    }
    if not atari:
        kwargs["train_frequency"] = args.train_frequency
        kwargs["atari_wrapper"] = False
        kwargs["grid_env"] = args.grid_env

    print(f"\n=== training {LABELS[algorithm]} (seed {seed}) ===", flush=True)
    train_fn(**kwargs)


def summarise(points: list[Point], algorithms: list[str]) -> None:
    """Report the mean gap per algorithm and their difference."""
    means: dict[str, float] = {}
    for algorithm in algorithms:
        gaps = [p.gap for p in points if p.algorithm == algorithm]
        if not gaps:
            continue
        means[algorithm] = float(np.mean(gaps))
        print(
            f"{LABELS[algorithm]:>10s}: gap overall = {np.mean(gaps):+7.3f} | "
            f"last half = {np.mean(gaps[len(gaps) // 2 :]):+7.3f} | "
            f"peak = {max(gaps):+7.3f}"
        )
    if "dqn" in means and "double_dqn" in means:
        print(
            f"{'difference':>10s}: DQN predicts "
            f"{means['dqn'] - means['double_dqn']:+.3f} more than Double DQN "
            "(positive = overestimation)"
        )


def save_csv(points: list[Point], path: str) -> None:
    """Write every measurement to disk so the run can be replotted later."""
    if not points:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0])))
        writer.writeheader()
        writer.writerows(asdict(p) for p in points)


def _smooth(values: list[float], window: int = 3) -> np.ndarray:
    """Centred moving average, so the curve shows trend rather than noise."""
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, values[0]), values])
    padded = np.concatenate([padded, np.full(window // 2, values[-1])])
    return np.convolve(padded, kernel, mode="valid")


def plot(points: list[Point], path: str, atari: bool) -> None:
    """One panel per algorithm plus a shared gap panel."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return
    if not points:
        return

    present = [a for a in ALGORITHMS if any(p.algorithm == a for p in points)]
    figure, axes = plt.subplots(
        1, len(present) + 1, figsize=(5.0 * (len(present) + 1), 4.2), squeeze=False
    )
    axes = axes[0]

    for index, algorithm in enumerate(present):
        ax = axes[index]
        rows = [p for p in points if p.algorithm == algorithm]
        steps = [p.step / 1000.0 for p in rows]
        predicted = _smooth([p.mean_max_q for p in rows])
        actual = _smooth([p.true_value for p in rows])

        ax.plot(steps, predicted, color=COLOURS[algorithm], label="predicted max Q")
        ax.plot(
            steps,
            actual,
            color=COLOURS[algorithm],
            linestyle="--",
            label="realised return",
        )
        ax.fill_between(
            steps,
            actual,
            predicted,
            where=predicted >= actual,
            color=COLOURS[algorithm],
            alpha=0.18,
            label="overestimation",
        )
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(f"{LABELS[algorithm]}: estimate vs reality")
        ax.set_xlabel("thousand agent steps")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)

    ax_gap = axes[-1]
    ax_gap.axhline(0.0, color="gray", linewidth=0.8)
    for algorithm in present:
        rows = [p for p in points if p.algorithm == algorithm]
        ax_gap.plot(
            [p.step / 1000.0 for p in rows],
            _smooth([p.gap for p in rows]),
            color=COLOURS[algorithm],
            label=LABELS[algorithm],
        )
    ax_gap.set_title("Overestimation gap: max Q - realised return")
    ax_gap.set_xlabel("thousand agent steps")
    ax_gap.set_ylabel("gap")
    ax_gap.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> None:
    """Run the DQN / Double DQN overestimation comparison."""
    args = build_args(argv)
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"Unknown algorithm(s) {unknown}; pick from {ALGORITHMS}")

    args.device = resolve_device(args.device)
    atari = is_atari(args.env_id)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path, png_path = output_paths(args)

    print(
        f"env={args.env_id} steps={args.steps} eval_every={args.eval_every} "
        f"device={args.device} network={'CNN' if atari else 'MLP'}"
    )
    print(f"order: {', '.join(LABELS[a] for a in algorithms)}")

    points: list[Point] = []
    for seed in range(args.seeds):
        for algorithm in algorithms:
            run_agent(algorithm, args, seed, atari, points, csv_path, png_path)

    print("\n=== summary (greedy eval: predicted vs realised) ===")
    summarise(points, algorithms)
    save_csv(points, csv_path)
    plot(points, png_path, atari)
    print(f"Saved measurements to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
