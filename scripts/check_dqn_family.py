#!/usr/bin/env python3
"""Test matrix for the DQN family: discrete obs/actions and continuous obs/discrete actions.

Runs every trainer in the DQN family on every environment type and asserts:
  * training completes and returns an nn.Module
  * observations/actions had the shapes the trainer expects
  * evaluation ran and produced finite returns
  * the Double DQN target really differs from the plain max target

Usage:
    python scripts/check_dqn_family.py            # all cases
    python scripts/check_dqn_family.py --only dqn # filter by algorithm
"""

import argparse
import sys
import traceback

import numpy as np
import torch

from neatrl import (
    train_double_dqn,
    train_double_dqn_cnn,
    train_dqn,
    train_dqn_cnn,
)

# (env_id, obs_kind, grid_env, trainer_key, needs_cnn)
CASES = [
    # discrete observations, discrete actions
    ("FrozenLake-v1", "discrete", True, "dqn", False),
    ("FrozenLake-v1", "discrete", True, "double_dqn", False),
    ("CliffWalking-v1", "discrete", True, "dqn", False),
    ("CliffWalking-v1", "discrete", True, "double_dqn", False),
    ("Taxi-v4", "discrete", True, "dqn", False),
    ("Taxi-v4", "discrete", True, "double_dqn", False),
    # continuous observations, discrete actions
    ("CartPole-v1", "continuous", False, "dqn", False),
    ("CartPole-v1", "continuous", False, "double_dqn", False),
    ("MountainCar-v0", "continuous", False, "dqn", False),
    ("MountainCar-v0", "continuous", False, "double_dqn", False),
    ("Acrobot-v1", "continuous", False, "dqn", False),
    ("Acrobot-v1", "continuous", False, "double_dqn", False),
    ("LunarLander-v3", "continuous", False, "dqn", False),
    ("LunarLander-v3", "continuous", False, "double_dqn", False),
    # image observations, discrete actions
    ("ALE/Pong-v5", "image", False, "dqn_cnn", True),
    ("ALE/Pong-v5", "image", False, "double_dqn_cnn", True),
    ("ALE/Breakout-v5", "image", False, "dqn_cnn", True),
    ("ALE/Breakout-v5", "image", False, "double_dqn_cnn", True),
]

TRAINERS = {
    "dqn": train_dqn,
    "double_dqn": train_double_dqn,
    "dqn_cnn": train_dqn_cnn,
    "double_dqn_cnn": train_double_dqn_cnn,
}


def build_args(case, device):
    """Small but real training run sized per observation kind."""
    env_id, obs_kind, grid_env, trainer_key, _ = case
    common = {
        "env_id": env_id,
        "seed": 0,
        "total_timesteps": 300,
        "buffer_size": 2000,
        "batch_size": 32,
        "learning_starts": 100,
        "eval_every": 200,
        "save_every": 0,
        "num_eval_eps": 2,
        "use_wandb": False,
        "capture_video": False,
        "device": device,
    }
    if trainer_key.endswith("_cnn"):
        common.update(
            {
                "total_timesteps": 120,
                "learning_starts": 60,
                "buffer_size": 500,
                "batch_size": 8,
                "eval_every": 100,
                "num_eval_eps": 1,
                "target_network_frequency": 50,
            }
        )
    else:
        common.update(
            {
                "grid_env": grid_env,
                "atari_wrapper": False,
                "train_frequency": 4,
                "target_network_frequency": 100,
                "max_grad_norm": 1.0,
            }
        )
    return common


def check_env(case):
    """Return (obs_shape, n_actions) for the environment without training."""
    env_id, _, grid_env, trainer_key, _ = case
    from neatrl.dqn_cnn import make_env as cnn_make_env
    from neatrl.dqn_mlp import make_env as mlp_make_env

    if trainer_key.endswith("_cnn"):
        env = cnn_make_env(
            env_id, 0, 0, atari_wrapper=True, frame_stack=4, screen_size=84
        )()
    else:
        env = mlp_make_env(env_id, 0, 0, atari_wrapper=False, grid_env=grid_env)()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()
    return obs_shape, n_actions


def run_case(case, device):
    """Train, evaluate, and sanity-check one (env, trainer) pair."""
    env_id, obs_kind, grid_env, trainer_key, _ = case
    train_fn = TRAINERS[trainer_key]
    obs_shape, n_actions = check_env(case)

    evals = []
    net = train_fn(
        **build_args(case, device),
        eval_callback=lambda step, m: evals.append((step, m)),
    )

    # 1. A network came back and is usable.
    assert isinstance(net, torch.nn.Module), f"{trainer_key} returned {type(net)}"
    # 2. Evaluation actually ran.
    assert evals, f"{trainer_key} never evaluated on {env_id}"
    step, metrics = evals[0]
    assert isinstance(step, int)
    assert np.isfinite(metrics["avg_return"]), f"non-finite return on {env_id}"
    # 3. The callback exposes the live model (the overestimation demo needs it).
    assert isinstance(metrics["model"], torch.nn.Module)
    # 4. A forward pass on a real observation has the right action dimension.
    env_mod = (
        __import__("neatrl.dqn_cnn", fromlist=["make_env"])
        if trainer_key.endswith("_cnn")
        else __import__("neatrl.dqn_mlp", fromlist=["make_env"])
    )
    if trainer_key.endswith("_cnn"):
        env = env_mod.make_env(env_id, 0, 0, atari_wrapper=True)()
    else:
        env = env_mod.make_env(env_id, 0, 0, atari_wrapper=False, grid_env=grid_env)()
    obs, _ = env.reset()
    with torch.no_grad():
        if trainer_key.endswith("_cnn"):
            from neatrl.dqn_cnn import _obs_tensor

            batch = _obs_tensor(obs, device)
        else:
            batch = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(
                0
            )
        q = net(batch)
    env.close()
    assert q.shape == (1, n_actions), (
        f"{trainer_key}/{env_id}: {q.shape} != (1, {n_actions})"
    )
    assert torch.isfinite(q).all(), f"non-finite Q on {env_id}"

    return {
        "obs_shape": tuple(obs_shape),
        "obs_kind": obs_kind,
        "n_actions": n_actions,
        "evals": len(evals),
        "avg_return": float(metrics["avg_return"]),
    }


def check_target_differs():
    """The Double DQN target must differ from the plain max target in general."""
    from neatrl.double_dqn_mlp import QNet, double_dqn_target

    torch.manual_seed(0)
    online = QNet(4, 3)
    target = QNet(4, 3)  # deliberately different weights
    obs = torch.randn(64, 4)
    rewards = torch.zeros(64, 1)
    dones = torch.zeros(64, 1)

    double = double_dqn_target(online, target, obs, rewards, dones, 0.99)
    with torch.no_grad():
        plain = 0.99 * target(obs).max(dim=1)[0]
    differing = (double - plain).abs() > 1e-6
    assert differing.any(), "double target identical to plain max target"
    return f"{int(differing.sum())}/{len(differing)} transitions differ"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="substring filter on trainer")
    parser.add_argument("--env-only", default=None, help="substring filter on env id")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cases = CASES
    if args.only:
        cases = [c for c in cases if args.only in c[3]]
    if args.env_only:
        cases = [c for c in cases if args.env_only.lower() in c[0].lower()]

    print(f"device={args.device}  cases={len(cases)}\n")
    failures = []
    for i, case in enumerate(cases, 1):
        env_id, obs_kind, grid_env, trainer_key, _ = case
        label = f"{trainer_key:<18s} {env_id:<18s} {obs_kind:<10s}"
        print(f"[{i:>2}/{len(cases)}] RUN  {label}", flush=True)
        try:
            info = run_case(case, args.device)
            print(
                f"[{i:>2}/{len(cases)}] PASS {label} "
                f"obs={info['obs_shape']} acts={info['n_actions']} "
                f"evals={info['evals']} avg_return={info['avg_return']:.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[{i:>2}/{len(cases)}] FAIL {label} {type(exc).__name__}: {exc}")
            failures.append((case, traceback.format_exc()))

    print("\n=== target check ===")
    try:
        print(f"PASS {check_target_differs()}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {type(exc).__name__}: {exc}")
        failures.append((("target",), traceback.format_exc()))

    print(f"\n{len(cases) + 1 - len(failures)}/{len(cases) + 1} passed")
    if failures:
        print("\n=== failure details ===")
        for case, tb in failures:
            print(f"\n--- {case[0]} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
