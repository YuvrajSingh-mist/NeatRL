#!/usr/bin/env python3
"""Instrument Dueling DQN training to find the instability the paper alludes to.

Wang et al. (2016) say that plain ``Q = V + A`` has "poor practical performance"
because it is unidentifiable, but they publish **no** diagnostic: no gradient
norms, no V/A magnitude curves. This script measures those things directly, for
both aggregation rules, so the claim can be checked instead of taken on faith.

What is recorded every ``--log-every`` steps:

  optimisation   loss, gradient norm of the whole net and of each sub-network
                 (features / value stream / advantage stream)
  stream scale   mean and max |V|, mean and max |A| -- do they drift?
  output         mean Q, action spread (max - min over actions)
  cancellation   mean_a A, and |V| relative to |Q|

The interesting question is not "does V diverge to infinity" -- it does not --
but whether the unconstrained split lets V and A drift to *large, canceling*
values, which would raise gradient magnitudes and make optimisation noisier.

Usage:
    .venv/bin/python scripts/dueling_instability_probe.py
    .venv/bin/python scripts/dueling_instability_probe.py \\
        --env-id ALE/Breakout-v5 --steps 20000 --log-every 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.buffers import ReplayBuffer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dueling_qva_experiment as exp  # noqa: E402
from neatrl.dueling_dqn_mlp import DuelingQNet  # noqa: E402


def grad_norm(params) -> float:
    """L2 norm of the gradients currently held by ``params``."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


def stream_stats(net: DuelingQNet, probe: torch.Tensor):
    """Forward a probe batch and summarise the scale of V, A and Q."""
    with torch.no_grad():
        q, v, adv, _ = net(probe)
    v = v.squeeze(-1)
    return {
        "mean_V": float(v.mean()),
        "std_V": float(v.std()),
        "max_abs_V": float(v.abs().max()),
        "mean_A": float(adv.mean()),
        "mean_abs_A": float(adv.abs().mean()),
        "max_abs_A": float(adv.abs().max()),
        "std_A_over_actions": float(adv.std(dim=1).mean()),
        "mean_Q": float(q.mean()),
        "max_abs_Q": float(q.abs().max()),
        "action_spread": float((q.max(dim=1).values - q.min(dim=1).values).mean()),
    }


def train_probe(
    env_id: str,
    subtract_mean: bool,
    total_timesteps: int,
    seed: int,
    hp: dict,
    probe: torch.Tensor,
    log_every: int,
    atari: bool,
    device: torch.device,
) -> dict:
    """Same loop as the main harness, plus gradient and stream instrumentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = exp.make_env(env_id, seed, record_stats=True, atari=atari)
    obs, _ = env.reset(seed=seed)

    obs_shape = exp.obs_shape_of(env)
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

    series: dict[str, list] = {
        k: []
        for k in (
            "step",
            "loss",
            "grad_total",
            "grad_features",
            "grad_values",
            "grad_adv",
            "gout_v",
            "gout_a",
            "mean_V",
            "std_V",
            "max_abs_V",
            "mean_A",
            "mean_abs_A",
            "max_abs_A",
            "std_A_over_actions",
            "mean_Q",
            "max_abs_Q",
            "action_spread",
            "epsilon",
        )
    }
    returns: list[tuple[int, float]] = []
    t0 = time.time()

    for step in range(total_timesteps):
        eps = exp.linear_epsilon(
            step, hp["start_e"], hp["end_e"], total_timesteps, hp["exploration_fraction"]
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
            np.asarray(obs, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(action),
            np.asarray(reward, dtype=np.float32),
            np.asarray(done),
            [info],
        )
        if done:
            obs, _ = env.reset()
            if "episode" in info:
                returns.append((step, float(info["episode"]["r"])))
        else:
            obs = next_obs

        if step > hp["learning_starts"] and step % hp["train_frequency"] == 0:
            data = buffer.sample(hp["batch_size"])
            with torch.no_grad():
                target_q, _, _, _ = target(data.next_observations.to(device))
                td_target = data.rewards.flatten().to(device) + hp["gamma"] * (
                    target_q.max(1)[0]
                ) * (1 - data.dones.flatten().to(device))

            q_values, _, _, _ = online(data.observations.to(device))
            old_val = q_values.gather(1, data.actions.to(device)).squeeze()
            loss = torch.nn.functional.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()

            g_total = grad_norm(online.parameters())
            g_feat = grad_norm(online.features.parameters())
            g_val = grad_norm(online.values.parameters())
            g_adv = grad_norm(online.adv.parameters())

            torch.nn.utils.clip_grad_norm_(
                online.parameters(), max_norm=hp["max_grad_norm"]
            )
            optimizer.step()

            # Log after the update so the recorded loss matches the step taken.
            if step % log_every == 0:
                s = stream_stats(online, probe)
                # Gradient w.r.t. the V and A *outputs*, not the parameters.
                # V and A are detached and re-attached as leaves so autograd
                # gives the signal each stream receives before the shared trunk
                # mixes it. Under plain addition these are the same numbers -
                # that identity is what identifiability means here.
                with torch.enable_grad():
                    _, v_out, a_out, _ = online(data.observations.to(device))
                    v_leaf = v_out.detach().requires_grad_(True)
                    a_leaf = a_out.detach().requires_grad_(True)
                    a_used = (
                        a_leaf - a_leaf.mean(dim=1, keepdim=True)
                        if online.subtract_mean
                        else a_leaf
                    )
                    q_leaf = (v_leaf + a_used).gather(
                        1, data.actions.to(device)
                    ).squeeze()
                    out_loss = torch.nn.functional.mse_loss(q_leaf, td_target)
                    gout_v, gout_a = torch.autograd.grad(
                        out_loss, [v_leaf, a_leaf]
                    )
                series["step"].append(step)
                series["loss"].append(float(loss.item()))
                series["gout_v"].append(float(gout_v.norm()))
                series["gout_a"].append(float(gout_a.norm()))
                series["grad_total"].append(g_total)
                series["grad_features"].append(g_feat)
                series["grad_values"].append(g_val)
                series["grad_adv"].append(g_adv)
                series["epsilon"].append(float(eps))
                for k, v in s.items():
                    series[k].append(v)

        if step % hp["target_network_frequency"] == 0:
            for qp, tp in zip(online.parameters(), target.parameters()):
                tp.data.copy_(hp["tau"] * qp.data + (1.0 - hp["tau"]) * tp.data)

    env.close()
    series["wall_time"] = float(time.time() - t0)
    series["returns"] = returns
    return series


def plot_all(data: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    variants = [("standard", "Q = V + A - mean(A)"), ("no_mean", "Q = V + A")]

    panels = [
        ("grad_total", "Total gradient norm", "||grad|| (L2)"),
        ("gout_v", "Gradient into the V output", "||dL/dV|| (L2)"),
        ("gout_a", "Gradient into the A output", "||dL/dA|| (L2)"),
        ("max_abs_V", "Value stream magnitude", "max |V(s)|"),
        ("max_abs_A", "Advantage stream magnitude", "max |A(s,a)|"),
        ("mean_A", "Unconstrained mean advantage", "mean_a A"),
        ("loss", "TD loss", "MSE"),
        ("action_spread", "Action spread of Q", "max_a Q - min_a Q"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for variant, label in variants:
            d = data[variant]
            y = np.asarray(d[key], dtype=float)
            ax.plot(d["step"], y, label=label, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("environment step")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Instrumented training: gradients and V/A magnitudes, both rules")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="CliffWalking-v1")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--probe-states", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out-dir", default="runs/dueling_instability")
    args = p.parse_args()

    atari = args.env_id.startswith("ALE/")
    os.makedirs(args.out_dir, exist_ok=True)
    hp = dict(exp.HP)
    if args.buffer_size is not None:
        hp["buffer_size"] = args.buffer_size
    elif atari:
        hp["buffer_size"] = 10_000

    device = exp.resolve_device(args.device)
    states = exp.collect_shared_states(
        args.env_id, args.probe_states, seed=7, atari=atari
    )
    probe = torch.as_tensor(states, dtype=torch.float32).to(device)

    out = {}
    for variant, subtract in [("standard", True), ("no_mean", False)]:
        print(f"training {variant} ({args.steps} steps, log every {args.log_every})",
              flush=True)
        out[variant] = train_probe(
            args.env_id, subtract, args.steps, args.seed, hp, probe,
            args.log_every, atari, device,
        )
        d = out[variant]
        print(f"  final grad norm {d['grad_total'][-1]:.4g}, "
              f"max|V| {d['max_abs_V'][-1]:.4g}, "
              f"max|A| {d['max_abs_A'][-1]:.4g}", flush=True)

    out["env_id"] = args.env_id
    out["steps"] = args.steps
    out["device"] = device.type
    with open(os.path.join(args.out_dir, "instability.json"), "w") as f:
        json.dump(out, f)

    plot_all(out, os.path.join(args.out_dir, "instability.png"))

    print("\n" + "=" * 78)
    print(f"{'metric':>22} | {'Q=V+A-mean(A)':>22} | {'Q=V+A':>22}")
    print("=" * 78)
    for key in ("grad_total", "max_abs_V", "max_abs_A", "mean_A", "action_spread"):
        a = np.asarray(out["standard"][key], dtype=float)
        b = np.asarray(out["no_mean"][key], dtype=float)
        print(f"{key:>22} | {a.mean():>22.4g} | {b.mean():>22.4g}")
    print(f"\nWrote {args.out_dir}/instability.json and instability.png")


if __name__ == "__main__":
    main()
