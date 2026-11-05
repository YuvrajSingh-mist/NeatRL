#!/usr/bin/env python3
"""Prove *why* Dueling DQN needs the mean-advantage subtraction.

This is not a training-curve argument. It settles the question arithmetically.

The claim under test
--------------------
Plain addition ``Q = V + A`` leaves ``V`` and ``A`` **unidentifiable**: for any
constant ``c``, the pair ``(V + c, A - c)`` produces a *bit-identical* ``Q``.
Since the loss is computed on ``Q``, the network has no way to prefer one
representative over another, so ``V`` and ``A`` are not well-defined numbers.

The mean subtraction ``Q = V + A - mean_a A`` **breaks** that symmetry: the
same shift now changes ``Q`` by exactly ``c``. One representative is pinned
(``mean_a A = 0``), so ``V`` becomes ``mean_a Q`` and is well-defined.

Two independent checks are run:

  A. Gauge shift. Shift ``V`` by ``c``, ``A`` by ``-c``, recompute ``Q`` under
     each aggregation rule. Plain addition: ``Q`` unchanged. Mean subtraction:
     ``Q`` moves by ``c``. This is the symmetry (and its destruction), measured.
     The greedy action is also checked to be unaffected by the shift, which is
     the paper's rank-preservation claim: the redundancy is policy-irrelevant,
     so removing it costs nothing in performance.
  B. Is V the state value? Since ``Q = V + A_eff``, we have
     ``V - mean_a Q = -mean_a A_eff``. So ``V = mean_a Q`` holds exactly when
     ``mean_a A_eff = 0`` -- which is precisely what the mean subtraction
     enforces. Measuring ``V - mean_a Q`` therefore tests the theorem directly,
     on trained networks, with no confounds from training dynamics.

Usage:
    .venv/bin/python scripts/dueling_gauge_proof.py
    .venv/bin/python scripts/dueling_gauge_proof.py \\
        --env-id ALE/Breakout-v5 --steps 40000 --seeds 2 --buffer-size 10000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dueling_qva_experiment as exp  # noqa: E402

# ── aggregation rules, written out explicitly ────────────────────────────────
# These are the only two lines that differ between the two architectures.


def aggregate(q_naive_v: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Plain addition: Q = V + A."""
    return q_naive_v + adv


def aggregate_mean(v: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """The paper's rule: Q = V + A - mean_a A."""
    return v + adv - adv.mean(axis=1, keepdims=True)


def shift(v: np.ndarray, adv: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Move ``c`` of value from the advantage stream into the value stream."""
    return v + c, adv - c


# ── A: the gauge-shift experiment ────────────────────────────────────────────


def gauge_shift_table(
    v: np.ndarray, adv: np.ndarray, shifts: list[float]
) -> list[dict]:
    """Recompute Q under both rules for each shift, and measure what moved."""
    rows = []
    q_mean_ref = aggregate_mean(v, adv)
    q_naive_ref = aggregate(v, adv)
    # The paper notes the shift preserves the relative rank of actions, so the
    # greedy policy is unaffected by which representative is chosen.
    greedy_ref = q_naive_ref.argmax(axis=1)

    for c in shifts:
        v2, adv2 = shift(v, adv, c)
        q_naive = aggregate(v2, adv2)
        q_mean = aggregate_mean(v2, adv2)
        rows.append(
            {
                "c": c,
                "naive_max_abs_dQ": float(np.abs(q_naive - q_naive_ref).max()),
                "mean_max_abs_dQ": float(np.abs(q_mean - q_mean_ref).max()),
                "naive_mean_adv_after": float(adv2.mean()),
                "mean_mean_adv_after": float(adv2.mean(axis=1).mean()),
                "naive_expected_dQ": 0.0,
                "mean_expected_dQ": abs(c),
                "naive_greedy_preserved": bool(
                    np.array_equal(q_naive.argmax(axis=1), greedy_ref)
                ),
                "mean_greedy_preserved": bool(
                    np.array_equal(q_mean.argmax(axis=1), greedy_ref)
                ),
            }
        )
    return rows


def hand_checkable_table(
    v: np.ndarray, adv: np.ndarray, state_idx: int, c: float
) -> dict:
    """One state, every number listed, so the arithmetic can be done by hand."""
    v1 = v[state_idx]
    a1 = adv[state_idx]
    v2, a2 = shift(v, adv, c)
    v2, a2 = v2[state_idx], a2[state_idx]

    q_naive = float((v1 + a1).max())
    q_naive_shifted = float((v2 + a2).max())
    q_mean = float((v1 + a1 - a1.mean()).max())
    q_mean_shifted = float((v2 + a2 - a2.mean()).max())

    return {
        "state_index": int(state_idx),
        "c": float(c),
        "V": [float(x) for x in np.atleast_1d(v1)],
        "A": [float(x) for x in a1],
        "mean_A": float(a1.mean()),
        "V_shifted": [float(x) for x in np.atleast_1d(v2)],
        "A_shifted": [float(x) for x in a2],
        "mean_A_shifted": float(a2.mean()),
        "naive_Q_before": q_naive,
        "naive_Q_after": q_naive_shifted,
        "naive_Q_changed_by": q_naive_shifted - q_naive,
        "mean_Q_before": q_mean,
        "mean_Q_after": q_mean_shifted,
        "mean_Q_changed_by": q_mean_shifted - q_mean,
    }


# ── B: is V actually the state value? ───────────────────────────────────────
#
# The theorem: mean subtraction is *equivalent* to V(s) = mean_a Q(s,a).
# Proof: Q = V + A  =>  mean_a Q = V + mean_a A, so V - mean_a Q = -mean_a A.
# Thus mean_a A = 0  <=>  V = mean_a Q. Nothing else is required.
#
# So the test is simply: measure V - mean_a Q. For the mean-subtracted net it
# must be zero; for the plain net it must equal -mean_a A, which is arbitrary.


def value_identity_stats(
    q: np.ndarray, v: np.ndarray, adv: np.ndarray, subtract_mean: bool
) -> dict:
    """Measure V - mean_a Q, and confirm it equals -mean_a A_effective.

    ``A_effective`` is the advantage stream as the variant actually uses it:
    the raw output for plain addition, and the mean-centred output for the
    paper's rule. Since Q = V + A_effective, we always have
    ``V - mean_a Q = -mean_a A_effective``; what differs is whether that
    quantity is zero.
    """
    mean_q = q.mean(axis=1)
    dev = v - mean_q
    adv_eff = adv - adv.mean(axis=1, keepdims=True) if subtract_mean else adv
    predicted = -adv_eff.mean(axis=1)
    return {
        "max_abs_dev": float(np.abs(dev).max()),
        "mean_abs_dev": float(np.abs(dev).mean()),
        "max_abs_error_vs_predicted": float(np.abs(dev - predicted).max()),
        "mean_abs_mean_adv_effective": float(np.abs(adv_eff.mean(axis=1)).mean()),
        "dev_matches_neg_mean_adv": bool(np.allclose(dev, predicted, atol=1e-5)),
    }


# ── plotting ─────────────────────────────────────────────────────────────────


def plot_gauge(rows_naive: list[dict], rows_mean: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cs = [r["c"] for r in rows_naive]
    ax.plot(
        cs,
        [r["naive_max_abs_dQ"] for r in rows_naive],
        "o-",
        label="Q = V + A  (unconstrained)",
        linewidth=2,
    )
    ax.plot(
        cs,
        [r["mean_max_abs_dQ"] for r in rows_mean],
        "s-",
        label="Q = V + A - mean(A)  (paper)",
        linewidth=2,
    )
    ax.plot(cs, [abs(c) for c in cs], "k--", alpha=0.5, label="identity |ΔQ| = |c|")
    ax.set_xlabel("shift c applied as V → V + c,  A → A - c")
    ax.set_ylabel("max |ΔQ| over all states and actions")
    ax.set_title("Plain addition is invariant to the shift; mean subtraction is not")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_identity(per_seed: dict[str, list[dict]], path: str) -> None:
    """Scatter V against mean_a Q. Points on the diagonal => V is the value."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, variant, title in [
        (axes[0], "no_mean", "Q = V + A"),
        (axes[1], "standard", "Q = V + A - mean(A)"),
    ]:
        d = per_seed[variant][0]
        mean_q = d["Q"].mean(axis=1)
        v = np.asarray(d["V"]).ravel()
        lims = [min(mean_q.min(), v.min()), max(mean_q.max(), v.max())]
        ax.plot(lims, lims, "k--", alpha=0.6, label="V = mean$_a$ Q")
        ax.scatter(mean_q, v, s=12, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("mean$_a$ Q(s,a)")
        ax.set_ylabel("V(s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        dev = np.abs(v - mean_q).mean()
        ax.text(
            0.03,
            0.95,
            f"mean |V - mean$_a$Q| = {dev:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
        )
    fig.suptitle(
        "Only the mean-subtracted rule makes V the state value (points on the diagonal)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="CliffWalking-v1")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--states", type=int, default=500)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out-dir", default="runs/dueling_gauge_proof")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reload cached per-seed Q/V/A from OUT_DIR/per_seed.npz instead of retraining.",
    )
    args = p.parse_args()

    atari = args.env_id.startswith("ALE/")
    os.makedirs(args.out_dir, exist_ok=True)

    hp = dict(exp.HP)
    if args.buffer_size is not None:
        hp["buffer_size"] = args.buffer_size
    elif atari:
        hp["buffer_size"] = 10_000

    device = exp.resolve_device(args.device)
    seeds = [1000 + i for i in range(args.seeds)]
    print(
        f"env={args.env_id} steps={args.steps} seeds={seeds} device={device.type}\n",
        flush=True,
    )

    shared = exp.collect_shared_states(args.env_id, args.states, seed=7, atari=atari)

    per_seed: dict[str, list[dict]] = {"standard": [], "no_mean": []}
    cache = os.path.join(args.out_dir, "per_seed.npz")

    if args.reuse and os.path.exists(cache):
        print(f"reusing cached runs from {cache}", flush=True)
        z = np.load(cache)
        for variant in ("standard", "no_mean"):
            for i, seed in enumerate(seeds):
                adv = z[f"{variant}_adv_{i}"]
                per_seed[variant].append(
                    {
                        "seed": seed,
                        "Q": z[f"{variant}_Q_{i}"],
                        "V": z[f"{variant}_V_{i}"],
                        "adv": adv,
                        "mean_adv": float(adv.mean()),
                    }
                )
    else:
        for variant, subtract in [("standard", True), ("no_mean", False)]:
            for seed in seeds:
                print(f"training {variant} seed={seed} ...", flush=True)
                net, _, _ = exp.train_variant(
                    args.env_id,
                    subtract,
                    args.steps,
                    seed,
                    hp,
                    atari=atari,
                    device=device,
                )
                q, v, adv = exp.qva_on(net, shared, device=device)
                per_seed[variant].append(
                    {
                        "seed": seed,
                        "Q": q,
                        "V": v,
                        "adv": adv,
                        "mean_adv": float(adv.mean()),
                    }
                )
                print(f"  mean_a A = {float(adv.mean()):+.4f}", flush=True)

        # Persist before any analysis/plotting, so a plotting error cannot
        # throw away the training run.
        flat = {}
        for variant, runs in per_seed.items():
            for i, d in enumerate(runs):
                flat[f"{variant}_Q_{i}"] = d["Q"]
                flat[f"{variant}_V_{i}"] = d["V"]
                flat[f"{variant}_adv_{i}"] = d["adv"]
        np.savez_compressed(cache, **flat)
        print(f"cached raw Q/V/A -> {cache}", flush=True)

    # ── A: gauge shift ──────────────────────────────────────────────────────
    # The invariance of plain addition (and the non-invariance of the mean
    # rule) holds for *any* (V, A) pair, so one trained net is enough. We run
    # it on both anyway to show the result is not net-specific.
    shifts = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    rows_mean = gauge_shift_table(
        per_seed["standard"][0]["V"].reshape(-1, 1),
        per_seed["standard"][0]["adv"],
        shifts,
    )
    rows_naive = gauge_shift_table(
        per_seed["no_mean"][0]["V"].reshape(-1, 1),
        per_seed["no_mean"][0]["adv"],
        shifts,
    )

    hand = hand_checkable_table(
        per_seed["no_mean"][0]["V"].reshape(-1, 1),
        per_seed["no_mean"][0]["adv"],
        state_idx=0,
        c=5.0,
    )

    identity_stats = {
        k: [
            value_identity_stats(d["Q"], d["V"], d["adv"], k == "standard")
            for d in runs
        ]
        for k, runs in per_seed.items()
    }

    plot_gauge(rows_naive, rows_mean, os.path.join(args.out_dir, "01_gauge_shift.png"))
    plot_identity(per_seed, os.path.join(args.out_dir, "02_v_is_state_value.png"))

    metrics = {
        "env_id": args.env_id,
        "steps": args.steps,
        "seeds": seeds,
        "device": device.type,
        "gauge_shift_naive": rows_naive,
        "gauge_shift_mean": rows_mean,
        "hand_checkable": hand,
        "value_identity": identity_stats,
    }
    with open(os.path.join(args.out_dir, "proof_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("A. GAUGE SHIFT  (V -> V + c,  A -> A - c)")
    print("=" * 72)
    print(f"\n{'c':>6} | {'Q=V+A  max|dQ|':>18} | {'Q=V+A-mean(A)  max|dQ|':>24}")
    for rn, rm in zip(rows_naive, rows_mean):
        print(
            f"{rn['c']:>6.1f} | {rn['naive_max_abs_dQ']:>18.3e} | "
            f"{rm['mean_max_abs_dQ']:>24.3e}"
        )
    print("\nPlain addition: Q does not move at all (~1e-6 is float32 noise).")
    print("Mean rule:      Q moves by exactly |c|. The shift is detectable,")
    print("                so the network is forced onto one representative.")
    print("\nGreedy action preserved under every shift?")
    print(f"  plain addition: {all(r['naive_greedy_preserved'] for r in rows_naive)}")
    print(f"  mean rule:      {all(r['mean_greedy_preserved'] for r in rows_mean)}")
    print("Both True: the gauge has no effect on the policy, so pinning it")
    print("costs nothing -- it only makes V and A readable.")

    print("\nB. IS V THE STATE VALUE?   V vs mean_a Q(s,a)")
    print("=" * 72)
    print(
        f"{'variant':>10} | {'seed':>6} | {'mean|V-mean_aQ|':>16} | "
        f"{'max|V-mean_aQ|':>15} | {'= -mean_a A ?':>13}"
    )
    for variant in ("standard", "no_mean"):
        for d, s in zip(per_seed[variant], identity_stats[variant]):
            print(
                f"{variant:>10} | {d['seed']:>6} | {s['mean_abs_dev']:>16.3e} | "
                f"{s['max_abs_dev']:>15.3e} | {str(s['dev_matches_neg_mean_adv']):>13}"
            )
    print("\nV = mean_a Q holds exactly when mean_a A = 0, i.e. only under the")
    print("mean subtraction. The last column confirms V - mean_a Q == -mean_a A.")

    print(
        f"\nWrote {args.out_dir}/proof_metrics.json, 01_gauge_shift.png,"
        " 02_v_is_state_value.png"
    )


if __name__ == "__main__":
    main()
