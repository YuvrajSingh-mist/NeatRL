# Dueling DQN: what the advantage head and the mean subtraction actually do

An ablation on NeatRL's `DuelingQNet`: run the paper's formulation against a
variant with the `mean(A)` subtraction removed, then compare `Q`, `V` and `A`.

> **For the rigorous version of this argument, see [`PROOF.md`](PROOF.md).**
> It settles the question arithmetically (the gauge shift, the
> `V = mean_a Q` identity) and measures both on trained networks, including
> Atari. This document is the broader write-up with the learning curves.

**Short answer:** removing the mean subtraction changed nothing about the policy
or the learned `Q`, but it destroyed the *meaning* of `V` and `A`.

## The three quantities

| Symbol | Meaning |
|---|---|
| `Q(s,a)` | Expected return from state `s` taking action `a`. The thing DQN learns. |
| `V(s)` | Expected return from `s` under the current policy, no action chosen. |
| `A(s,a)` | `Q(s,a) - V(s)`. How much better `a` is than the state's baseline. |

## Why split into two heads at all

In most states the action barely matters - in Breakout with the ball far away,
almost any move is fine. A vanilla DQN relearns that "this state is safe" once
per action. The dueling head learns `V` once and spends the remaining capacity
only on the differences between actions.

That is the actual source of the paper's gains: **the two-stream architecture**,
not the subtraction.

## Why the paper subtracts the mean

`Q = V + A` is over-parameterised. Add `c` to `V` and subtract `c` from `A` and
`Q` is unchanged, so only the sum is pinned down - the split is free to drift.
This is a gauge freedom.

Subtracting `mean_a A` forces `mean_a A = 0`, which pins the split:

```
mean_a A(s,a) = 0   =>   V(s) = mean_a Q(s,a)
```

`V` becomes *identifiable* - it now means the literal average action value of
the state. Without it, `V` is only defined up to an arbitrary constant.

Note what the constraint does **not** do: it makes `Q = V + A` and
`Q = V + A - mean(A)` span the *same* set of functions. It is a
re-parameterisation, not added expressiveness. The loss is on `Q` only, so the
gradient on the composite is identical in both cases.

## What we changed

A single switch, defaulting to the paper's behaviour:

```python
class DuelingQNet(nn.Module):
    def __init__(self, state_space, action_space, subtract_mean=True):
        ...
        self.subtract_mean = subtract_mean

    def forward(self, x):
        ...
        if self.subtract_mean:
            q_values = values + adv - adv.mean(dim=1, keepdim=True)
        else:
            q_values = values + adv
        return q_values, values, adv, feat
```

Surfaced as `adv_mean_subtraction` on `train_dueling_dqn` and
`Config.adv_mean_subtraction`. See `README.md` in this folder for how to revert
it.

## Results

### CliffWalking-v1 (converged, 60k steps, seed 42)

| Metric | Standard | No mean subtraction |
|---|---|---|
| Eval return | **-13.0 ± 0.0** | **-13.0 ± 0.0** |
| `mean_abs(mean_A)` | 1.6e-07 | 3.49 |
| mean `V` | -20.58 | -19.00 |
| mean `Q` | -20.578 | -20.449 |
| std `Q` | 28.55 | 28.58 |

Both variants reach the optimal -13 return. The `Q` functions are effectively
identical: `corr(Q) = 0.99994`, mean absolute difference `0.22` against a `Q`
spread of `±28.5`.

The two rows that matter are `mean_abs(mean_A)` and mean `V`:

- Standard keeps `mean_A` at `1.6e-07` - the constraint is genuinely enforced,
  and `mean_V == mean_Q` to numerical precision, exactly as
  `V = mean_a Q` requires.
- Without subtraction `mean_A` drifts to `3.49`, and `V` moves by `~1.6`.
  `Q` did not move, so that `1.6` is pure gauge.

Correlation between the `V` shift and the mean advantage: **-0.998**. The drift
is almost perfectly explained by the unconstrained `mean_a A`, which is the
mechanism predicted above.

### Atari Breakout, ALE/Breakout-v5 (40k steps, Apple MPS)

| Metric | Standard | No mean subtraction |
|---|---|---|
| Eval return | 1.0 ± 0.0 | 3.3 ± 2.4 |
| `mean_abs(mean_A)` | 1.3e-10 | 0.255 |
| mean `V` | 0.687 | 0.851 |
| mean `Q` | 0.687 | 1.106 |
| `corr(Q)` | - | 0.779 |
| `corr(V shift, mean_A)` | - | 0.368 |

The gauge drift reproduces: `mean_A` is pinned at `~1e-10` in one variant and
`0.255` in the other, and `mean_V == mean_Q` again holds only for the standard
variant.

**Treat this run as weak evidence.** 40k steps does not converge Breakout - both
variants are near-random, the eval returns are within noise of each other, and
`corr(V shift, mean_A)` is only `0.37` rather than the `-0.998` seen on
CliffWalking. The mechanism is visible; the clean quantitative confirmation is
the CliffWalking run.

### Throughput

Same script, 1500 Atari steps each, freshly measured on an M4:

| Device | Training loop |
|---|---|
| CPU | 9.4 s |
| Apple MPS | 3.7 s |

Roughly **2.5x** on the loop. The larger `~3.7x` figure quoted in `README.md`
came from the full 40k-step run, where fixed per-episode cost is amortised
differently; it is not reproducible from the saved artifacts, so the measured
2.5x is the number to cite.

## What this means

1. **The advantage split is the win.** Learning `V` once instead of per-action is
   the architecture's contribution. That is what you would keep.
2. **The mean subtraction is a guard rail, not a learning signal.** It changes
   *what the two heads mean*, not how well the agent performs. It bought exactly
   zero return here.
3. **Without it, `V` and `A` are not individually interpretable.** You can still
   read the sum `Q`, but a statement like "this state is worth X" becomes
   meaningless, because `X` includes an arbitrary constant.
4. This matches the paper's own finding: the dueling *structure* produced the
   gains, while the centred-vs-uncentred comparison showed only a small
   difference. The subtraction exists for identifiability.

Practical rule: if you only care about the policy, the subtraction is optional.
If you want to inspect, debug, or plot `V` and `A`, keep it - otherwise you are
reading gauge noise.

## Plots

Generated per run into the output directory (local only, `runs/` is gitignored):

| File | Contents |
|---|---|
| `01_qva_decomposition.png` | `V`, `A`, `Q` per demo state, both variants |
| `02_training_gauge.png` | `V`, mean advantage, `Q` during training |
| `03_diagnostics.png` | learning curves, mean-advantage gauge, `Q`/`V` agreement |
| `04_trajectory.png` | `V`, `max Q`, chosen-action advantage along one episode |

## Reproducing

```bash
# CliffWalking: the clean, converged comparison
.venv/bin/python scripts/dueling_qva_experiment.py \
  --steps 60000 --states 2000 --episodes 10 --seed 42 \
  --out-dir runs/dueling_qva

# Atari: needs  uv pip install "gymnasium[atari]" ale-py
.venv/bin/python scripts/dueling_qva_experiment.py \
  --env-id ALE/Breakout-v5 --steps 40000 --states 500 --episodes 3 \
  --buffer-size 10000 --device mps --seed 42 \
  --out-dir runs/dueling_qva_atari_mps
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. `--buffer-size` matters:
NeatRL's default of 30000 holds two float32 `(4,84,84)` stacks, about 2.3 GB.

## Reverting the change

```bash
./scripts/dueling_qva_changes/revert.sh --dry-run
./scripts/dueling_qva_changes/revert.sh
```
