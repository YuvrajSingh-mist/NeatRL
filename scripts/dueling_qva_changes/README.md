# Reverting the Dueling-DQN changes

This folder exists so the changes made to NeatRL for the Q / V / A experiment
can be removed at any time, cleanly.

## The exact change set

Only two tracked files were edited. Both edits are additive and default to the
original behaviour.

| File | Change | Default behaviour |
|---|---|---|
| `src/neatrl/dueling_dqn_mlp.py` | New `subtract_mean` argument on `DuelingQNet`, new `adv_mean_subtraction` argument on `train_dueling_dqn`, new `Config.adv_mean_subtraction` field | `True` — identical to before |
| `tests/test_network_forward_pass.py` | Two extra tests covering both aggregation paths | Adds coverage only |

Nothing was removed or renamed, so an unmodified call site behaves exactly as
it did before the change.

## Revert (recommended)

```bash
./scripts/dueling_qva_changes/revert.sh --dry-run   # read the diff first
./scripts/dueling_qva_changes/revert.sh
```

The script only reverts those two files, refuses to run if they contain
changes unrelated to this experiment, and leaves
`scripts/dueling_qva_experiment.py` and this folder in place.

## Revert (manual)

```bash
# option A: discard the working-tree edits
git checkout -- src/neatrl/dueling_dqn_mlp.py tests/test_network_forward_pass.py

# option B: apply the inverse of the saved patch
git apply -R scripts/dueling_qva_changes/neatrl-dueling-changes.patch
```

If you committed the change instead, revert the commit:

```bash
git revert <commit-sha>
```

## Verify the revert

```bash
grep -c subtract_mean src/neatrl/dueling_dqn_mlp.py        # expect 0
grep -c adv_mean_subtraction src/neatrl/dueling_dqn_mlp.py # expect 0
git status --short                                          # expect only the new script files
```

## After reverting

`scripts/dueling_qva_experiment.py` builds its own `DuelingQNet` with
`subtract_mean=...`, so the ablation script stops working once the flag is
gone. The saved artifacts in `runs/dueling_qva*/` and the plots stay valid;
they were produced while the flag existed.

## Reproducing the runs

`runs/` is gitignored, so the artifacts are local only.

```bash
# CliffWalking-v1: converged, both variants reach the optimal -13
.venv/bin/python scripts/dueling_qva_experiment.py \
  --steps 60000 --states 2000 --episodes 10 --seed 42 \
  --out-dir runs/dueling_qva

# Atari: needs the optional extra
#   uv pip install "gymnasium[atari]" ale-py
.venv/bin/python scripts/dueling_qva_experiment.py \
  --env-id ALE/Breakout-v5 --steps 40000 --states 500 --episodes 3 \
  --buffer-size 10000 --device mps --seed 42 \
  --out-dir runs/dueling_qva_atari_mps
```

`--device auto|cpu|cuda|mps` defaults to `auto`, which prefers CUDA, then Apple
MPS, then CPU. This matters because NeatRL's own `Config.device` only ever
resolves to `cuda` or `cpu`, so on Apple silicon the GPU is never used unless
asked for. On M4, MPS was ~3.7x faster end-to-end for Atari (215 vs 58 steps/s).

`--buffer-size` matters on Atari: NeatRL's default of 30000 holds two float32
`(4,84,84)` stacks, about 2.3 GB, so the Atari runs default to 10000 unless
overridden.

Each run writes:

| File | Contents |
|---|---|
| `01_qva_decomposition.png` | `V`, `A` and `Q` per demo state, both variants |
| `02_training_gauge.png` | `V`, mean advantage and `Q` during training |
| `03_diagnostics.png` | learning curves, mean-advantage gauge, `Q`/`V` agreement |
| `04_trajectory.png` | `V`, `max Q` and chosen-action advantage along one episode |
| `metrics.json` | all scalars, including training-time probe series |
| `qva_shared_states.npz` | raw `Q`/`V`/`A` on the shared state set |
| `dueling_standard.pt`, `dueling_no_mean.pt` | trained weights |

