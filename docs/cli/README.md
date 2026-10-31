# The `neatrl` command line

`neatrl train <algo> [env_id]` trains one algorithm in one environment, in a
single command, with a live terminal dashboard. Everything else — the trainers,
the networks, the math — is reachable from Python, but that is a separate
document: see [`docs/non-cli/`](../non-cli/).

```console
$ pip install neatrl
$ neatrl train dqn CartPole-v1
```

```
│  NeatRL 2.5.4   CPU: 24.2%     GPU: 99.0%     RAM: 50.0%     MPS: 0.5M       │
│     Algorithm       DQN                td_loss               0.0000          │
```

---

## Install

| | |
|---|---|
| **Base** | `pip install neatrl` — the CLI, the dashboard, all classic-control environments |
| **Everything** | `pip install "neatrl[all]"` — adds Atari (ALE), Box2D and MuJoCo |
| **Isolated** | `uv tool install neatrl` (or `pipx install neatrl`), then `neatrl` is on `PATH` |
| **Upgrade** | `pip install --upgrade neatrl` / `uv tool upgrade neatrl` |
| **Version** | `neatrl --version` |

Environment families that need a heavy native package stay opt-in:

| Extra | Installs | Unlocks |
|---|---|---|
| `neatrl[atari]` (alias `[ale]`) | `gymnasium[atari]`, `ale-py` | `ALE/Pong-v5`, `ALE/Breakout-v5`, … |
| `neatrl[box2d]` | `gymnasium[box2d]` | `CarRacing-v3`, `LunarLander-v3` |
| `neatrl[mujoco]` | `gymnasium[mujoco]` | `HalfCheetah-v4`, `Ant-v5`, `Walker2d-v5` |

`neatrl[cli]` and `neatrl[classic]` still resolve, but they install nothing you
do not already get: `rich`, `psutil`, `nvidia-ml-py` and
`gymnasium[classic-control]` ship in the base install.

---

## Commands

```text
usage: neatrl [-h] [--version] command ...

  train     Train an agent
  --version show program's version number and exit

usage: neatrl train [-h] [--steps N] [--wandb] algo [env_id]
```

| Invocation | Effect |
|---|---|
| `neatrl` | Print help, exit 0 |
| `neatrl --version` | Print `neatrl <version>`, exit 0 |
| `neatrl train <algo>` | Train with the alias's default environment |
| `neatrl train <algo> <env_id>` | Train in the environment you name |
| `neatrl train --help` | List every alias with its default flags |

### Options

| Option | Meaning |
|---|---|
| `--steps N` | Training steps, or **episodes** for `reinforce` / `reinforce-cnn`, which count whole rollouts rather than individual transitions. Without it, the trainer's own default applies. |
| `--wandb` | Log the run to Weights & Biases. **Off by default** — a terminal command should not start an upload unless asked. |

---

## Algorithms

21 aliases over 11 algorithms. The `-cnn` alias is the same algorithm reading
pixels instead of a flat observation vector. Each alias links to its own page,
which lists its default environment and the commands that run it.

| Algorithm | Aliases | Default environment | Observation | Actions | Trainer |
|---|---|---|---|---|---|
| [Q-learning](q_learning.md) | [`q-learning`](q_learning.md) | `FrozenLake-v1` | tabular | discrete | `neatrl.q_learning.train_q_learning` |
| [DQN](dqn.md) | [`dqn`](dqn.md) · [`dqn-cnn`](dqn.md) | `CartPole-v1` · `ALE/Pong-v5` | vector · Atari | discrete | `neatrl.dqn_mlp.train_dqn` · `neatrl.dqn_cnn.train_dqn_cnn` |
| [Double DQN](double_dqn.md) | [`double-dqn`](double_dqn.md) · [`double-dqn-cnn`](double_dqn.md) | `CartPole-v1` · `ALE/Pong-v5` | vector · Atari | discrete | `neatrl.double_dqn_mlp.train_double_dqn` · `neatrl.double_dqn_cnn.train_double_dqn_cnn` |
| [Dueling DQN](dueling_dqn.md) | [`dueling-dqn`](dueling_dqn.md) · [`dueling-dqn-cnn`](dueling_dqn.md) | `CliffWalking-v1` · `ALE/Pong-v5` | vector · Atari | discrete | `neatrl.dueling_dqn_mlp.train_dueling_dqn` (both; `atari_wrapper=True` for CNN) |
| [REINFORCE](reinforce.md) | [`reinforce`](reinforce.md) · [`reinforce-cnn`](reinforce.md) | `CartPole-v1` · `ALE/Pong-v5` | vector · Atari | discrete | `neatrl.reinforce_mlp.train_reinforce` · `neatrl.reinforce_cnn.train_reinforce_cnn` |
| [A2C](a2c.md) | [`a2c`](a2c.md) · [`a2c-cnn`](a2c.md) | `CartPole-v1` · `ALE/Pong-v5` | vector · Atari | discrete | `neatrl.a2c_mlp.train_a2c` · `neatrl.a2c_cnn.train_a2c_cnn` |
| [PPO](ppo.md) | [`ppo`](ppo.md) · [`ppo-cnn`](ppo.md) | `CartPole-v1` · `ALE/Pong-v5` | vector · Atari | discrete + continuous | `neatrl.ppo_mlp.train_ppo` · `neatrl.ppo_cnn.train_ppo_cnn` |
| [RND](rnd.md) | [`rnd`](rnd.md) · [`rnd-cnn`](rnd.md) | `CliffWalking-v1` · `ALE/Pong-v5` | vector · Atari | discrete + continuous | `neatrl.rnd_mlp.train_ppo_rnd` · `neatrl.rnd_cnn.train_ppo_rnd_cnn` |
| [DDPG](ddpg.md) | [`ddpg`](ddpg.md) · [`ddpg-cnn`](ddpg.md) | `Pendulum-v1` · `CarRacing-v3` | vector · Box2D | continuous | `neatrl.ddpg_mlp.train_ddpg` · `neatrl.ddpg_cnn.train_ddpg_cnn` |
| [TD3](td3.md) | [`td3`](td3.md) · [`td3-cnn`](td3.md) | `Pendulum-v1` · `CarRacing-v3` | vector · Box2D | continuous | `neatrl.td3_mlp.train_td3` · `neatrl.td3_cnn.train_td3_cnn` |
| [SAC](sac.md) | [`sac`](sac.md) · [`sac-cnn`](sac.md) | `Pendulum-v1` · `CarRacing-v3` | vector · Box2D | continuous | `neatrl.sac_mlp.train_sac` · `neatrl.sac_cnn.train_sac_cnn` |

`ppo` and `rnd` read the action space from the environment: a `Discrete` space
gets a categorical head, a `Box` space a Gaussian one, so `Pendulum-v1` and
`CartPole-v1` both run with no extra flags. Everything else is fixed — the
DQN family, REINFORCE and A2C are discrete-only, DDPG, TD3 and SAC are
continuous-only. `q-learning` is tabular and has no CNN sibling, so it only
accepts discrete environments with small observation spaces.

---

## Environment routing

The alias and the environment have to agree on what a single observation
*looks* like. Rather than failing deep inside torch with a shape error, the
CLI checks first:

**1. An image environment named on a plain alias switches to the CNN trainer**
and says so on stderr:

```console
$ neatrl train dqn ALE/Pong-v5
note: ALE/Pong-v5 returns images, so this run uses the CNN trainer (dqn-cnn).
```

**2. A `-cnn` alias given a vector environment is an error** (exit 2):

```console
$ neatrl train dqn-cnn CartPole-v1
neatrl: error: dqn-cnn expects atari image observations, but CartPole-v1 is a
vector environment. Use `neatrl train dqn CartPole-v1` instead.
```

**3. An image family with no matching preprocessing is an error** that names
the aliases that do support it:

```console
$ neatrl train dqn CarRacing-v3
neatrl: error: CarRacing-v3 needs carracing image preprocessing, which dqn-cnn
does not apply (ALE/Pong-v5 is its default). Try `neatrl train ddpg-cnn
CarRacing-v3`, `neatrl train sac-cnn CarRacing-v3`, `neatrl train td3-cnn
CarRacing-v3`.
```

In short: **Atari environments go to DQN-family, A2C, PPO, REINFORCE and RND;
CarRacing goes to DDPG, TD3 and SAC.**

---

## Missing packages

If the environment's dependency is absent, the real traceback is printed — it
is the ground truth and hiding it would make genuine bugs harder to read — and
the last line names the install:

```console
$ neatrl train dqn-cnn
NamespaceNotFound: Namespace ALE not found. Have you installed the proper package for ALE?
...
neatrl: this environment needs the 'ale-py' package, which is not installed.
neatrl: install the 'atari' extra to enable Atari games such as ALE/Pong-v5 and ALE/Breakout-v5:

    pip install "neatrl[atari]"

neatrl: with uv tool: uv tool install --force "neatrl[atari]"
```

This path exits **1** (as opposed to **2** for a usage mistake).

---

## The dashboard

The TUI replaces itself in place; it is not a scrolling log.

| Key | Action |
|---|---|
| `l` | Toggle the log view (last 500 lines) |
| `↑` / `↓` | Scroll the log view |
| `c` | Toggle the config view (every field of the run's dataclass) |
| `v` | Toggle the charts view |
| `l` / `c` / `v` again | Return to the dashboard |

The dashboard keeps the summary, the loss values and a progress bar. Every
time-series metric lives behind `v`, as a two-column grid: episode return and
length, training loss, steps per second, then whatever the algorithm reports
(`epsilon`, `q_mean`, `entropy`, `approx_kl`, …). A metric a trainer starts
pushing gets a chart without any registration — the label table only supplies
a nicer name and colour for the keys it recognises.

While the dashboard is up it suppresses `tqdm`: two things writing to one
terminal is what makes a progress bar flicker against a panel that repaints
itself. The bar's numbers are in the frame instead, and `--steps` progress is
the `Progress` row.

Header fields:

| Field | What it measures |
|---|---|
| `CPU` | System-wide CPU utilization, all cores |
| `GPU` | Utilization of the compute engine, not memory |
| `RAM` | Physical memory in use — "how full is the machine" |
| `MPS` / `VRAM` | Memory held by the GPU side |

GPU figures by platform, because the three do not expose the same counters:

| Platform | `GPU` | GPU memory |
|---|---|---|
| Apple Silicon | `ioreg` device utilization, sampled at 1 s | `MPS: 240M` — absolute bytes from `torch.mps.driver_allocated_memory()` |
| NVIDIA (Linux/Windows) | NVML, handle initialised once per process | `VRAM: 18.4%` of the card's total |
| AMD (ROCm, Linux) | `amdsmi` | `VRAM: …%` |
| None available | `GPU: n/a` | `VRAM: n/a` |

On Apple Silicon the display is an absolute size rather than a percentage
because `driver_allocated_memory()` is bounded by PyTorch's *soft* MPS budget
(~74% of RAM), so a percentage sat next to a RAM percentage with a different
denominator and read as misleadingly small. `GPU` is the number that actually
moves during training.

The dashboard only appears on an interactive terminal. Piped or redirected
output is plain, scrolling text, with no escape sequences.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Run finished, help printed, or `--version` |
| `1` | The environment's optional dependency is missing (a hint is printed) |
| `2` | Usage error: unknown alias, or an alias/environment mismatch |

Ctrl-C interrupts training; the cursor and terminal modes are restored on the
way out, including when the run dies on an exception.

---

## Recipes

```bash
# Smoke test: a few thousand steps to confirm the install works
neatrl train dqn CartPole-v1 --steps 5000

# Longer CartPole run, no environment argument (its default)
neatrl train ppo --steps 200000

# Atari, and the same thing spelled out
neatrl train dqn-cnn
neatrl train dqn-cnn ALE/Breakout-v5

# Continuous control from pixels
neatrl train td3-cnn CarRacing-v3 --steps 100000

# Tabular, with a fixed exploration schedule
neatrl train q-learning FrozenLake-v1

# Exploration bonus on a sparse-reward grid
neatrl train rnd CliffWalking-v1

# Track the run
neatrl train sac Pendulum-v1 --wandb
```

Back-to-back runs, one alias at a time:

```bash
for algo in dqn double-dqn dueling-dqn; do
  neatrl train "$algo" CartPole-v1 --steps 200000
done
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Namespace ALE not found` | `pip install "neatrl[atari]"` |
| `box2d is not installed` | `pip install "neatrl[box2d]"` |
| `mujoco is not installed` | `pip install "neatrl[mujoco]"` |
| `mat1 and mat2 shapes cannot be multiplied` | An image environment reached an MLP trainer — name the `-cnn` alias, or let the routing switch for you |
| No dashboard, just `tqdm` | stdout is not a TTY, or `rich` failed to import (the CLI says so on stderr) |
| `argument algo: invalid choice` | The alias is misspelled; `neatrl train --help` lists all 21 |
| `neatrl --version` reports an old number | Another install shadows this one: `which -a neatrl` |

---

## Where this lives

| Path | Role |
|---|---|
| `src/neatrl/cli/main.py` | Argument parsing, exit codes, error routing |
| `src/neatrl/cli/recipes.py` | The alias table: trainer, default environment, observation family |
| `src/neatrl/cli/dashboard.py` | The Rich terminal dashboard |
| `src/neatrl/cli/extras.py` | Missing-package detection and install hints |
| `src/neatrl/cli/charts.py` | Sparklines and metric bars |
| `docs/cli/<algo>.md` | One page per algorithm: defaults, runnable commands, algorithm guide link |

The CLI is a thin layer: every alias dispatches to a public `train_*` function
in `neatrl`, the same one the Python API exposes. For programmatic control —
custom networks, callbacks, your own training loop — read
[`docs/non-cli/`](../non-cli/) instead.
