# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2.5.5] - 2026-10-31

### Fixed
- **PPO and PPO-RND train continuous action spaces with the built-in actor.**
  `ActorNet` in `ppo_mlp`, `ppo_cnn`, `rnd_mlp` and `rnd_cnn` takes the action
  space into account: a `Discrete` space keeps the categorical head, a `Box`
  space gets a Normal head with a learnable `log_std` (the same actor as
  `docs/cli/ppo.py` and the old `rnd.py`). `train_ppo(env_id="Pendulum-v1")`
  used to die inside gymnasium with `IndexError: invalid index to scalar
  variable` after the first rollout; it now trains, as do `ppo-cnn` and
  `rnd-cnn` on `CarRacing-v3`. Discrete runs are unchanged -- `continuous`
  defaults to False, so two-argument callers (including custom actor classes
  and the docs scripts) keep working.
- **`rnd_cnn` evaluation on a continuous environment** passed a `(1, action_dim)`
  array to `env.step()`, which CarRacing reads as a single action and then
  fails on `action[1]`. It now squeezes the batch dimension like the other
  trainers.

## [2.5.4] - 2026-10-31

### Added
- **`neatrl --version`** prints the installed version. Checking that an upgrade
  took effect used to mean launching a training run.

## [2.5.3] - 2026-10-31

### Added
- **`./scripts/publish.sh` (or `make publish`) releases to PyPI in one step**:
  it runs the tests, builds, and uploads using a token from `.env` (gitignored)
  -- never from the command line, where it would land in shell history.
  `make release V=2.6.0` also bumps `pyproject.toml` and opens a CHANGELOG
  entry, and `make upgrade` refreshes the local tool install.

### Fixed
- **The terminal cursor blinked over the dashboard.** Every frame homes to the
  origin and rewrites ~20 rows, so the terminal's own cursor was visible
  wherever the write had reached -- and over SSH, where a frame arrives in
  chunks, it appeared to wander around the panel. It is now hidden for the
  duration of the run and restored by `close()` and by an `atexit` handler, so
  a crash or Ctrl-C cannot leave the shell without a cursor.

### Changed
- **The dashboard imports and runs on Windows and Linux.** `termios` and `tty`
  were imported unconditionally, which made the whole package unimportable on
  Windows; they are now an optional POSIX capability, with `msvcrt` handling
  key input (arrow keys normalised to the same escape sequences). On a Windows
  console ANSI output is enabled via
  `ENABLE_VIRTUAL_TERMINAL_PROCESSING`; if that fails, or stdout is not a tty,
  the frame is printed once as plain text instead of leaking escape sequences.
- **GPU metrics no longer assume one platform.** `ioreg` is only probed on
  macOS, NVML is initialised once per process instead of once per frame, and
  ROCm hosts (where `torch.cuda` is true but NVML does not exist) fall back to
  `amdsmi`. A host with no utilization counter now reads `GPU: n/a` and
  `VRAM: n/a` instead of a misleading `0.0%`.

## [2.5.2] - 2026-10-30

### Changed
- **On Apple Silicon the GPU memory cell is an absolute size, not a
  percentage.** `VRAM: 8.5%` was the share of `driver_allocated_memory()` over
  `recommended_max_memory()` -- a soft budget worth ~74% of RAM, so the number
  sat next to a RAM percentage with a different denominator and read as
  surprisingly small (a 74M-parameter net at 100% GPU load only reaches 18%).
  It now reads `MPS: 240M` / `MPS: 2.2G`. GPU utilization was already the
  metric that tracks load and still is: 2-3% idle, 97-100% under a large net.
- **The algorithm name moved from the dashboard header to the Summary panel.**
  Names like `Dueling-DQN` and `DQN-CNN` did not fit the header row at 80
  columns and were silently cut off.

### Fixed
- **`neatrl train td3-cnn` crashed on exit** with
  `UnboundLocalError: latest_avg_return` when the run ended before the first
  evaluation; the closing dashboard message reads that value unconditionally.
- **`NeatRL-Website` is no longer tracked by this repository.** It was a
  gitlink with no `.gitmodules` pointing at a separate repo; the directory
  stays on disk and is gitignored.

## [2.5.1] - 2026-10-30

### Changed
- **The dashboard header says `RAM` instead of `DRAM`.** The figure was always
  correct -- `psutil.virtual_memory()` percent, i.e. system memory in use, not
  storage -- but "DRAM" beside a percentage reads like disk to anyone who is
  not thinking of the memory chips. Nothing else changed; VRAM still reports
  GPU-side allocation.

## [2.5.0] - 2026-10-30

### Changed
- **`pip install neatrl` is now enough to train.** The terminal dashboard
  (`rich`, `psutil`, `nvidia-ml-py`) and classic-control environments
  (CartPole, Pendulum, Acrobot, MountainCar) moved from extras into the base
  dependencies. `neatrl[cli]` and `neatrl[classic]` still resolve, so existing
  instructions keep working, but nothing extra is needed for the default
  `neatrl train <algo>` aliases.
- **The `box2d` extra no longer pins `swig`.** gymnasium already requests it
  only where it is needed (Python 3.14); on 3.10–3.13 the `box2d` wheels make
  it unnecessary, so the install no longer risks a source build.

### Added
- **`neatrl[ale]`** as an alias for `neatrl[atari]`, spelled after the
  namespace `ale-py` registers.
- **A missing environment dependency now tells you how to fix it.** Running a
  `-cnn` alias without `ale-py` prints gymnasium's own error unchanged --
  `NamespaceNotFound: Namespace ALE not found. Have you installed the proper
  package for ALE?` -- followed by the command that installs it:

  ```
  neatrl: this environment needs the 'ale-py' package, which is not installed.
  neatrl: install the 'atari' extra to enable Atari games such as ALE/Pong-v5:
      pip install "neatrl[atari]"
  ```

  The same treatment covers Box2D and MuJoCo. Unrelated failures still raise
  normally, so nothing is swallowed.

## [2.4.0] - 2026-10-30

### Added
- **An image environment picks the CNN trainer for you.**
  `neatrl train dqn ALE/Pong-v5` now runs the DQN CNN trainer and says so,
  instead of dying inside torch with `mat1 and mat2 shapes cannot be
  multiplied (33600x3 and 210x256)`. Each recipe records the observation
  family its trainer preprocesses for (`vector`, `atari`, `carracing`), so the
  swap only happens where the training code genuinely matches.
- **Mismatches are one-line errors, not tracebacks.** A CarRacing env given to
  `ppo-cnn` or an Atari env given to `td3-cnn` now explains which alias does
  support it; a vector env given to a `-cnn` alias points back at the MLP one;
  and an image env given to a tabular algorithm (`q-learning`) says it has no
  CNN trainer at all.

## [2.3.0] - 2026-10-30

### Added
- **`neatrl train dueling-dqn-cnn`.** The dueling trainer already had an
  `atari_wrapper` image path, but no alias exposed it, so the recipe table
  silently skipped the CNN variant. `DuelingQNet` now builds a conv trunk from
  the full observation shape instead of a single `nn.Linear` sized by
  `shape[0]`, which would have fed a `(4, 84, 84)` frame to a 4-input layer.
- **`neatrl train --wandb`.** Logging to Weights & Biases is now opt-in.

### Fixed
- **Every recipe default env is registered again.** `CliffWalking-v0` and
  `Taxi-v3` are retired in current gymnasium, so `neatrl train dueling-dqn` and
  `neatrl train rnd` died with `DeprecatedEnv` before the first step. All
  references moved to `CliffWalking-v1` / `Taxi-v4`.
- **`neatrl train <algo>` no longer uploads to wandb.** The trainers default
  `use_wandb=True`, which the CLI inherited, so a local `neatrl train dqn`
  quietly created a run in the user's account. Recipes now pass
  `use_wandb=False` unless `--wandb` is given.
- **The headless video-capture crash is gone from the CLI path.** Trainers
  render `rgb_array` evaluation video when wandb is on, and that calls
  `pygame.display.init()`, which raises on any machine without a display
  server. Since the CLI no longer enables wandb, `neatrl train sac` and
  friends complete in CI and over SSH.

### Docs
- **`neatrl train` examples match the real command.** They used `--no-wandb`,
  `--atari`, `--grid-env` and `--timesteps`, none of which exist, plus retired
  env ids (`BreakoutNoFrameskip-v4`, `cartpole`, `lunar`) and the obsolete
  `--no-wandb` workaround that this release makes unnecessary.

### Changed
- **Requires Python 3.10+ and `gymnasium>=1.3.0`.** The current environment ids
  (`CliffWalking-v1`, `Taxi-v4`) only exist from gymnasium 1.3, and 1.3 dropped
  Python 3.9. Installing on 3.9 now fails at resolve time instead of at
  `gym.make`.

## [2.2.0] - 2026-10-30

### Added
- **`neatrl[all]` extra.** One install covers everything the `train` aliases
  reach: the terminal dashboard (`rich`, `psutil`, `nvidia-ml-py`), Atari
  (`ale-py` + ROMs), Box2D, classic control and MuJoCo. The image-based
  `-cnn` aliases default to `ALE/Pong-v5` and `CarRacing-v3`, so on a plain
  `pip install neatrl` they could not run at all. `uv tool install
  "neatrl[all]"` is now the documented setup; the individual extras remain.

## [2.1.4] - 2026-10-30

### Fixed
- **The dashboard reported 0% GPU on Apple Silicon.** The MPS branch set
  `gpu = 0.0` unconditionally, so the figure was never real. macOS has no NVML
  equivalent and `powermetrics` needs root, but `ioreg` publishes the Metal
  driver's own `Device Utilization %` unprivileged; that is now read (and
  cached for a second, since it spawns a process). It sits near 0% for small
  MLP workloads because the GPU really is idle, and reaches 99% under load.
- **"VRAM" showed the Python process RSS, not GPU memory.** It was
  `Process().memory_info().rss / virtual_memory().total`, which is ordinary
  system memory. It now reports the share of the MPS memory budget the driver
  has allocated (`driver_allocated_memory()` vs `recommended_max_memory()`).

### Note
- On Apple Silicon the GPU shares system RAM, so DRAM and VRAM describe the
  same physical pool from different sides: DRAM is overall unified-memory
  pressure, VRAM is how much of PyTorch's MPS budget is in use.

## [2.1.3] - 2026-10-30

### Fixed
- **The dashboard printed a new panel for every frame instead of redrawing
  one.** Frames ended up stacked down the scrollback. It repositioned with a
  relative cursor-up (`ESC[nA`) measured from the previous frame, which breaks
  the moment the terminal scrolls — a frame taller than the window, or a line
  printed by another library such as `wandb`, shifts the origin permanently and
  every later frame lands lower. The dashboard now homes the cursor and
  overwrites the screen, so it behaves like a single live panel again.
- The frame is trimmed to the terminal height (keeping its closing border) and
  each row is cleared to end-of-line, so a frame can no longer scroll the
  screen or leave fragments of a wider frame behind.
- The panel is sized to the real terminal width instead of being capped at 80
  columns, and a pseudo-terminal reporting 0 rows no longer collapses it.

## [2.1.2] - 2026-10-30

### Added
- `neatrl train` now prints a one-line note when it has to skip the dashboard
  because `rich` is not installed, naming the fix
  (`pip install "neatrl[cli]"`). Previously the TUI just did not appear, with
  nothing to explain why.

### Note
- Install with the `cli` extra to keep the dashboard:
  `uv tool install --force "neatrl[cli]"`. `uv` records the extra, so later
  `uv tool upgrade neatrl` runs keep it.

## [2.1.1] - 2026-10-30

### Fixed
- **The dashboard printed its styling as literal text.** Section headers
  showed up as `[bold cyan]── DQN Metrics ──[/bold cyan]`, sparklines as
  `[magenta]▄[/magenta]`, and buffer bars as `[green][/green][dim]░░░░[/dim]`.
  `charts.py` returns Rich markup, but the dashboard fed those strings through
  `Text.append()`, which does not parse markup. It now uses `Text.from_markup`,
  so headings and chart colours render as intended.

### Note
- The dashboard needs `rich`, which is the optional `cli` extra:
  `uv tool install "neatrl[cli]"` or `pip install "neatrl[cli]"`. Without it,
  training runs with plain logging instead of the TUI.

## [2.1.0] - 2026-10-30

Every algorithm was run through `neatrl train` on two environments each (40
runs). Seventeen of the first 40 failed; this release fixes all of them.

### Fixed
- **`neatrl train dueling-dqn` hung forever on its own default environment.**
  `evaluate()` looped until the episode ended, but `CliffWalking-v0` is
  registered without a `TimeLimit`. Evaluation now stops at the env's
  `max_episode_steps` (10,000 if unset) in all ten trainers that were missing
  the cap: `dueling_dqn_mlp`, `q_learning`, `ddpg_mlp`, `ddpg_cnn`,
  `sac_mlp`, `sac_cnn`, `td3_mlp`, `td3_cnn`, `reinforce_mlp`,
  `reinforce_cnn`.
- **`rnd` and `rnd-cnn` crashed on the first PPO update** with
  `TypeError: get_action() takes 2 positional arguments but 3 were given`.
  Internal callers still passed the action to score, which 1.0.0 removed; they
  now score the stored actions against the returned distribution.
- **`dueling-dqn` crashed with `IndexError: Dimension out of range`** as soon
  as training left the random-action phase, because the Q-network was called
  with one observation that had no batch dimension.
- **`grid_env` no longer has to match the environment.** Discrete observation
  spaces are one-hot encoded automatically, so
  `neatrl train dueling-dqn CartPole-v1` and `... rnd CartPole-v1` work, and
  `grid_env=True` on a `Box` env no longer raises
  `AttributeError: 'Box' object has no attribute 'n'`.
- **`ddpg-cnn` and `td3-cnn` were silently training MLPs** — their default
  networks were `ActorNet`/`QNet` imported from the MLP modules. They now
  default to `ActorNetCNN`/`QNetCNN`.
- **`td3-cnn` used its 512-dim hidden state as the action**; the output layer
  was constructed but never applied.
- **`ddpg-cnn`, `sac-cnn` and `td3-cnn` failed on `CarRacing-v3` (96x96)**
  because the conv trunk hard-coded the 84x84 Atari flattened size and
  assumed channel-first input. A shared trunk in `neatrl.utils.nn_utils` now
  measures its own output size and accepts both HWC and CHW frames.
- **`a2c-cnn`, `ppo-cnn`, `reinforce-cnn` and `rnd-cnn` could not start at
  all** — their networks were `nn.Linear` layers handed a tuple observation
  shape. They use the same image trunk and now run on image observations.
- **Every run with no `env` argument crashed on exit** with
  `AttributeError: 'NoneType' object has no attribute 'close'`, because ten
  trainers called `env.close()` when `env` was `None`. Guarded.
- **`a2c` and `a2c-cnn` ignored `--steps`**, collecting whole episodes per
  update: `--steps 200` on Pong ran ~132,000 steps. They now stop at
  `total_timesteps`.

### Changed
- The `-cnn` aliases default to image environments: `a2c-cnn`, `ppo-cnn`,
  `reinforce-cnn`, `rnd-cnn` use `ALE/Pong-v5`; `ddpg-cnn`, `sac-cnn`,
  `td3-cnn` use `CarRacing-v3`. Install `neatrl[atari]` or `neatrl[box2d]` for
  those environments.
- Discrete observation spaces are one-hot encoded by inspecting the space
  instead of trusting the `grid_env` flag.

## [2.0.3] - 2026-10-29

### Fixed
- **Training crashed without the `[cli]` extra.** 2.0.2 made `Dashboard`
  optional so that `import neatrl` worked without `rich`, but the trainers
  still called `Dashboard(...)` and died with
  `TypeError: 'NoneType' object is not callable` on the first step. Fifteen
  trainers failed on every run — `a2c_cnn`, `a2c_mlp`, `ddpg_cnn`, `ddpg_mlp`,
  `dueling_dqn_mlp`, `ppo_cnn`, `ppo_mlp`, `reinforce_cnn`, `reinforce_mlp`,
  `rnd_cnn`, `rnd_mlp`, `sac_cnn`, `sac_mlp`, `td3_cnn`, `td3_mlp` — and the
  remaining four (`dqn_mlp`, `dqn_cnn`, `double_dqn_mlp`, `double_dqn_cnn`)
  only skipped the dashboard when stdout was not a TTY, so they still crashed
  in an interactive terminal. Every trainer now builds it only when
  `Dashboard is not None and sys.stdout.isatty()`, so a bare
  `pip install neatrl` can train without the extra.
- `train_a2c` on a continuous-control environment (e.g. `Pendulum-v1`) used to
  fail inside gymnasium with `IndexError: invalid index to scalar variable`,
  because the built-in actor is categorical and returned a scalar action. It
  now raises a `ValueError` naming the continuous-control trainers
  (`train_ppo`, `train_sac`, `train_td3`, `train_ddpg`) and noting that a
  custom `actor_class` can be supplied.

## [2.0.2] - 2026-10-29

### Fixed
- **`import neatrl` raised `ModuleNotFoundError: No module named 'rich'`** on
  a plain `pip install neatrl`. Every trainer imports the terminal dashboard,
  which imports `rich`, and `rich` is only in the optional `cli` extra — so
  the package was unusable without it. `neatrl.cli` now guards that import
  and exposes `Dashboard = None` when the extra is absent; training runs with
  plain logging instead of a crash. `pip install "neatrl[cli]"` still enables
  the dashboard.

## [2.0.1] - 2026-10-29

### Fixed
- **`neatrl train` failed on every release installed from PyPI** with
  `Script not found: .../docs/cli/<algo>.py`. The command loaded its example
  scripts by path, and that directory only exists in a repository checkout.
  Dispatch now comes from `neatrl.cli.recipes`, a table inside the package,
  so the installed console script works from any directory.

### Changed
- `neatrl train` no longer requires an environment argument. Omitting it uses
  the algorithm's default: `neatrl train dqn` trains on `CartPole-v1`,
  `neatrl train sac` on `Pendulum-v1`, `neatrl train rnd` on
  `CliffWalking-v0`.
- `neatrl train --steps N` overrides the training length — `total_episodes`
  for `reinforce`, `total_timesteps` for everything else.
- The README documented fuzzy environment matching (`cartpole` resolving to
  `CartPole-v1`) that the command never implemented. It now documents the
  behaviour that exists.

## [2.0.0] - 2026-10-29

### Breaking
- **`learning_rate=` is now `lr=`** on every trainer that already existed in
  1.0.0: `train_dqn`, `train_reinforce`, `train_ppo`, `train_sac`,
  `train_td3`, `train_a2c`, `train_ddpg`, `train_dueling_dqn`,
  `train_ppo_rnd` and their `_cnn` variants.
- **`train_reinforce` / `train_reinforce_cnn` take `total_episodes=`**
  instead of `total_steps=`.
- **Algorithm modules are split by network type**, so the old import paths
  are gone: `neatrl.dqn` → `neatrl.dqn_mlp`, and the same for `ppo`, `sac`,
  `td3`, `a2c`, `ddpg`, `reinforce`, `dueling_dqn` and `rnd`, each keeping a
  `_cnn` sibling. `import neatrl` and the `neatrl.train_*` names are unchanged.

### Added - CLI and dashboard
- **`neatrl` console command**: `neatrl train <algo> <env>` with 20 aliases,
  including `dqn-cnn`, `double-dqn`, `double-dqn-cnn`, `q-learning`,
  `sac-cnn` and `td3-cnn`. It runs the matching script in `docs/cli/`, so it
  expects a repository checkout.
- **Live terminal dashboard** (`cli/dashboard.py`, built on `rich`): logs,
  config and charts views with `l` / `c` / `v` keys. It only takes over the
  terminal when stdout is a TTY, and skips raw mode entirely when stdin has
  no file descriptor. New `cli` extra: `pip install "neatrl[cli]"` (`rich`,
  `psutil`, `nvidia-ml-py`).
- **`custom_agent=`** on `train_dqn`, `train_dueling_dqn`, `train_reinforce`
  and `train_reinforce_cnn` to pass your own `torch.nn.Module`; it is
  dimension-checked against the environment before training starts.
- **`neatrl.__version__`**, read from the installed distribution metadata.

### Added - Atari DQN-CNN
- **`dqn_cnn.py` / `train_dqn_cnn()`**: DQN with the convolutional network and
  Atari preprocessing from Mnih et al. (2015)
  - 84x84 grayscale frames, 4-frame stacks, frame skip 4, no-op starts, sticky
    actions disabled and rewards clipped to `{-1, 0, 1}`
  - Target network copied every 10k steps, epsilon annealed 1.0 → 0.1
  - Evaluation logs `values/mean_max_q` next to
    `values/true_discounted_return` so the overestimation gap is directly
    readable while training
  - New CLI alias: `neatrl train dqn-cnn ALE/Pong-v5`

### Added - Double DQN
- **`double_dqn_mlp.py` / `train_double_dqn()`** and
  **`double_dqn_cnn.py` / `train_double_dqn_cnn()`**: the Double DQN target of
  van Hasselt et al. (2016), `R + gamma * Q_target(s', argmax_a Q_online(s', a))`
  - Drop-in replacements for `train_dqn` / `train_dqn_cnn`; same
    preprocessing and hyperparameters, only the bootstrap target differs
  - `double_dqn_target()` helper in both modules, unit-tested against a manual
    target computation and against terminal transitions
  - `eval_callback` hook on `train_dqn`, `train_dqn_cnn`, `train_double_dqn`
    and `train_double_dqn_cnn`; the metrics dict carries `model`, so callers
    can measure the network being trained
  - New CLI aliases: `neatrl train double-dqn CartPole-v1` and
    `neatrl train double-dqn-cnn ALE/Pong-v5`
- **`docs/non-cli/DQN/overestimation_demo.py`**: reproduces the Figure 3
  protocol of van Hasselt et al. (2016), comparing `max_a Q(s, a)` with the
  discounted return the greedy policy actually earns, for DQN and Double DQN
  side by side (MLP or CNN depending on the environment, live CSV + plot output)
  - `--grid-env` one-hot encodes discrete observations so FrozenLake,
    CliffWalking and Taxi can be used
- **`docs/non-cli/DQN/overestimation_results.md`**: measured results and plots
  for CartPole, LunarLander and Pong, plus the protocol and its failure modes
- **`scripts/check_dqn_family.py`**: sweeps all four trainers over 18
  env/trainer pairs (discrete, continuous and image observations) and verifies
  the returned network on a real observation
- **`scripts/check_tui_pty.py`**: runs each trainer inside a pseudo-terminal
  because the Dashboard only activates on a TTY, then asserts ANSI frames were
  drawn and the l/c/v views cycle

### Added - Tabular Q-learning
- **`q_learning.py` / `train_q_learning()`**: tabular off-policy TD control,
  `Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]`, for
  small discrete Gym environments (FrozenLake, CliffWalking, Taxi)
  - Rejects `Box` observation or action spaces up front with an error that
    names DQN as the function-approximation alternative
  - Returns the trained `numpy` Q-table; checkpoints and evaluation reuse the
    same `eval_every` / `save_every` / `num_eval_eps` contract as the
    neural trainers
  - New CLI alias: `neatrl train q-learning FrozenLake-v1`

### Changed
- `train_sac`, `train_sac_cnn`, `train_td3` and `train_td3_cnn` are now
  exported from `neatrl` directly; they were only reachable through their
  modules.
- Every trainer calls `configure_logging()` and writes a log file to
  `runs/<exp_name>/`.
- `pygame-ce` is a runtime dependency, matching the classic-control error
  hint that already told users to install it.

### Fixed
- `evaluate()` in `dqn_mlp` and `dqn_cnn` looped forever on environments
  registered without a `TimeLimit` (e.g. `CliffWalking-v0`), because a policy
  that circles never sets `terminated`/`truncated`. Episodes are now capped at
  the spec's `max_episode_steps`, falling back to 10,000 steps.
- `estimate_values` (DQN-CNN) compared the mean of `max_a Q(s, a)` over every
  visited state against a single per-episode `G_0`, which measures two
  different quantities and hid the bias. It now pairs each state with the
  discounted return that followed it, matching Figure 3 of van Hasselt et al.
  (2016).
- `Dashboard` no longer crashes when stdin has no file descriptor (pytest,
  notebooks, some CI runners); terminal raw mode is skipped in that case.
- `train_dqn(save_every=0)` raised `ZeroDivisionError`; checkpointing is now
  skipped, matching the other trainers. The same guard was added for
  `eval_every=0` in `dqn_mlp`, `dueling_dqn_mlp` and `q_learning` (and for
  `save_every=0` in `dueling_dqn_mlp` / `q_learning`).
- `Dashboard.push_many` calls in `dqn_mlp`, `dueling_dqn_mlp`, `ppo_mlp` and
  `sac_mlp` now use the existing `push_metrics` method, so training no longer
  crashes on the first dashboard update.
- `validate_q_network_dimensions` now accepts image observation shapes and
  checks for a convolutional trunk, matching the other validators.

## [1.0.0] - 2025-12-29

### Added - SAC
- **`neatrl/sac.py` / `train_sac()` and `train_sac_cnn()`**: maximum-entropy
  actor-critic with twin critics, a stochastic Gaussian policy and
  reparameterised sampling
  - Temperature is tuned automatically against a target entropy
    (`autotune_alpha`, `target_entropy_scale`, `alpha`)
  - Delayed policy updates via `policy_frequency`, with `target_network_frequency`
    for the critics
- **`neatrl/td3.py` / `train_td3()` and `train_td3_cnn()`**: twin critics, a
  delayed deterministic actor and target policy smoothing
- Neither module was re-exported from `neatrl/__init__.py`; both were reachable
  only by importing the module, until 2.0.0 added the top-level names.

### Changed - breaking
- **`get_action()` on the PPO, RND and A2C actor networks no longer takes the
  unused `action` argument**: `get_action(x, action=None)` became
  `get_action(x)`. Code that subclassed those networks and passed or overrode
  the argument has to drop it.

### Fixed
- **TD3 training loop**: the TD target is detached so gradients cannot flow
  into the target networks; observations, actions, rewards and dones are
  cloned to avoid in-place operation errors; redundant `zero_grad` calls are
  gone; and gradients are clipped on both the critics and the actor.

## [0.6.0] - 2025-12-26

### Added - A2C
- **`neatrl/a2c.py` / `train_a2c()` and `train_a2c_cnn()`**: synchronous
  advantage actor-critic, exported from `neatrl` alongside the other
  trainers
  - Vectorised rollouts via `n_envs`. The loss is weighted by `ENTROPY_COEFF`,
    and the CNN variant additionally exposes `VALUE_COEFF`
  - Optional learning-rate annealing (`anneal_lr`) and observation/reward
    normalisation (`normalize_obs`, `normalize_reward`)
- **`mujoco` extra**: `pip install "neatrl[mujoco]"` for MuJoCo continuous
  control environments (`gymnasium[mujoco]`).

### Changed
- **A2C continuous actions are bounded**: the policy squashes its mean with
  `2 * tanh`, so it emits actions inside the environment's range instead of
  unbounded reals — required for `Pendulum-v1`, whose actions live in [-2, 2].
  The Pendulum example now trains on four parallel environments (`n_envs=4`).
- **DDPG-CNN CarRacing actor fixed**: the actor took the observation *shape*
  as its own `obs_shape` argument, so `Conv2d(obs_shape, …)` received a tuple
  and the fully-connected layer was sized for a wrong feature map
  (`64*8*8`). It now takes `obs_shape` per-instance, builds
  `Conv2d(obs_shape[0], …)`, and uses `64*7*7`.
- The DDPG examples pass `env_id=` rather than a pre-built `env=`, and the
  BipedalWalker example records video (`capture_video=True`).

## [0.5.0] - 2025-12-24

> The TD3, SAC and A2C sections below were written ahead of this release. The
> published 0.5.0 wheel contained DQN, Dueling DQN, PPO, REINFORCE, RND and
> DDPG; A2C shipped in 0.6.0, and TD3 and SAC in 1.0.0.

### Added - PPO Algorithm Implementation
- **PPO (Proximal Policy Optimization)**: Full implementation of PPO with Generalized Advantage Estimation (GAE)
  - **Standard PPO**: `train_ppo()` function for discrete and continuous action spaces
  - **PPO with CNN**: `train_ppo_cnn()` function for Atari/image-based environments with automatic CNN architecture
  - **GAE Support**: Generalized Advantage Estimation with configurable lambda parameter for improved advantage estimation
  - **Vectorized Environments**: Support for parallel environments (`n_envs` parameter)

### Added - TD3 Algorithm Implementation
- **TD3 (Twin Delayed DDPG)**: Advanced actor-critic method for continuous action spaces with improved stability
  - **Twin Q-Networks**: Uses two Q-networks to reduce overestimation bias
  - **Delayed Policy Updates**: Updates policy less frequently than Q-networks for stability
  - **Target Policy Smoothing**: Adds noise to target actions to improve exploration
  - **Experience Replay**: Experience replay buffer for stable learning
  - **CNN Support**: `train_td3_cnn()` function for image-based environments like CarRacing

### Added - SAC Algorithm Implementation
- **SAC (Soft Actor-Critic)**: State-of-the-art maximum entropy RL algorithm
  - **Stochastic Policy**: Uses Gaussian policies with reparameterization trick
  - **Twin Q-Networks**: Two Q-networks to reduce overestimation bias
  - **Automatic Entropy Tuning**: Adaptive temperature parameter (alpha) for entropy regularization
  - **Maximum Entropy**: Balances exploration and exploitation through entropy maximization
  - **CNN Support**: `train_sac_cnn()` function for image-based environments


### Added - DDPG Algorithm Implementation
- **DDPG (Deep Deterministic Policy Gradient)**: Full implementation of actor-critic method for continuous action spaces
  - **Deterministic Policy Gradient**: Optimized for continuous control tasks with deterministic policies
  - **Experience Replay**: Experience replay buffer for stable learning and reduced correlation
  - **Target Networks**: Separate target networks for actor and critic with soft updates
  - **Exploration Noise**: Ornstein-Uhlenbeck process for action exploration in continuous spaces
  - **Continuous Action Support**: Designed for environments like robotics and physics simulations

### Added - A2C Algorithm Implementation
- **A2C (Advantage Actor-Critic)**: Synchronous actor-critic algorithm implementation
  - **Advantage Function**: Uses advantage estimation for reduced variance in policy gradients
  - **Synchronous Training**: Single-threaded implementation for stable and reproducible training
  - **Monte Carlo Returns**: Episode-based value estimation using full trajectory returns
  - **Parallel Environments**: Vectorized environment support for efficient data collection
  - **Flexible Action Spaces**: Support for both discrete and continuous action spaces


## [0.4.0] - 2025-12-21

### Added - RND
- **`neatrl/rnd.py` / `train_rnd()`**: Random Network Distillation exploration
  on top of PPO — a fixed randomly initialised target network and a trainable
  predictor, with the prediction error used as the intrinsic reward
  - A `RunningMeanStd` normalises intrinsic rewards against the running
    statistics of the return, so the curiosity bonus stays on a comparable
    scale to the extrinsic reward
  - `OneHotWrapper` one-hot encodes discrete observations, so grid
    environments (FrozenLake, CliffWalking) work through the same code path
  - `train_rnd` was renamed to `train_ppo_rnd` (with `train_ppo_rnd_cnn`
    added) in 0.5.0

## [0.3.0] - 2025-12-17

> Written ahead of the release: the 0.3.0 wheel contained DQN, Dueling DQN and
> REINFORCE only. RND first shipped as `train_rnd` in 0.4.0.

### Added - PPO-RND Algorithm Implementation
- **PPO-RND Algorithm**: Full implementation of Proximal Policy Optimization with Random Network Distillation
  - Intrinsic motivation through novelty detection using predictor and target networks
  - Combined extrinsic and intrinsic rewards for enhanced exploration
  - Support for both discrete and continuous action spaces



## [0.2.1] - 2025-12-17

> This version has no distribution files left on PyPI; installing it will fail.

### Added - REINFORCE Algorithm Enhancements
- **Atari Support**: REINFORCE now supports Atari games with automatic CNN architecture detection and preprocessing
- **Parallel Environments**: Added support for vectorized environments (`n_envs` parameter) for faster training
- **Continuous Action Spaces**: REINFORCE now handles both discrete and continuous action spaces seamlessly
- **Advanced Gradient Logging**: Added per-layer gradient norm logging for WandB integration
  - Logs gradient norms before and after clipping
  - Tracks gradient clip ratio
  - Per-layer gradient monitoring for all network parameters
- **Flexible Action Handling**: Automatic detection and conversion for discrete vs continuous action spaces
- **Episode-based Training**: Changed from episodes to total_steps parameter for consistency with DQN
- **Enhanced Documentation**: Updated REINFORCE docs with Atari examples, parallel training guides, and new parameters

### Changed
- **Training Parameter**: Renamed `episodes` to `total_steps` in `train_reinforce()` for API consistency
- **Default Configuration**: Set `n_envs=4` as default for improved training speed
- **Network Architecture**: PolicyNet automatically detects observation space type (Atari vs non-Atari)


## [0.2.0] - 2025-12-14

### Added
- **Grid Environment Support**: Added support for grid-based environments from Gymnasium, including automatic one-hot encoding for discrete states

### Changed
- **Parameter Renaming**: Renamed `record` parameter to `capture_video` for consistency in video recording functionality

## [0.1.5] - 2025-12-13

### Fixed
- **Atari wrappers imported from the wrong package**: `dqn.py` did
  `from gym.wrappers import AtariPreprocessing, FrameStackObservation` and
  called `FrameStackObservation(env, num_stack=4)`. Gymnasium's wrapper is
  imported from `gymnasium.wrappers` and takes `stack_size=`. Any run with
  `atari_wrapper=True` failed on the import.
- **`evaluate()` ignored `atari_wrapper`**: it built its environment without
  the Atari preprocessing the training env used, so evaluation fed the network
  raw frames of the wrong shape. It now takes `atari_wrapper` and applies the
  same wrappers.

### Changed
- `make_env` returns a thunk (`make_env(...)()`) so each of the `n_envs=4`
  parallel environments gets its own seed (`seed + idx`) instead of sharing
  one.
- Removed the ad-hoc `"Hooray! Episode reward exceeded 500, stopping early"`
  early exit from training.

## [0.1.4] - 2025-12-13

### Added
- **Custom Agent Support**: Added ability to pass custom neural network classes to the DQN training function
- **Network Architecture Display**: Added torchinfo integration to display Q-network architecture summary during training
- **Improved Error Handling**: Better validation for custom agent constructors with informative error messages

### Changed
- **Agent Parameter**: Modified `custom_agent` parameter to accept nn.Module subclasses instead of instances
- **Target Network Creation**: Ensured target networks are properly instantiated for custom agents

### Fixed
- **Constructor Validation**: Added checks to ensure custom agent classes have compatible constructors

## [0.1.3] - 2025-12-13

### Changed
- Dropped `cv2`, the TensorBoard `SummaryWriter` and the `RecordVideo`
  wrapper from `dqn.py`; collected frames are stacked with `np.stack` and
  written through `imageio`.

### Added
- Weights & Biases integration for experiment tracking
- Comprehensive documentation

### Features
- DQN algorithm with experience replay
- Support for Atari and classic control environments
- Modular configuration system
- Evaluation and model saving utilities

## [0.1.2] - 2025-12-13

### Added
- First published release: `train_dqn()` with an MLP Q-network, experience
  replay, target-network updates, evaluation and checkpointing.