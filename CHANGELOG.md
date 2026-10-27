# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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

## [0.5.0] - 2025-12-24

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


## [0.3.0] - 2025-12-22

### Added - PPO-RND Algorithm Implementation
- **PPO-RND Algorithm**: Full implementation of Proximal Policy Optimization with Random Network Distillation
  - Intrinsic motivation through novelty detection using predictor and target networks
  - Combined extrinsic and intrinsic rewards for enhanced exploration
  - Support for both discrete and continuous action spaces



## [0.2.1] - 2025-12-17

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

## [0.1.3] - 2025-12-01

### Added
- Initial release with DQN implementation
- Weights & Biases integration for experiment tracking
- Video recording capabilities
- Comprehensive documentation

### Features
- DQN algorithm with experience replay
- Support for Atari and classic control environments
- Modular configuration system
- Evaluation and model saving utilities