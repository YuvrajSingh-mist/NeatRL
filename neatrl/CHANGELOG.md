# Changelog

All notable changes to this project will be documented in this file.

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
  - **Comprehensive Logging**: WandB integration with entropy metrics (alpha, entropy, target_entropy)

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