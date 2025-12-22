# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2025-12-22

### Added - PPO-RND Algorithm Implementation
- **PPO-RND Algorithm**: Full implementation of Proximal Policy Optimization with Random Network Distillation
  - Intrinsic motivation through novelty detection using predictor and target networks
  - Combined extrinsic and intrinsic rewards for enhanced exploration
  - Support for both discrete and continuous action spaces
  - Automatic render mode handling for video recording (fixes render warnings)
  - Comprehensive WandB logging with global step as x-axis for all charts
- **Enhanced Action Space Support**: Improved handling for continuous actions with proper tensor shapes
- **Video Recording Fixes**: Automatic render mode detection and video transpose handling for grayscale/RGB frames
- **Global Step Logging**: All WandB logs now use global_step for consistent x-axis across charts
- **Distribution Logging**: Adaptive logging for both Categorical (discrete) and Normal (continuous) distributions

### Changed
- **Action Storage**: Fixed tensor shape issues for continuous action spaces in PPO training
- **WandB Integration**: Replaced "step" with "global_step" in all logging keys for better tracking

### Fixed
- **Render Mode Warnings**: Automatic specification of render_mode to prevent Gymnasium warnings
- **Video Transpose Errors**: Proper handling of frame dimensions for video logging
- **Continuous Action Logging**: Fixed AttributeError for Normal distributions in WandB logs

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

### Fixed
- **Multi-environment Action Handling**: Fixed tensor-to-scalar conversion for vectorized environments
- **Action Space Detection**: Proper handling of both discrete and continuous action spaces in training loop
- **Evaluation Function**: Improved action conversion logic for different environment types

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