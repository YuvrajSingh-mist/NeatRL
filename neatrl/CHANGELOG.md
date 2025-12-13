# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-12-14

### Added
- **MPS Device Support**: Added support for Apple Silicon MPS device with automatic fallback to CPU if unavailable
- **Atari Training Example**: Added comprehensive Atari game training example in documentation with convolutional network
- **Dynamic Device Handling**: Improved device parameter to accept strings and handle CUDA/MPS availability checks

### Changed
- **Network Architecture Display**: Simplified network printing by removing torchinfo dependency, now uses basic print
- **Atari Action Space**: Updated Atari example to use correct action space (4 for Breakout) and proper test observations
- **Documentation**: Added Atari games to supported environments list and included example script references

### Fixed
- **Evaluation Function Call**: Corrected argument order in evaluate function call during final video saving
- **Atari Custom Agent**: Fixed action space mismatch in Atari training script (was 2, now 4)
- **Device Parameter**: Fixed device parameter handling to properly convert strings to torch.device objects

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