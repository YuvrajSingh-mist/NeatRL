---
title: "SimplePO"
excerpt: "From scratch implementation of SimplePO"
collection: models
layout: model-implementation
category: "Fine-tuning"
framework: "PyTorch"
dataset: "UltraFeedback"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/SimplePO"
date: 2025-04-04
---

## Overview
From scratch implementation of SimplePO

## Technical Details
- **Framework**: PyTorch
- **Dataset**: UltraFeedback
- **Category**: Fine-tuning

## Implementation Details


Trained OPT-330M model using SimplePO in Pytorch for Instruction Following

[SimplePO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 128        | The number of samples processed before the model is updated.                |
| `max_lr`     | 2e-5     | Maximum learning rate.                                                      |
| `device`     | 'cuda:0' | The device to run the model on (e.g., 'cuda:0' for GPU).                    |
| `beta`      | 2 | Beta values                                                                 |           
| `gamma`| 1.6     | Gamma values for the optimizer                                       |

### Datasets

[UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

### Frameworks:
**Pytorch**


## Source Code
📁 **GitHub Repository**: [SimplePO](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/SimplePO)

View the complete implementation, training scripts, and documentation on GitHub.
