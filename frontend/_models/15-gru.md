---
title: "GRU"
excerpt: "From scratch implementation of GRU"
collection: models
layout: model-implementation
category: "Sequential Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/GRU"
date: 2025-03-05
---

## Overview
From scratch implementation of GRU

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Sequential Models

## Implementation Details

Trained a GRU model coded from scratch in Pytorch 

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 16       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.2      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 16      | Seq Len                                     |
| `No of neurons`| 16      | No of neurons in an GRU per layer                                          |    

### Frameworks:
**Pytorch**

### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch

### Losses

Train loss - 0.51 

Val loss - 0.48

### Loss Curves

[📊 View Training Loss Curves](https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/GRU/img/loss_curves.jpg)

## Source Code
📁 **GitHub Repository**: [GRU](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/GRU)

View the complete implementation, training scripts, and documentation on GitHub.
