---
title: "RNNs"
excerpt: "From scratch implementation of RNNs"
collection: models
layout: model-implementation
category: "Sequential Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/RNNs"
date: 2025-03-07
---

## Overview
From scratch implementation of RNNs

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Sequential Models

## Implementation Details

Trained a RNN model coded from scratch in Pytorch 

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 16       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.2      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 16      | Sequence Length                                       |
| `No of neurons`| 16      | No of neurons in an RNN per layer                                          |    

### Frameworks:
**Pytorch**

### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch

### Losses

Train loss - 0.51 

Val loss - 0.50

### Loss Curves

[📊 View Training Loss Curves](https://github.com/YuvrajSingh-mist/Paper-Replications/raw/master/RNNs/https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/RNNs/img/loss_curves.jpg)


## Source Code
📁 **GitHub Repository**: [RNNs](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/RNNs)

View the complete implementation, training scripts, and documentation on GitHub.
