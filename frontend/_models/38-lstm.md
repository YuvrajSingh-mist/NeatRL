---
title: "LSTM"
excerpt: "From scratch implementation of lstm"
collection: models
layout: model-implementation
category: "Sequential Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/lstm"
date: 2025-04-25
---

## Overview
From scratch implementation of lstm

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Sequential Models

## Implementation Details


Trained 128K LSTM model coded from scratch in Pytorch 

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 32       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.1      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 64       | Sequence length                                                             |
| `No of neurons`     | 128       | Epochs                                                               |   

### Frameworks:
**Pytorch**

### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch

### Losses

Train loss - 0.49 

Val loss - 0.48

### Loss Curves

[📊 View Training Loss Curves](https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/lstm/img/loss_curves.jpg)

## Source Code
📁 **GitHub Repository**: [lstm](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/lstm)

View the complete implementation, training scripts, and documentation on GitHub.
