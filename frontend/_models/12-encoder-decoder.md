---
title: "Encoder-Decoder"
excerpt: "From scratch implementation of Encoder-Decoder"
collection: models
layout: model-implementation
category: "Sequential Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Encoder-Decoder"
date: 2025-03-07
---

## Overview
From scratch implementation of Encoder-Decoder

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Sequential Models

## Implementation Details

Trained on the on German (de) to English (en) dataset

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 32       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.2      | Dropout.                                                                    |
| `epochs`     | 10       | Epochs                                                                      |           
| `block_size` | 32      | Seq Len                                                                     |
| `num_layers` | 4      | Layers for deep lstms                                                                |
| `No of neurons`| 128      | No of neurons in an GRU per layer                                          |    

### Frameworks:
**Pytorch**

### Epochs/Steps
Epochs (train) = 10

Val iterations = every epoch

### Losses

Train loss - 1.38

Val loss - 1.39

### Loss Curves

[📊 View Training Loss Curves](https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/Encoder-Decoder/img/loss.jpg)

## Source Code
📁 **GitHub Repository**: [Encoder-Decoder](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Encoder-Decoder)

View the complete implementation, training scripts, and documentation on GitHub.
