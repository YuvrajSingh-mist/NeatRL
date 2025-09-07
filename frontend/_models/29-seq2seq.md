---
title: "Seq2Seq"
excerpt: "From scratch implementation of Seq2Seq"
collection: models
layout: model-implementation
category: "Sequential Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Seq2Seq"
date: 2025-04-25
---

## Overview
From scratch implementation of Seq2Seq

## Key Features
- Attention Mechanism

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Sequential Models

## Implementation Details

Trained a Seq2Seq model with the said attention mechanism  coded from scratch in Pytorch 

[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.0473)

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 32       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.1      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 32      | Seq Len                                                                      |
| `No of neurons`| 128      | No of neurons in an GRU per layer                                         |    
| `hidden_dim`| 4*embedding_dims      | No of neurons in FFN                                            |  
| `No of neurons`| 128      | No of neurons in an GRU per layer                                         |  

### Frameworks:
**Pytorch**

### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch

### Loss Curves

[📊 View Training Loss Curves](https://github.com/YuvrajSingh-mist/Paper-Replications/raw/master/Seq2Seq/https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/Seq2Seq/img/loss_curves.jpg)


## Source Code
📁 **GitHub Repository**: [Seq2Seq](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Seq2Seq)

View the complete implementation, training scripts, and documentation on GitHub.
