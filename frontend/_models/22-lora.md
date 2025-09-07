---
title: "LoRA"
excerpt: "From scratch implementation of LoRA"
collection: models
layout: model-implementation
category: "Fine-tuning"
framework: "PyTorch"
dataset: "TinyShakespeare"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/LoRA"
date: 2025-04-05
---

## Overview
From scratch implementation of LoRA

## Technical Details
- **Framework**: PyTorch
- **Dataset**: TinyShakespeare
- **Category**: Fine-tuning

## Implementation Details

I implemented the LoRA framework using Pytorch on Tinyshakespeare dataset.

[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)

### Datasets

**Tineshakespeare**: in the /data folder

### Frameworks:
**Pytorch**

### Results (on A100 GPU Single)

**Training steps:** 1000
**Validation steps:** per 100 training steps

**Train loss:**  3.51
**Val loss:** 3.50

## Source Code
📁 **GitHub Repository**: [LoRA](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/LoRA)

View the complete implementation, training scripts, and documentation on GitHub.
