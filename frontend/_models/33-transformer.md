---
title: "Transformer"
excerpt: "From scratch implementation of Transformer"
collection: models
layout: model-implementation
category: "Language Models"
framework: "PyTorch"
dataset: "Custom"
github_url: "https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Transformer"
date: 2025-03-10
---

## Overview
From scratch implementation of Transformer

## Key Features
- Attention Mechanism
- Transformer Architecture

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Language Models

## Implementation Details


I implemented the Vanilla Transformers using Pytorch on the German-English dataset.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Datasets

**Multi30k de-en**: [Link](https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/)

### Frameworks:
**Pytorch**

### Results (on T4 GPU Single)

**Training epochs:** 3
**Val epochs:** 5

**Train loss:** 0.02  (mean)
**Val loss:** 0.03 (mean)

[NOTE]: The train and val loss seems to be off. Please submit a PR or open a discussion if you find the issue and would really appreciate your help!

## Source Code
📁 **GitHub Repository**: [Transformer](https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/Transformer)

View the complete implementation, training scripts, and documentation on GitHub.
