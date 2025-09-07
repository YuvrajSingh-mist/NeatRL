#!/usr/bin/env python3

def categorize_model(name, description, readme_content):
    """Categorize the model based on its content"""
    content = (name + " " + description + " " + readme_content).lower()
    
    print(f"Checking content for: {name}")
    print(f"Content contains 'whisper': {'whisper' in content}")
    print(f"Content contains 'audio': {'audio' in content}")
    print(f"Content contains 'speech': {'speech' in content}")
    print(f"Content contains 'attention': {'attention' in content}")
    
    # Check more specific categories first
    if any(term in content for term in ['audio', 'speech', 'clap', 'whisper']):
        print("✅ Matched Audio/Speech")
        return "Audio/Speech"
    elif any(term in content for term in ['computer vision', 'clip', 'siglip', 'vit', 'vision transformer', 'resnet', 'efficientnet', 'yolo', 'object detection', 'image classification', 'semantic segmentation']):
        print("✅ Matched Computer Vision")
        return "Computer Vision"
    elif any(term in content for term in ['gan', 'dcgan', 'cyclegan', 'cgan', 'generative']):
        print("✅ Matched Generative Models")
        return "Generative Models"
    elif any(term in content for term in ['fine', 'tuning', 'peft', 'dpo']):
        print("✅ Matched Fine-tuning")
        return "Fine-tuning"
    elif any(term in content for term in ['training', 'ddp', 'distributed', 'optimization']):
        print("✅ Matched Training Methods")
        return "Training Methods"
    elif any(term in content for term in ['gpt', 'llama', 'bert', 'transformer', 'language model', 'text', 'nlp']):
        print("✅ Matched Language Models")
        return "Language Models"
    elif any(term in content for term in ['attention', 'differential']):
        print("✅ Matched Attention Mechanisms")
        return "Attention Mechanisms"
    else:
        print("✅ Matched Other")
        return "Other"

# Test with Whisper data
name = "Whisper"
description = "From scratch implementation of Whisper"
readme_content = """
# Whisper model in Pytorch from scratch implementation

Trained a small whisper model coded and trained from scratch in Pytorch 


[Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)

## ModelArgs Hyperparameters

| Parameter               | Value                  | Description                                                                 |
|-------------------------|------------------------|-----------------------------------------------------------------------------|
| `batch_size`            | 64                     | The number of samples processed before the model is updated.                |
| `max_lr`                | 2e-4                   | Maximum learning rate.                                                      |
| `dropout`               | 0.1                    | Dropout rate for regularization.                                            |
| `epochs`                | 10                     | Number of training epochs.                                                  |
| `block_size`            | 64                     | Sequence length (number of tokens or time steps).                           |
| `tgt_vocab_size`        | 50262     | Size of the target vocabulary.                                              |
| `embeddings_dims`       | 384                    | Dimensionality of token embeddings.                                         |
| `attn_dropout`          | 0.1                    | Dropout rate for attention layers.                                          |
| `no_of_heads`           | 6                      | Number of attention heads in multi-head attention.                          |
| `no_of_decoder_layers`  | 6                      | Number of decoder layers in the model.                                      |
| `weight_decay_optim`    | 0.01                   | Weight decay for the optimizer.                                             |
| `log_mel_features`      | 80                     | Number of Mel spectrogram features.                                         |
| `kernel_size`           | 3                      | Kernel size for convolutional layers.                                       |
| `stride`                | 2             | Stride for convolutional layers.                                            |
| `sr`                    | 16000                  | Sampling rate of the audio.                                                 |
| `device`                | `'cuda:0'`             | Device to run the model on (e.g., GPU).                                     |
| `SAMPLING_RATE`         | 16000                  | Sampling rate of the audio.                                                 |
| `N_MELS`                | 80                     | Number of Mel bins in the spectrogram.                                      |
| `WINDOW_DURATION`       | 0.025                  | Duration of the analysis window in seconds (25 ms).                         |
| `STRIDE_DURATION`       | 0.010                  | Stride between consecutive windows in seconds (10 ms).                      |
| `max_t`                 | 500                    | Maximum time steps in the spectrogram.                                      |
| `n_channels`            | 80                     | Number of channels in the input spectrogram.                                |
| `hidden_dim`            | 4 * `embeddings_dims`  | Number of neurons in the feed-forward network (FFN).                        |

### Dataset

[Gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)

Used the 'xs' snapshot.

### Frameworks:
**Pytorch**


### Epochs/Steps
Epochs (train) = 10

Val iterations = every epoch


### Loss Curves

![Train and Val loss curves](img/loss.jpg)
"""

result = categorize_model(name, description, readme_content)
print(f"Final category: {result}")
