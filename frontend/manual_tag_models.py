#!/usr/bin/env python3
"""
Script to manually tag models with proper categories and tags based on their actual functionality
"""

import os
import re
from pathlib import Path

def update_model_tags():
    """Update model tags manually based on content analysis"""
    
    # Manual model mappings based on actual functionality
    model_tags = {
        # Audio/Speech models
        'whisper': {
            'category': 'Audio/Speech',
            'tags': ['audio', 'speech-recognition', 'transformer', 'asr', 'openai']
        },
        'clap': {
            'category': 'Audio/Speech', 
            'tags': ['audio', 'contrastive-learning', 'multimodal', 'clip-like']
        },
        'moonshine': {
            'category': 'Audio/Speech',
            'tags': ['audio', 'speech-recognition', 'transformer', 'asr', 'fast-inference']
        },
        'tts': {
            'category': 'Audio/Speech',
            'tags': ['audio', 'text-to-speech', 'synthesis', 'voice']
        },
        
        # Computer Vision models
        'clip': {
            'category': 'Computer Vision',
            'tags': ['computer-vision', 'multimodal', 'contrastive-learning', 'vision-language']
        },
        'siglip': {
            'category': 'Computer Vision', 
            'tags': ['computer-vision', 'multimodal', 'vision-language', 'sigmoid-loss']
        },
        'vit': {
            'category': 'Computer Vision',
            'tags': ['computer-vision', 'vision-transformer', 'attention', 'image-classification']
        },
        'llava': {
            'category': 'Computer Vision',
            'tags': ['computer-vision', 'multimodal', 'vision-language', 'llm', 'conversation']
        },
        'paligemma': {
            'category': 'Computer Vision',
            'tags': ['computer-vision', 'multimodal', 'vision-language', 'gemma-based']
        },
        
        # Generative Models
        'vae': {
            'category': 'Generative Models',
            'tags': ['generative', 'variational-autoencoder', 'unsupervised', 'latent-space']
        },
        'cgans': {
            'category': 'Generative Models',
            'tags': ['generative', 'gan', 'conditional', 'adversarial', 'mnist']
        },
        'dcgans': {
            'category': 'Generative Models', 
            'tags': ['generative', 'gan', 'deep-convolutional', 'adversarial']
        },
        'cyclegan': {
            'category': 'Generative Models',
            'tags': ['generative', 'gan', 'cycle-consistency', 'unpaired-translation']
        },
        'wgans': {
            'category': 'Generative Models',
            'tags': ['generative', 'gan', 'wasserstein', 'adversarial']
        },
        'pix2pix': {
            'category': 'Generative Models',
            'tags': ['generative', 'gan', 'image-translation', 'paired-data']
        },
        
        # Language Models
        'mixtral': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'mixture-of-experts', 'moe']
        },
        'llama': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'meta', 'decoder-only']
        },
        'llama4': {
            'category': 'Language Models', 
            'tags': ['language-model', 'transformer', 'meta', 'decoder-only', 'llama4']
        },
        'gpt': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'gpt', 'generative', 'openai']
        },
        'bert': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'encoder-only', 'bidirectional', 'bert']
        },
        'gemma': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'google', 'open-weights']
        },
        'gemma3': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'google', 'open-weights', 'gemma3']
        },
        'deepseekv3': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'deepseek', 'moe', 'mixture-of-experts']
        },
        'kimi-k2': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'kimi', 'long-context']
        },
        'transformer': {
            'category': 'Language Models',
            'tags': ['transformer', 'attention', 'encoder-decoder', 'foundational']
        },
        
        # Sequential Models 
        'encoder-decoder': {
            'category': 'Sequential Models',
            'tags': ['sequence-to-sequence', 'translation', 'encoder-decoder', 'lstm-based']
        },
        'seq2seq': {
            'category': 'Sequential Models',
            'tags': ['sequence-to-sequence', 'translation', 'rnn-based']
        },
        'rnns': {
            'category': 'Sequential Models',
            'tags': ['rnn', 'recurrent', 'sequence-modeling', 'vanilla-rnn']
        },
        'lstm': {
            'category': 'Sequential Models',
            'tags': ['lstm', 'recurrent', 'sequence-modeling', 'long-memory']
        },
        'gru': {
            'category': 'Sequential Models', 
            'tags': ['gru', 'recurrent', 'sequence-modeling', 'gated']
        },
        
        # Attention Mechanisms
        'attention-mechanisms': {
            'category': 'Attention Mechanisms',
            'tags': ['attention', 'bahdanau', 'luong', 'mechanism', 'neural-machine-translation']
        },
        'differential-transformer': {
            'category': 'Attention Mechanisms',
            'tags': ['attention', 'differential', 'transformer', 'advanced-attention']
        },
        
        # Fine-tuning Methods
        'lora': {
            'category': 'Fine-tuning',
            'tags': ['fine-tuning', 'lora', 'low-rank-adaptation', 'parameter-efficient']
        },
        'fine-tuning-using-peft': {
            'category': 'Fine-tuning',
            'tags': ['fine-tuning', 'peft', 'parameter-efficient', 'huggingface']
        },
        'dpo': {
            'category': 'Fine-tuning',
            'tags': ['fine-tuning', 'dpo', 'direct-preference-optimization', 'rlhf-alternative']
        },
        'orpo': {
            'category': 'Fine-tuning',
            'tags': ['fine-tuning', 'orpo', 'odds-ratio-preference-optimization']
        },
        'simplepo': {
            'category': 'Fine-tuning',
            'tags': ['fine-tuning', 'simple-preference-optimization', 'alignment']
        },
        
        # Training Methods
        'ddp': {
            'category': 'Training Methods',
            'tags': ['distributed-training', 'pytorch', 'data-parallel', 'multi-gpu']
        }
    }
    
    # SmolHub playground models
    smolhub_tags = {
        'smolmixtral': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'mixture-of-experts', 'compact', 'educational']
        },
        'smoltransformer': {
            'category': 'Translation Models',
            'tags': ['translation', 'transformer', 'encoder-decoder', 'english-hindi', 'compact']
        },
        'storykimi': {
            'category': 'Language Models', 
            'tags': ['language-model', 'transformer', 'storytelling', 'deepseek-inspired', 'moe']
        },
        'storyllama': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'storytelling', 'llama-inspired', 'compact']
        },
        'storymixtral': {
            'category': 'Language Models',
            'tags': ['language-model', 'transformer', 'storytelling', 'mixtral-inspired', 'moe']
        }
    }
    
    # Update regular models
    models_dir = Path('_models')
    for md_file in models_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        
        # Extract model name from filename 
        filename = md_file.stem
        model_name = '-'.join(filename.split('-')[1:])  # Remove number prefix
        
        if model_name in model_tags:
            tags_info = model_tags[model_name]
            category = tags_info['category']
            tags = tags_info['tags']
            
            # Update category
            content = re.sub(r'category: "[^"]*"', f'category: "{category}"', content)
            
            # Add tags if not present
            if 'tags:' not in content:
                # Add tags after the date line
                content = re.sub(
                    r'(date: [^\n]+\n)', 
                    f'\\1tags: {tags}\n', 
                    content
                )
            else:
                # Update existing tags
                content = re.sub(
                    r'tags: \[[^\]]*\]',
                    f'tags: {tags}',
                    content
                )
            
            md_file.write_text(content, encoding='utf-8')
            print(f"✅ Updated {md_file.name}: {category} - {tags}")
    
    # Update SmolHub models
    smolhub_dir = Path('_smolhub')
    for md_file in smolhub_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        
        # Extract model name from filename
        filename = md_file.stem
        # Extract the actual model name after 'playground-XX-'
        model_name = '-'.join(filename.split('-')[2:])
        
        if model_name in smolhub_tags:
            tags_info = smolhub_tags[model_name]
            category = tags_info['category'] 
            tags = tags_info['tags']
            
            # Update the technical details section
            content = re.sub(
                r'- \*\*Category\*\*: [^\n]+',
                f'- **Category**: {category}',
                content
            )
            
            # Update existing tags
            content = re.sub(
                r'tags: \[[^\]]+\]',
                f'tags: {tags}',
                content
            )
            
            md_file.write_text(content, encoding='utf-8')
            print(f"✅ Updated {md_file.name}: {category} - {tags}")

if __name__ == "__main__":
    update_model_tags()
    print("\n🎉 Manual tagging completed!")
