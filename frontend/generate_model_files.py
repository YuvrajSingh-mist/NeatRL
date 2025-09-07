#!/usr/bin/env python3
"""
Script to generate markdown files for from scratch implementations from models.json
This script reads the models.json file and creates individual markdown files
for each model in the _models collection.
"""

import json
import os
import re
import argparse
from datetime import datetime
from pathlib import Path


def slugify(text):
    """Convert text to a URL-friendly slug"""
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def clean_markdown_content(content):
    """Clean and format markdown content"""
    if not content:
        return ""
    
    # Remove extra whitespace and normalize line endings
    content = content.strip()
    
    # Fix common markdown issues
    content = re.sub(r'\r\n', '\n', content)  # Normalize line endings
    content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive line breaks
    
    return content


def extract_framework_and_dataset(readme_content, description):
    """Extract framework and dataset information from readme content"""
    framework = "PyTorch"  # Default
    dataset = "Custom"     # Default
    
    # Look for framework mentions
    content_lower = (readme_content + " " + description).lower()
    if 'pytorch' in content_lower:
        framework = "PyTorch"
    elif 'tensorflow' in content_lower:
        framework = "TensorFlow"
    elif 'jax' in content_lower:
        framework = "JAX"
    
    # Look for dataset mentions
    dataset_patterns = {
        'tinystories': 'TinyStories',
        'tinyshakespeare': 'TinyShakespeare', 
        'imagenet': 'ImageNet',
        'cifar': 'CIFAR',
        'mnist': 'MNIST',
        'celeba': 'CelebA',
        'flickr': 'Flickr',
        'cornell': 'Cornell Movie Dialogs',
        'cityscapes': 'Cityscapes',
        'gigaspeech': 'Gigaspeech',
        'ultrafeedback': 'UltraFeedback',
        'fineweb': 'FineWeb'
    }
    
    for pattern, name in dataset_patterns.items():
        if pattern in content_lower:
            dataset = name
            break
    
    return framework, dataset


def categorize_model(name, description, readme_content):
    """Manual categorization based on model name - no automatic detection"""
    # Manual mapping for specific models
    manual_categories = {
        'whisper': 'Audio/Speech',
        'clap': 'Audio/Speech', 
        'moonshine': 'Audio/Speech',
        'tts': 'Audio/Speech',
        'clip': 'Computer Vision',
        'siglip': 'Computer Vision',
        'vit': 'Computer Vision',
        'llava': 'Computer Vision',
        'paligemma': 'Computer Vision',
        'vae': 'Generative Models',
        'cgans': 'Generative Models',
        'dcgans': 'Generative Models',
        'cyclegan': 'Generative Models',
        'wgans': 'Generative Models',
        'pix2pix': 'Generative Models',
        'mixtral': 'Language Models',
        'llama': 'Language Models',
        'gpt': 'Language Models',
        'bert': 'Language Models',
        'gemma': 'Language Models',
        'deepseekv3': 'Language Models',
        'kimi-k2': 'Language Models',
        'transformer': 'Language Models',
        'encoder-decoder': 'Sequential Models',
        'seq2seq': 'Sequential Models',
        'rnn': 'Sequential Models',
        'lstm': 'Sequential Models',
        'gru': 'Sequential Models',
        'attention': 'Attention Mechanisms',
        'differential': 'Attention Mechanisms',
        'lora': 'Fine-tuning',
        'peft': 'Fine-tuning',
        'dpo': 'Fine-tuning',
        'orpo': 'Fine-tuning',
        'simplepo': 'Fine-tuning',
        'ddp': 'Training Methods'
    }
    
    name_lower = name.lower()
    for key, category in manual_categories.items():
        if key in name_lower:
            return category
    
    return "Other"


def extract_key_features(readme_content):
    """Extract key features from readme content"""
    features = []
    
    # Look for common patterns
    if 'mixture of experts' in readme_content.lower() or 'moe' in readme_content.lower():
        features.append("Mixture of Experts (MoE)")
    if 'attention' in readme_content.lower():
        features.append("Attention Mechanism")
    if 'transformer' in readme_content.lower():
        features.append("Transformer Architecture")
    if 'distributed' in readme_content.lower() or 'ddp' in readme_content.lower():
        features.append("Distributed Training")
    if 'gradient checkpointing' in readme_content.lower():
        features.append("Memory Optimization")
    if 'fine-tuning' in readme_content.lower() or 'finetuning' in readme_content.lower():
        features.append("Fine-tuning")
    
    return features


def generate_model_markdown(model_data, index):
    """Generate markdown content for a single model"""
    name = model_data.get('name', 'Unknown Model')
    description = model_data.get('description', '')
    readme_content = model_data.get('readme_content', '')
    github_url = model_data.get('github_url', '')
    
    # Extract additional information
    framework, dataset = extract_framework_and_dataset(readme_content, description)
    category = categorize_model(name, description, readme_content)
    features = extract_key_features(readme_content)
    
    # Clean the readme content
    cleaned_readme = clean_markdown_content(readme_content)
    
    # Use GitHub date if available, otherwise use current date as fallback
    github_date = model_data.get('github_date') or model_data.get('created_date')
    if github_date:
        model_date = github_date
    else:
        model_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create the frontmatter
    frontmatter = f"""---
title: "{name}"
excerpt: "{description[:200]}{'...' if len(description) > 200 else ''}"
collection: models
layout: single
category: "{category}"
framework: "{framework}"
dataset: "{dataset}"
github_url: "{github_url}"
date: {model_date}
---

"""
    
    # Create the content
    content = frontmatter
    
    # Add a brief overview if description exists
    if description and description != cleaned_readme[:200]:
        content += f"## Overview\n{description}\n\n"
    
    # Add key features if any were found
    if features:
        content += "## Key Features\n"
        for feature in features:
            content += f"- {feature}\n"
        content += "\n"
    
    # Add technical details
    content += f"""## Technical Details
- **Framework**: {framework}
- **Dataset**: {dataset}
- **Category**: {category}

"""
    
    # Add the main readme content if it exists
    if cleaned_readme:
        content += "## Implementation Details\n\n"
        content += cleaned_readme
        content += "\n\n"
    
    # Validate if ModelArgs is relevant
    if 'ModelArgs' in readme_content and 'hyperparameters' in readme_content.lower():
        content += "## ModelArgs Hyperparameters\n\n"
        content += "| Parameter | Value | Description |\n"
        content += "|-----------|-------|-------------|\n"
        # Add logic to extract and format ModelArgs hyperparameters here
        # Ensure this section is only added if relevant to the model

    # Otherwise, skip adding ModelArgs section
    
    # Add GitHub link
    if github_url:
        content += f"""## Source Code
📁 **GitHub Repository**: [{name}]({github_url})

View the complete implementation, training scripts, and documentation on GitHub.
"""
    
    return content


def main():
    """Main function to generate all model markdown files"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate model markdown files')
    parser.add_argument('--source', default='paper-replications', 
                       help='Source repository type (default: paper-replications)')
    args = parser.parse_args()
    
    print(f"🔧 Generating model files for source: {args.source}")
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Define paths
    models_json_path = script_dir / '_data' / 'models.json'
    output_dir = script_dir / '_models'
    
    # Check if models.json exists
    if not models_json_path.exists():
        print(f"❌ Error: {models_json_path} not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Load the models data
    try:
        with open(models_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading models.json: {e}")
        return
    
    if 'models' not in data:
        print("❌ Error: 'models' key not found in JSON data")
        return
    
    models = data['models']
    
    # Clean up existing files
    for existing_file in output_dir.glob('*.md'):
        existing_file.unlink()
        print(f"🗑️  Removed old file: {existing_file.name}")
    
    # Generate markdown files for each model
    successful_count = 0
    for index, model in enumerate(models, 1):
        try:
            name = model.get('name', f'model-{index}')
            slug = slugify(name)
            filename = f"{index:02d}-{slug}.md"
            filepath = output_dir / filename
            
            # Generate the markdown content
            content = generate_model_markdown(model, index)
            
            # Write the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Generated: {filename}")
            successful_count += 1
            
        except Exception as e:
            print(f"❌ Error generating file for model {model.get('name', 'unknown')}: {e}")
    
    print(f"\n🎉 Successfully generated {successful_count} model files in {output_dir}")
    print(f"📁 Files are ready to be used by Jekyll!")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   - Total models processed: {len(models)}")
    print(f"   - Files generated: {successful_count}")
    print(f"   - Output directory: {output_dir}")


if __name__ == "__main__":
    main()
