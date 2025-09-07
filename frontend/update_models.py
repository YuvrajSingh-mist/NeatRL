#!/usr/bin/env python3
"""
Automated script to fetch latest models from GitHub Paper-Replications repo
and generate markdown files for the Jekyll website.

This script can:
1. Fetch the latest data from GitHub API
2. Update the local models.json file  
3. Generate markdown files for all models
4. Be run automatically via GitHub Actions or manually

Usage:
    python update_models.py                    # Use existing models.json
    python update_models.py --fetch            # Fetch from GitHub first
    python update_models.py --github-token TOKEN  # Use with GitHub token for higher rate limits
"""

import json
import os
import re
import argparse
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


def fetch_github_data(github_token=None):
    """Fetch the latest models data from GitHub API"""
    print("🔄 Fetching latest data from GitHub...")
    
    # GitHub API endpoint for the Paper Replications repo
    repo_owner = "YuvrajSingh-mist"
    repo_name = "Paper-Replications"
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
    
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if github_token:
        headers['Authorization'] = f'token {github_token}'
        print("🔑 Using GitHub token for authenticated requests")
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        folders = response.json()
        models = []
        
        for folder in folders:
            if folder['type'] == 'dir' and folder['name'] not in ['.git', '.github', 'README.md']:
                folder_name = folder['name']
                print(f"📁 Processing folder: {folder_name}")
                
                # Fetch README content for this folder
                readme_url = f"{api_url}/{quote(folder_name)}/README.md"
                readme_content = ""
                
                try:
                    readme_response = requests.get(readme_url, headers=headers)
                    if readme_response.status_code == 200:
                        readme_data = readme_response.json()
                        if readme_data.get('content'):
                            import base64
                            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                except Exception as e:
                    print(f"⚠️  Warning: Could not fetch README for {folder_name}: {e}")
                
                # Create model entry
                model = {
                    "name": folder_name,
                    "display_name": folder_name.replace('-', ' ').replace('_', ' ').title(),
                    "description": f"From scratch implementation of {folder_name}",
                    "readme_content": readme_content,
                    "github_url": f"https://github.com/{repo_owner}/{repo_name}/tree/master/{folder_name}",
                    "api_url": f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{quote(folder_name)}?ref=master",
                    "download_url": None
                }
                
                models.append(model)
        
        print(f"✅ Successfully fetched {len(models)} models from GitHub")
        return {"models": models}
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from GitHub: {e}")
        print("💡 You may have hit rate limits. Try using a GitHub token with --github-token")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def slugify(text):
    """Convert text to a URL-friendly slug"""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def clean_markdown_content(content):
    """Clean and format markdown content"""
    if not content:
        return ""
    
    content = content.strip()
    content = re.sub(r'\r\n', '\n', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content


def extract_framework_and_dataset(readme_content, description):
    """Extract framework and dataset information from readme content"""
    framework = "PyTorch"
    dataset = "Custom"
    
    content_lower = (readme_content + " " + description).lower()
    
    if 'pytorch' in content_lower:
        framework = "PyTorch"
    elif 'tensorflow' in content_lower:
        framework = "TensorFlow"
    elif 'jax' in content_lower:
        framework = "JAX"
    
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
        'fineweb': 'FineWeb',
        'shakespeare': 'Shakespeare'
    }
    
    for pattern, name in dataset_patterns.items():
        if pattern in content_lower:
            dataset = name
            break
    
    return framework, dataset


def categorize_model(name, description, readme_content):
    """Manual categorization - use existing category from model data or default"""
    # For update_models.py, we'll preserve existing categories
    # New models will get "Other" and should be manually categorized later
    return "Other"


def extract_key_features(readme_content):
    """Extract key features from readme content"""
    features = []
    
    content_lower = readme_content.lower()
    
    if 'mixture of experts' in content_lower or 'moe' in content_lower:
        features.append("Mixture of Experts (MoE)")
    if 'attention' in content_lower:
        features.append("Attention Mechanism")
    if 'transformer' in content_lower:
        features.append("Transformer Architecture")
    if 'distributed' in content_lower or 'ddp' in content_lower:
        features.append("Distributed Training")
    if 'gradient checkpointing' in content_lower:
        features.append("Memory Optimization")
    if 'fine-tuning' in content_lower or 'finetuning' in content_lower:
        features.append("Fine-tuning")
    if 'multimodal' in content_lower:
        features.append("Multimodal")
    if 'vision' in content_lower and 'language' in content_lower:
        features.append("Vision-Language")
    
    return features


def generate_model_markdown(model_data, index):
    """Generate markdown content for a single model"""
    name = model_data.get('name', 'Unknown Model')
    display_name = model_data.get('display_name', name)
    description = model_data.get('description', '')
    readme_content = model_data.get('readme_content', '')
    github_url = model_data.get('github_url', '')

    # Extract additional information
    framework, dataset = extract_framework_and_dataset(readme_content, description)
    category = categorize_model(name, description, readme_content)
    features = extract_key_features(readme_content)

    # Clean the readme content
    cleaned_readme = clean_markdown_content(readme_content)

    # Convert image URLs to raw GitHub user content URLs
    cleaned_readme = re.sub(
        r'!\[(.*?)\]\((.*?)\)',
        lambda match: f"![{match.group(1)}](https://raw.githubusercontent.com/YuvrajSingh-mist/{match.group(2).replace('blob/', '')})",
        cleaned_readme
    )

    # Create the frontmatter
    frontmatter = f"""---
    title: "{display_name}"
    excerpt: "{description[:200]}{'...' if len(description) > 200 else ''}"
    collection: paper_replications
    layout: model-implementation
    category: "{category}"
    framework: "{framework}"
    dataset: "{dataset}"
    github_url: "{github_url}"
    date: {datetime.now().strftime('%Y-%m-%d')}
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

    # Add GitHub link
    if github_url:
        content += f"""## Source Code
        📁 **GitHub Repository**: [{name}]({github_url})

        View the complete implementation, training scripts, and documentation on GitHub.
        """

    return content


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Update Paper Replication models')
    parser.add_argument('--fetch', action='store_true', 
                       help='Fetch latest data from GitHub API')
    parser.add_argument('--github-token', type=str,
                       help='GitHub token for API authentication')
    parser.add_argument('--output-dir', default='_models', 
                        help='Output directory for markdown files')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    models_json_path = script_dir / '_data' / 'models.json'
    output_dir = script_dir / args.output_dir
    
    # Fetch from GitHub if requested
    if args.fetch:
        github_data = fetch_github_data(args.github_token)
        if github_data:
            # Backup existing models.json
            if models_json_path.exists():
                backup_path = models_json_path.with_suffix('.json.backup')
                models_json_path.rename(backup_path)
                print(f"📄 Backed up existing models.json to {backup_path}")
            
            # Save new data
            with open(models_json_path, 'w', encoding='utf-8') as f:
                json.dump(github_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved updated models.json with {len(github_data['models'])} models")
        else:
            print("❌ Failed to fetch data from GitHub, using existing models.json")
    
    # Check if models.json exists
    if not models_json_path.exists():
        print(f"❌ Error: {models_json_path} not found!")
        print("💡 Try running with --fetch to download from GitHub")
        return
    
    # Create output directory
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
    categories = {}
    frameworks = {}
    for model in models:
        readme = model.get('readme_content', '')
        desc = model.get('description', '')
        cat = categorize_model(model.get('name', ''), desc, readme)
        fw, _ = extract_framework_and_dataset(readme, desc)
        
        categories[cat] = categories.get(cat, 0) + 1
        frameworks[fw] = frameworks.get(fw, 0) + 1
    
    print(f"\n📊 Summary:")
    print(f"   - Total models processed: {len(models)}")
    print(f"   - Files generated: {successful_count}")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Categories: {dict(sorted(categories.items()))}")
    print(f"   - Frameworks: {dict(sorted(frameworks.items()))}")


if __name__ == "__main__":
    main()
