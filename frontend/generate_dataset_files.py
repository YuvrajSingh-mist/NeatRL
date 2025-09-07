#!/usr/bin/env python3
"""
Generate dataset markdown files from datasets.json
This script creates standardized markdown files for all datasets in the datasets.json file.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

def load_datasets_data():
    """Load datasets data from JSON file"""
    datasets_file = "_data/datasets.json"
    if not os.path.exists(datasets_file):
        print(f"❌ Datasets file {datasets_file} not found!")
        return None
    
    with open(datasets_file, 'r') as f:
        return json.load(f)

def clean_content(content):
    """Clean and format content for Jekyll"""
    if not content:
        return ""
    
    # Convert Windows line endings to Unix
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove any triple quotes that might break frontmatter
    content = content.replace('"""', '"')
    content = content.replace("'''", "'")
    
    # Escape any YAML special characters in content
    content = content.replace('{{', '{{ "{{" }}')
    content = content.replace('}}', '{{ "}}" }}')
    
    return content

def convert_images_to_links(content, github_url):
    """Convert local image references to GitHub links"""
    if not content or not github_url:
        return content
    
    # Pattern to match image markdown: ![alt](path) or ![alt](./path) or images/file
    image_patterns = [
        (r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg|webp))\)', True),  # ![alt](image.png)
        (r'<img[^>]+src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg|webp))["\'][^>]*>', False),  # <img src="image.png">
        (r'((?:images?|assets?|figures?)/[^)\s]+\.(?:png|jpg|jpeg|gif|svg|webp))', False)  # images/file.png
    ]
    
    base_url = github_url.replace('/tree/main/', '/raw/main/').replace('/tree/master/', '/raw/master/')
    
    for pattern, is_markdown in image_patterns:
        def replace_image(match):
            if is_markdown:
                alt_text = match.group(1)
                image_path = match.group(2)
                if not image_path.startswith('http'):
                    if image_path.startswith('./'):
                        image_path = image_path[2:]
                    full_url = f"{base_url}/{image_path}"
                    return f"![{alt_text}]({full_url})"
                return match.group(0)
            else:
                # Handle other patterns
                return match.group(0)  # For now, keep as-is
        
        content = re.sub(pattern, replace_image, content, flags=re.IGNORECASE)
    
    return content

def generate_frontmatter(dataset):
    """Generate YAML frontmatter for dataset"""
    tags = dataset.get('tags', [])
    if isinstance(tags, str):
        tags = [tags]
    
    tasks = dataset.get('tasks', [])
    if isinstance(tasks, str):
        tasks = [tasks]
    
    # Create excerpt from description
    description = dataset.get('description', '')
    if len(description) > 200:
        excerpt = description[:197] + "..."
    else:
        excerpt = description
    
    # Add image placeholder
    excerpt += '<br/><img src=\'/images/500x300.png\'>'
    
    frontmatter = f"""---
title: {dataset.get('display_name', dataset.get('name', 'Dataset'))}
excerpt: "{excerpt}"
collection: datasets
github_url: {dataset.get('github_url', '')}
size: {dataset.get('size', 'Unknown')}
format: {dataset.get('format', 'Unknown')}
samples: {dataset.get('samples', 'Unknown')}
license: {dataset.get('license', 'Unknown')}
last_updated: {dataset.get('last_updated', datetime.now().isoformat())}
tags:"""
    
    for tag in tags:
        frontmatter += f"\n  - {tag}"
    
    frontmatter += "\ntasks:"
    for task in tasks:
        frontmatter += f"\n  - {task}"
    
    frontmatter += "\n---"
    
    return frontmatter

def generate_dataset_content(dataset):
    """Generate the main content for the dataset markdown"""
    name = dataset.get('display_name', dataset.get('name', 'Dataset'))
    description = dataset.get('description', '')
    github_url = dataset.get('github_url', '')
    
    content = f"""
## Dataset Overview
{description}

## Technical Details
- **Size**: {dataset.get('size', 'Unknown')}
- **Format**: {dataset.get('format', 'Unknown')}
- **Samples**: {dataset.get('samples', 'Unknown')}
- **License**: {dataset.get('license', 'Unknown')}
- **Repository**: [{name}]({github_url})

## Applications
This dataset can be used for:
"""
    
    tasks = dataset.get('tasks', [])
    for task in tasks:
        content += f"- **{task}**: Training and evaluation of machine learning models\n"
    
    content += f"""
## Access and Usage

### Download
📁 **GitHub Repository**: [{name}]({github_url})

Visit the repository to download the dataset and view detailed documentation.

### Citation
When using this dataset, please cite the original repository:
```
{name}. Available at: {github_url}
```

## Dataset Information
- **Last Updated**: {dataset.get('last_updated', 'Unknown')}
- **Format**: {dataset.get('format', 'Unknown')}
- **Size**: {dataset.get('size', 'Unknown')}

"""
    
    # Add README content if available
    readme_content = dataset.get('readme_content', '')
    if readme_content and len(readme_content) > 100:
        # Clean and convert images
        readme_content = clean_content(readme_content)
        readme_content = convert_images_to_links(readme_content, github_url)
        
        content += f"""## Detailed Documentation

{readme_content}
"""
    
    content += f"""
---

*This dataset is part of the Datasets Collection - a curated collection of high-quality datasets for machine learning research and applications.*
"""
    
    return content

def generate_filename(dataset):
    """Generate filename for the dataset markdown file"""
    name = dataset.get('name', 'dataset')
    # Convert to lowercase and replace spaces/special chars with hyphens
    filename = re.sub(r'[^a-zA-Z0-9\-_]', '-', name.lower())
    filename = re.sub(r'-+', '-', filename)  # Remove multiple consecutive hyphens
    filename = filename.strip('-')  # Remove leading/trailing hyphens
    return f"{filename}.md"

def create_dataset_markdown(dataset):
    """Create a complete markdown file for a dataset"""
    frontmatter = generate_frontmatter(dataset)
    content = generate_dataset_content(dataset)
    
    return frontmatter + content

def main():
    """Main function to generate all dataset markdown files"""
    print("🔄 Generating dataset markdown files...")
    
    # Load datasets data
    datasets_data = load_datasets_data()
    if not datasets_data:
        return False
    
    datasets = datasets_data.get('datasets', [])
    if not datasets:
        print("⚠️ No datasets found in datasets.json")
        return False
    
    # Create _datasets directory if it doesn't exist
    datasets_dir = Path("_datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    generated_count = 0
    for dataset in datasets:
        try:
            # Generate filename
            filename = generate_filename(dataset)
            filepath = datasets_dir / filename
            
            # Generate markdown content
            markdown_content = create_dataset_markdown(dataset)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"  ✅ Generated: {filename}")
            generated_count += 1
            
        except Exception as e:
            dataset_name = dataset.get('name', 'Unknown')
            print(f"  ❌ Error generating {dataset_name}: {e}")
    
    print(f"\n✅ Successfully generated {generated_count} dataset markdown files!")
    print(f"📁 Files saved in: {datasets_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
