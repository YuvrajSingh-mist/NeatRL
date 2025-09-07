#!/usr/bin/env python3
"""
Comprehensive automation script to standardize all model markdown files
and create templates for future markdown creation.

This script will:
1. Fix all existing markdown files to follow consistent patterns
2. Standardize frontmatter format
3. Ensure proper image links using raw.githubusercontent.com
4. Create templates for future use
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

class MarkdownStandardizer:
    def __init__(self):
        self.models_dir = Path("_models")
        self.data_dir = Path("_data")
        self.templates_dir = Path("_templates")
        self.fixed_count = 0
        self.total_count = 0
        
    def load_models_json(self):
        """Load the models.json file to get canonical data"""
        models_file = self.data_dir / "models.json"
        if not models_file.exists():
            print("❌ models.json not found!")
            return {}
        
        with open(models_file, 'r') as f:
            data = json.load(f)
        
        # Handle both formats: {"models": [...]} or [...]
        models_list = data.get('models', data) if isinstance(data, dict) else data
        
        # Create a lookup dictionary by name
        models_lookup = {}
        for model in models_list:
            if isinstance(model, dict) and 'name' in model:
                models_lookup[model['name']] = model
        
        return models_lookup
    
    def extract_model_name_from_filename(self, filename):
        """Extract model name from filename (e.g., '38-lstm.md' -> 'lstm')"""
        # Remove number prefix and .md suffix
        name = re.sub(r'^\d+-', '', filename)
        name = re.sub(r'\.md$', '', name)
        return name
    
    def standardize_frontmatter(self, content, filename, models_data):
        """Standardize the frontmatter section"""
        model_name = self.extract_model_name_from_filename(filename)
        model_info = models_data.get(model_name, {})
        
        # Extract existing frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if not frontmatter_match:
            print(f"  ⚠️  No frontmatter found in {filename}")
            return content
        
        frontmatter_content = frontmatter_match.group(1)
        body_content = frontmatter_match.group(2)
        
        # Parse existing frontmatter
        frontmatter_data = {}
        for line in frontmatter_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter_data[key.strip()] = value.strip().strip('"')
        
        # Use models.json data if available, otherwise keep existing
        title = model_info.get('name', frontmatter_data.get('title', model_name))
        excerpt = model_info.get('description', frontmatter_data.get('excerpt', f'From scratch implementation of {model_name}'))
        github_url = model_info.get('github_url', frontmatter_data.get('github_url', ''))
        
        # Determine category, framework, dataset from existing data or defaults
        category = frontmatter_data.get('category', 'Machine Learning')
        framework = frontmatter_data.get('framework', 'PyTorch')
        dataset = frontmatter_data.get('dataset', 'Custom')
        date = frontmatter_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Create standardized frontmatter
        new_frontmatter = f'''---
title: "{title}"
excerpt: "{excerpt}"
collection: models
layout: model-implementation
category: "{category}"
framework: "{framework}"
dataset: "{dataset}"
github_url: "{github_url}"
date: {date}
---'''
        
        return new_frontmatter + '\n\n' + body_content
    
    def fix_github_image_url(self, url):
        """Convert various GitHub image URL forms to raw.githubusercontent.com URL"""
        # Already raw
        if url.startswith('https://raw.githubusercontent.com/'):
            return url
        # Convert github.com/blob/... -> raw
        if 'github.com' in url and '/blob/' in url:
            return url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        # Convert github.com/.../raw/... -> raw.githubusercontent.com
        if 'github.com' in url and '/raw/' in url:
            return url.replace('github.com', 'raw.githubusercontent.com').replace('/raw/', '/')
        return url
    
    def standardize_body_content(self, content):
        """Standardize the body content structure and fix image links"""
        
        # Try to extract github_url from frontmatter to construct raw base URL
        github_url_match = re.search(r'github_url:\s*"([^"]+)"', content)
        raw_base_url = None
        if github_url_match:
            github_url = github_url_match.group(1)
            # Convert https://github.com/user/repo/tree/{branch}/path -> https://raw.githubusercontent.com/user/repo/{branch}/path
            raw_base_url = (
                github_url
                .replace('https://github.com/', 'https://raw.githubusercontent.com/')
                .replace('/tree/master/', '/master/')
                .replace('/tree/main/', '/main/')
            )
        
        # Remove local image references
        content = re.sub(r'<!-- Main image reference -->\s*<img src="[^"]*" alt="[^"]*"[^>]*>\s*', '', content, flags=re.MULTILINE)
        
        # If we have a raw base URL, convert local markdown image references to raw.githubusercontent.com URLs
        if raw_base_url:
            def replace_local_md_image(match):
                alt_text = match.group(1)
                image_path = match.group(2)
                if image_path.startswith('http'):
                    return match.group(0)
                clean_path = image_path.lstrip('./').lstrip('/')
                return f'![{alt_text}]({raw_base_url}/{clean_path})'

            local_md_image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:jpg|jpeg|png|gif|svg|webp))\)'
            content = re.sub(local_md_image_pattern, replace_local_md_image, content)

            # Convert local HTML <img> tags as well
            def replace_local_html_image(match):
                src = match.group(1)
                if src.startswith('http'):
                    return match.group(0)
                clean_path = src.lstrip('./').lstrip('/')
                return match.group(0).replace(src, f'{raw_base_url}/{clean_path}')

            local_html_image_pattern = r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png|gif|svg|webp))["\'][^>]*>'
            content = re.sub(local_html_image_pattern, replace_local_html_image, content, flags=re.IGNORECASE)

        # Fix GitHub image links - convert [text](github_url) to ![text](raw_github_url)
        github_link_pattern = r'\[([^\]]+)\]\((https://github\.com/[^)]+\.(jpg|jpeg|png|gif|svg))\)'
        def replace_github_link(match):
            text = match.group(1)
            url = match.group(2)
            fixed_url = self.fix_github_image_url(url)
            return f'![{text}]({fixed_url})'
        
        content = re.sub(github_link_pattern, replace_github_link, content)
        
        # Fix any remaining GitHub blob URLs
        blob_url_pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/([^)\s]+\.(jpg|jpeg|png|gif|svg))'
        def replace_blob_url(match):
            user = match.group(1)
            repo = match.group(2)
            branch = match.group(3)
            path = match.group(4)
            return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}'
        
        content = re.sub(blob_url_pattern, replace_blob_url, content)

        # Fix github.com/.../raw/... links that slipped through
        raw_path_pattern = r'https://github\.com/([^/]+)/([^/]+)/raw/([^/]+)/([^)\s]+\.(jpg|jpeg|png|gif|svg))'
        def replace_raw_path(match):
            user, repo, branch, path = match.groups()
            return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}'
        content = re.sub(raw_path_pattern, replace_raw_path, content)
        
        # Remove fallback reference comments
        content = re.sub(r'<!-- Fallback reference -->\s*\n', '', content)
        
        # Clean up multiple empty lines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Ensure standard sections exist
        if '## Overview' not in content:
            title_match = re.search(r'title: "([^"]+)"', content)
            title = title_match.group(1) if title_match else 'Model'
            overview_section = f'''
## Overview
From scratch implementation of {title}

## Technical Details
- **Framework**: PyTorch
- **Dataset**: Custom
- **Category**: Machine Learning

## Implementation Details
'''
            # Insert after frontmatter
            frontmatter_end = content.find('---\n', 4) + 4
            content = content[:frontmatter_end] + overview_section + content[frontmatter_end:]
        
        return content
    
    def process_file(self, file_path, models_data):
        """Process a single markdown file"""
        filename = file_path.name
        print(f"📝 Processing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Standardize frontmatter
        content = self.standardize_frontmatter(content, filename, models_data)
        
        # Standardize body content
        content = self.standardize_body_content(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Standardized {filename}")
            return True
        else:
            print(f"  ℹ️  No changes needed for {filename}")
            return False
    
    def create_templates(self):
        """Create templates for future markdown creation"""
        self.templates_dir.mkdir(exist_ok=True)
        
        # Create model template
        model_template = '''---
title: "{MODEL_NAME}"
excerpt: "From scratch implementation of {MODEL_NAME}"
collection: models
layout: model-implementation
category: "{CATEGORY}"
framework: "PyTorch"
dataset: "{DATASET}"
github_url: "{GITHUB_URL}"
date: {DATE}
---

## Overview
From scratch implementation of {MODEL_NAME}

## Technical Details
- **Framework**: PyTorch
- **Dataset**: {DATASET}
- **Category**: {CATEGORY}

## Implementation Details

# {MODEL_NAME} Implementation

{DESCRIPTION}

## Model Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 32       | The number of samples processed before the model is updated.                |
| `learning_rate` | 1e-4   | Learning rate for optimization.                                             |
| `epochs`     | 50       | Number of training epochs.                                                  |

### Training Details

- **Framework**: PyTorch
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

### Results

Train loss - TBD
Val loss - TBD

### Loss Curves

![📊 View Training Loss Curves]({LOSS_CURVE_URL})

## Source Code
📁 **GitHub Repository**: [{MODEL_NAME}]({GITHUB_URL})

View the complete implementation, training scripts, and documentation on GitHub.
'''
        
        template_path = self.templates_dir / "model_template.md"
        with open(template_path, 'w') as f:
            f.write(model_template)
        
        print(f"📄 Created template: {template_path}")
        
        # Create automation script for new models
        automation_script = '''#!/usr/bin/env python3
"""
Script to create new model markdown files from template
Usage: python create_new_model.py MODEL_NAME CATEGORY DATASET GITHUB_URL
"""

import sys
import re
from pathlib import Path
from datetime import datetime

def create_new_model(model_name, category="Machine Learning", dataset="Custom", github_url="", description=""):
    """Create a new model markdown file from template"""
    
    templates_dir = Path("_templates")
    models_dir = Path("_models")
    
    template_path = templates_dir / "model_template.md"
    if not template_path.exists():
        print("❌ Template file not found!")
        return False
    
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Find next available number
    existing_files = list(models_dir.glob("*.md"))
    existing_numbers = []
    for file in existing_files:
        match = re.match(r'^(\\d+)-', file.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    next_number = max(existing_numbers, default=0) + 1
    filename = f"{next_number:02d}-{model_name.lower().replace(' ', '-')}.md"
    
    # Prepare loss curve URL
    loss_curve_url = f"https://raw.githubusercontent.com/YuvrajSingh-mist/Paper-Replications/master/{model_name}/img/loss.jpg"
    if not github_url:
        github_url = f"https://github.com/YuvrajSingh-mist/Paper-Replications/tree/master/{model_name}"
    
    # Replace placeholders
    content = template_content.replace("{MODEL_NAME}", model_name)
    content = content.replace("{CATEGORY}", category)
    content = content.replace("{DATASET}", dataset)
    content = content.replace("{GITHUB_URL}", github_url)
    content = content.replace("{DATE}", datetime.now().strftime('%Y-%m-%d'))
    content = content.replace("{DESCRIPTION}", description or f"Implementation of {model_name} from scratch")
    content = content.replace("{LOSS_CURVE_URL}", loss_curve_url)
    
    # Write new file
    output_path = models_dir / filename
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created new model file: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_new_model.py MODEL_NAME [CATEGORY] [DATASET] [GITHUB_URL]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "Machine Learning"
    dataset = sys.argv[3] if len(sys.argv) > 3 else "Custom"
    github_url = sys.argv[4] if len(sys.argv) > 4 else ""
    
    create_new_model(model_name, category, dataset, github_url)
'''
        
        script_path = Path("create_new_model.py")
        with open(script_path, 'w') as f:
            f.write(automation_script)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        print(f"🤖 Created automation script: {script_path}")
    
    def run(self):
        """Run the standardization process"""
        print("🔧 SmolHub Markdown Standardization")
        print("=" * 50)
        
        if not self.models_dir.exists():
            print("❌ _models directory not found!")
            return
        
        # Load models data
        print("📊 Loading models.json data...")
        models_data = self.load_models_json()
        print(f"  Found {len(models_data)} models in JSON")
        
        # Process all markdown files
        print("\\n📝 Processing markdown files...")
        for md_file in sorted(self.models_dir.glob("*.md")):
            self.total_count += 1
            if self.process_file(md_file, models_data):
                self.fixed_count += 1
        
        # Create templates and automation
        print("\\n📄 Creating templates and automation...")
        self.create_templates()
        
        print("\\n" + "=" * 50)
        print(f"📊 Summary: Standardized {self.fixed_count}/{self.total_count} files")
        
        if self.fixed_count > 0:
            print("✅ All markdown files have been standardized!")
            print("🎯 Key improvements:")
            print("   - Consistent frontmatter format")
            print("   - Proper layout: model-implementation")
            print("   - Fixed image links using raw.githubusercontent.com")
            print("   - Standardized section structure")
        
        print("\\n🤖 Automation created:")
        print("   - Template: _templates/model_template.md")
        print("   - Script: create_new_model.py")
        print("\\n📝 To create new models:")
        print("   python create_new_model.py 'Model Name' 'Category' 'Dataset' 'GitHub URL'")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Standardize markdown files')
    parser.add_argument('--target', default='models', 
                       help='Target type to standardize (default: models)')
    args = parser.parse_args()
    
    print(f"🔧 Standardizing {args.target} markdown files...")
    
    standardizer = MarkdownStandardizer()
    standardizer.run()
