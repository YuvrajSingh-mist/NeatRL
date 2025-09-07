#!/usr/bin/env python3
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
        match = re.match(r'^(\d+)-', file.name)
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
