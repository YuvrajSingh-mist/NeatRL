#!/usr/bin/env python3
"""
Fix all model markdown files to remove 'Paper Replications' references
and improve the content to reflect 'From Scratch Implementation'
"""

import os
import re
import json

def fix_model_file(file_path):
    """Fix a single model markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update excerpt to remove "Paper Replications repository"
    content = re.sub(
        r'(excerpt:\s*["\'][^"\']*?)from the Paper Replications repository([^"\']*["\'])',
        r'\1from scratch\2',
        content
    )
    
    content = re.sub(
        r'(excerpt:\s*["\'][^"\']*?)Paper Replications repository([^"\']*["\'])',
        r'\1From Scratch Implementation\2',
        content
    )
    
    # Update overview section
    content = re.sub(
        r'## Overview\s*\nImplementation of ([^.]+) from the Paper Replications repository',
        r'## Overview\nFrom scratch implementation of \1.',
        content
    )
    
    # Replace other "Paper Replications" references in content
    content = content.replace('Paper Replications repository', 'From Scratch Implementation')
    content = content.replace('from the Paper Replications', 'from scratch')
    content = content.replace('Paper-Replications', 'From-Scratch-Implementation')
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {os.path.basename(file_path)}")
        return True
    else:
        print(f"No changes needed: {os.path.basename(file_path)}")
        return False

def main():
    """Main function to fix all model files"""
    models_dir = '_models'
    
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found!")
        return
    
    # Get all markdown files in models directory
    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.md'):
            model_files.append(os.path.join(models_dir, filename))
    
    print(f"Found {len(model_files)} model files to check")
    
    # Fix each file
    fixed_count = 0
    for file_path in sorted(model_files):
        if fix_model_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {len(model_files)} model files!")

if __name__ == '__main__':
    main()
