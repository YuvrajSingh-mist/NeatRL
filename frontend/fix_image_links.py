#!/usr/bin/env python3
"""
Fix all image links in model markdown files to use raw.githubusercontent.com URLs
This will replace broken GitHub blob URLs and remove local image references
"""

import os
import re
from pathlib import Path

def fix_github_image_url(url):
    """Convert GitHub blob URL to raw.githubusercontent.com URL"""
    if 'github.com' in url and '/blob/' in url:
        # Convert github.com/user/repo/blob/branch/path to raw.githubusercontent.com/user/repo/branch/path
        return url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    return url

def process_file(file_path):
    """Process a single markdown file to fix image links"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Remove local image references like <img src="data/..."/>
    local_img_pattern = r'<!-- Main image reference -->\s*<img src="[^"]*" alt="[^"]*"[^>]*>\s*'
    content = re.sub(local_img_pattern, '', content, flags=re.MULTILINE)
    
    # Pattern 2: Fix GitHub image links in markdown format [text](github_url)
    github_link_pattern = r'\[([^\]]+)\]\((https://github\.com/[^)]+\.(jpg|jpeg|png|gif|svg))\)'
    def replace_github_link(match):
        text = match.group(1)
        url = match.group(2)
        fixed_url = fix_github_image_url(url)
        return f'![{text}]({fixed_url})'
    
    content = re.sub(github_link_pattern, replace_github_link, content)
    
    # Pattern 3: Fix any remaining GitHub blob URLs
    blob_url_pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/([^)\s]+\.(jpg|jpeg|png|gif|svg))'
    def replace_blob_url(match):
        user = match.group(1)
        repo = match.group(2)
        branch = match.group(3)
        path = match.group(4)
        return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}'
    
    content = re.sub(blob_url_pattern, replace_blob_url, content)
    
    # Remove empty lines that might be left behind
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Remove fallback reference comments if they're now redundant
    content = re.sub(r'<!-- Fallback reference -->\s*\n', '', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Fixed image links in {file_path}")
        return True
    else:
        print(f"  ℹ️  No changes needed in {file_path}")
        return False

def main():
    """Fix image links in all model files"""
    print("🔧 Fixing image links in model files...")
    print("=" * 50)
    
    models_dir = Path("_models")
    if not models_dir.exists():
        print("❌ _models directory not found!")
        return
    
    fixed_count = 0
    total_count = 0
    
    for md_file in sorted(models_dir.glob("*.md")):
        total_count += 1
        if process_file(md_file):
            fixed_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: Fixed {fixed_count}/{total_count} files")
    
    if fixed_count > 0:
        print("✅ Image links have been fixed!")
        print("🔗 All GitHub image URLs now use raw.githubusercontent.com")
        print("🗑️  Local image references have been removed")
    else:
        print("ℹ️  No fixes were needed")

if __name__ == "__main__":
    main()
