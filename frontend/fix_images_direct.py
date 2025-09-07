#!/usr/bin/env python3
"""
Convert local image references to GitHub hyperlinks in all model markdown files
Uses the github_url from each file's frontmatter directly
"""

import os
import re

def convert_image_to_link(match, raw_base_url):
    """Convert an image markdown to a hyperlink"""
    alt_text = match.group(1)
    image_path = match.group(2)
    
    # Create appropriate link text based on alt text
    if 'loss' in alt_text.lower():
        link_text = "📊 View Training Loss Curves"
    elif 'train' in alt_text.lower() and 'val' in alt_text.lower():
        link_text = "📈 View Train and Validation Loss"
    elif 'result' in alt_text.lower():
        link_text = "🖼️ View Results"
    elif 'sample' in alt_text.lower():
        link_text = "🎨 View Generated Samples"
    elif 'architecture' in alt_text.lower():
        link_text = "🏗️ View Model Architecture"
    elif 'output' in alt_text.lower():
        link_text = "📋 View Model Output"
    elif 'arithmetic' in alt_text.lower():
        link_text = "🔢 View Latent Arithmetic"
    else:
        link_text = f"🔗 View {alt_text}"
    
    # Create raw GitHub URL
    github_image_url = f"{raw_base_url}/{image_path}"
    
    # Return as hyperlink
    return f"[{link_text}]({github_image_url})"

def update_markdown_file(filepath):
    """Update a single markdown file"""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract GitHub URL from frontmatter
    github_url_match = re.search(r'github_url:\s*"([^"]+)"', content)
    if not github_url_match:
        print(f"  ⚠️  No github_url found in {filepath}")
        return False
    
    github_url = github_url_match.group(1)
    # Convert to raw base URL
    raw_base_url = (
        github_url
        .replace('https://github.com/', 'https://raw.githubusercontent.com/')
        .replace('/tree/master/', '/master/')
        .replace('/tree/main/', '/main/')
    )
    
    # Find all image references
    image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(jpg|jpeg|png|gif|svg))\)'
    images_found = re.findall(image_pattern, content)
    
    if not images_found:
        print(f"  ✅ No images found in {filepath}")
        return False
    
    print(f"  🖼️  Found {len(images_found)} images: {[img[1] for img in images_found]}")
    
    # Replace all image references with hyperlinks
    def replace_image(match):
        return convert_image_to_link(match, raw_base_url)
    
    updated_content = re.sub(image_pattern, replace_image, content)
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✅ Updated {len(images_found)} image references in {filepath}")
    return True

def main():
    """Main function to update all markdown files"""
    models_dir = "_models"
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory {models_dir} not found")
        return
    
    # Get all markdown files
    markdown_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(models_dir, filename)
            markdown_files.append(filepath)
    
    print(f"🔍 Found {len(markdown_files)} markdown files to process")
    
    # Process each file
    updated_count = 0
    for filepath in sorted(markdown_files):
        if update_markdown_file(filepath):
            updated_count += 1
    
    print(f"\n🎉 Successfully updated {updated_count} markdown files!")
    print("📝 All local image references have been converted to GitHub hyperlinks")

if __name__ == "__main__":
    main()
