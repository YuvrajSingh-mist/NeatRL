#!/usr/bin/env python3
"""
Script to refresh RL implementations from the Reinforcement-Learning GitHub repository.
This script fetches the folder structure from GitHub API and generates both 
JSON data file and individual Jekyll markdown files.
"""

import os
import json
import requests
import re
from datetime import datetime
from pathlib import Path
import argparse
import posixpath

# Configuration
REPO_OWNER = "YuvrajSingh-mist"
REPO_NAME = "Reinforcement-Learning"
BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
DATA_FILE = "_data/rl.json"
MODELS_DIR = "_rl"
BASE_PATH = Path(__file__).parent

# GitHub Authentication
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
if GITHUB_TOKEN:
    HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
    print("🔑 Using GitHub token authentication")
else:
    HEADERS = {}
    print("⚠️  No GitHub token found - using unauthenticated requests (rate limited)")

def fetch_github_content(path=""):
    """Fetch repository content from GitHub API."""
    url = f"{BASE_URL}/contents/{path}"
    print(f"🔍 Fetching: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def fetch_readme_content(path):
    """Fetch README content from a specific folder."""
    readme_file = "README.md"  # Always README.md
    
    readme_path = f"{path}/{readme_file}" if path else readme_file
    content = fetch_github_content(readme_path)
    
    if content and isinstance(content, dict) and content.get('download_url'):
        try:
            readme_response = requests.get(content['download_url'], headers=HEADERS)
            readme_response.raise_for_status()
            # Ensure we get the FULL content without any truncation
            full_content = readme_response.text
            print(f"✅ README fetched for {path}: {len(full_content)} characters")
            return full_content
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching README for {path}: {e}")
            return ""
    
    return ""

def absolutize_markdown_links(markdown_text, base_path):
    """Convert relative markdown and HTML links/images to absolute raw.githubusercontent.com URLs.

    - Handles markdown images ![alt](src) and links [text](href)
    - Handles HTML <img src="..."> and <a href="..."> attributes
    - Leaves http(s), mailto, and anchor (#) links unchanged
    """
    if not markdown_text:
        return markdown_text

    def to_absolute(url):
        if not url:
            return url
        url = url.strip()
        # Ignore absolute and anchor/mailto links
        if re.match(r"^(?:https?:)?//", url) or url.startswith("http://") or url.startswith("https://") or url.startswith("mailto:") or url.startswith("#"):
            return url
        # Normalize path relative to repo
        if url.startswith('/'):
            resolved = url.lstrip('/')
        else:
            # Join with the folder that the README belongs to
            joined = posixpath.join(base_path, url) if base_path else url
            resolved = posixpath.normpath(joined)
        return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/master/{resolved}"

    # Markdown images ![alt](...)
    def repl_md_image(match):
        alt = match.group(1)
        url = match.group(2)
        title = match.group(3) or ''
        abs_url = to_absolute(url)
        return f"![{alt}]({abs_url}{title})"

    # Markdown links [text](...)
    def repl_md_link(match):
        text = match.group(1)
        url = match.group(2)
        title = match.group(3) or ''
        abs_url = to_absolute(url)
        return f"[{text}]({abs_url}{title})"

    # HTML src and href attributes
    def repl_html_src(match):
        prefix = match.group(1)
        url = match.group(2)
        suffix = match.group(3)
        abs_url = to_absolute(url)
        return f"{prefix}{abs_url}{suffix}"

    # Patterns (support optional title in markdown links/images: (url "title"))
    md_image_pattern = re.compile(r"!\[([^\]]*)\]\(([^\)\s]+)(\s+\"[^\"]*\")?\)")
    md_link_pattern = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^\)\s]+)(\s+\"[^\"]*\")?\)")
    html_src_pattern = re.compile(r"(src=\")([^\"]+)(\")", re.IGNORECASE)
    html_href_pattern = re.compile(r"(href=\")([^\"]+)(\")", re.IGNORECASE)

    # Apply replacements
    updated = md_image_pattern.sub(repl_md_image, markdown_text)
    updated = md_link_pattern.sub(repl_md_link, updated)
    updated = html_src_pattern.sub(repl_html_src, updated)
    updated = html_href_pattern.sub(repl_html_src, updated)
    return updated

def categorize_rl_algorithm(name, path, readme_content=""):
    """Categorize RL algorithms based on name and content (single primary category)."""
    name_lower = name.lower()
    path_lower = path.lower()
    content_lower = readme_content.lower()
    
    # Prefer Multi-agent FIRST to avoid matching 'ppo' inside 'ippo' etc.
    multi_agent_terms = ['marl', 'multi-agent', 'multi agent', 'ippo', 'mappo', 'self-play', 'self play']
    if any(term in name_lower or term in path_lower or term in content_lower for term in multi_agent_terms):
        return "Multi-Agent"

    # Map policy keywords to Actor-Critic (we don't expose Policy-Based)
    if re.search(r'(^|[^a-z0-9])ppo([^a-z0-9]|$)', name_lower) or 'reinforce' in name_lower or 'policy' in name_lower:
        return "Actor-Critic"
    
    # Actor-Critic
    if any(term in name_lower for term in ['a2c', 'a3c', 'sac', 'td3', 'ddpg']) or ('actor' in name_lower or 'critic' in name_lower):
        return "Actor-Critic"
        
    # Exploration
    if any(term in name_lower for term in ['rnd', 'exploration', 'curiosity']):
        return "Exploration"
        
    # Imitation learning
    if any(term in name_lower for term in ['imitation', 'behavioral', 'cloning']):
        return "Imitation Learning"
        
    # Game environments
    if any(term in name_lower for term in ['vizdoom', 'flappybird', 'game']):
        return "Game Environments"
        
    # Unity ML-Agents
    if any(term in name_lower for term in ['ml-agents', 'unity']):
        return "Unity ML-Agents"
    
    return "Other"

def collect_categories(name, path, readme_content=""):
    """Collect all applicable categories for a project (multi-category support)."""
    name_lower = name.lower()
    path_lower = path.lower()
    content_lower = readme_content.lower()

    categories = set()

    # Multi-Agent
    if any(term in name_lower or term in path_lower or term in content_lower for term in ['marl', 'multi-agent', 'multi agent', 'ippo', 'mappo', 'self-play', 'self play']):
        categories.add("Multi-Agent")

    # Map PPO/REINFORCE/policy to Actor-Critic
    if (re.search(r'(^|[^a-z0-9])ppo([^a-z0-9]|$)', name_lower) or
        re.search(r'(^|[^a-z0-9])ppo([^a-z0-9]|$)', content_lower) or
        'reinforce' in name_lower or 'policy' in name_lower or 'policy gradient' in content_lower):
        categories.add("Actor-Critic")

    # Actor-Critic
    if any(term in name_lower for term in ['a2c', 'a3c', 'sac', 'td3', 'ddpg']) or ('actor' in name_lower or 'critic' in name_lower) or 'actor-critic' in content_lower:
        categories.add("Actor-Critic")

    # Do not add Value-Based category (hidden)

    # Exploration
    if any(term in name_lower for term in ['rnd']) or any(term in content_lower for term in ['exploration', 'curiosity']):
        categories.add("Exploration")

    # Imitation Learning
    if any(term in name_lower for term in ['imitation', 'behavioral', 'cloning']) or 'dagger' in content_lower:
        categories.add("Imitation Learning")

    # If none matched, default to Other
    if not categories:
        categories.add("Other")

    return sorted(categories)

def detect_environment(name, readme_content=""):
    """Detect the environment used by the RL algorithm."""
    name_lower = name.lower()
    content_lower = readme_content.lower()
    
    # Specific environments
    if 'atari' in name_lower or 'atari' in content_lower:
        return "Atari"
    elif 'mujoco' in name_lower or 'mujoco' in content_lower:
        return "MuJoCo"
    elif 'lunar' in name_lower or 'lunarlander' in content_lower:
        return "LunarLander"
    elif 'taxi' in name_lower or 'taxi' in content_lower:
        return "Taxi"
    elif 'frozenlake' in name_lower or 'frozen' in content_lower:
        return "Frozenlake"
    elif 'flappybird' in name_lower or 'flappy' in content_lower:
        return "Flappybird"
    elif 'vizdoom' in name_lower or 'vizdoom' in content_lower:
        return "Vizdoom"
    elif 'pong' in content_lower:
        return "Atari"
    elif any(env in content_lower for env in ['gymnasium', 'gym']):
        return "Gymnasium"
    
    return "Custom Environment"

def clean_display_name(name):
    """Clean folder name to create a display name."""
    # Handle special cases
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize appropriately
    words = name.split()
    cleaned_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in ['dqn', 'ppo', 'a2c', 'sac', 'td3', 'ddpg', 'rnd', 'marl']:
            cleaned_words.append(word.upper())
        elif word_lower == 'rl':
            cleaned_words.append('RL')
        else:
            cleaned_words.append(word.capitalize())
    
    return ' '.join(cleaned_words)

def process_directory_recursive(path="", parent_name="", max_depth=10):
    """Recursively process directories and find ALL folders with README files."""
    content = fetch_github_content(path)
    if not content:
        return []
    
    implementations = []
    current_depth = path.count('/') if path else 0
    
    # Check if current directory has a README
    readme_content = fetch_readme_content(path)
    # Convert relative links/images to absolute raw URLs for site rendering
    processed_readme_content = absolutize_markdown_links(readme_content, path)
    
    # If this directory has a README and it's not the root, add it as an implementation
    if readme_content and path:  # Don't add root directory
        item_name = path.split('/')[-1]  # Get last part of path
        
        # Skip common non-algorithm directories
        if item_name.lower() not in ['.git', '__pycache__', 'node_modules', '.vscode']:
            # Create display name with full hierarchy
            if parent_name:
                display_name = f"{clean_display_name(item_name)} ({parent_name})"
            else:
                display_name = clean_display_name(item_name)
            
            # Determine categories (multi) and primary category, plus environment
            categories = collect_categories(item_name, path, readme_content)
            category = categories[0] if categories else categorize_rl_algorithm(item_name, path, readme_content)
            environment = detect_environment(item_name, readme_content)
            
            # Create implementation entry
            implementation = {
                "name": item_name,
                "path": path,
                "display_name": display_name,
                "description": f"Implementation of {item_name} reinforcement learning algorithm",
                "readme_content": processed_readme_content if processed_readme_content else "",
                "github_url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/master/{path}",
                "api_url": f"{BASE_URL}/contents/{path}",
                "download_url": None,
                "created_date": datetime.now().strftime('%Y-%m-%d'),
                "github_date": datetime.now().strftime('%Y-%m-%d'),
                "category": category,
                "categories": categories,
                "framework": "PyTorch",  # Default assumption
                "environment": environment
            }
            
            implementations.append(implementation)
            print(f"📁 Found README in: {path} -> {display_name}")
    
    # Continue recursively processing subdirectories if not at max depth
    if current_depth < max_depth:
        for item in content:
            if item['type'] == 'dir':
                item_name = item['name']
                item_path = item['path']
                
                # Skip common non-algorithm directories
                if item_name.lower() in ['.git', '__pycache__', 'node_modules', '.vscode']:
                    continue
                
                # Create parent name for nested items
                if path:
                    new_parent_name = f"{parent_name} {clean_display_name(item_name.split('/')[-1])}" if parent_name else clean_display_name(item_name)
                else:
                    new_parent_name = clean_display_name(item_name)
                
                # Recursively process this directory
                sub_implementations = process_directory_recursive(item_path, new_parent_name, max_depth)
                implementations.extend(sub_implementations)
    
    return implementations

def process_directory(path="", parent_name=""):
    """Wrapper function for backward compatibility."""
    return process_directory_recursive(path, parent_name, max_depth=10)

def generate_rl_json():
    """Generate the RL JSON data file."""
    print("🚀 Starting RL data generation...")
    
    # Fetch all RL implementations
    implementations = process_directory()
    
    if not implementations:
        print("❌ No RL implementations found!")
        return False
    
    # Create the JSON structure
    rl_data = {
        "rl_implementations": implementations
    }
    
    # Write to JSON file
    data_path = BASE_PATH / DATA_FILE
    data_path.parent.mkdir(exist_ok=True)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {DATA_FILE} with {len(implementations)} RL implementations")
    return True

def generate_markdown_files():
    """Generate individual markdown files for each RL implementation."""
    # Read the JSON data
    data_path = BASE_PATH / DATA_FILE
    if not data_path.exists():
        print("❌ RL JSON data not found. Run JSON generation first.")
        return False
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    implementations = data.get('rl_implementations', [])
    if not implementations:
        print("❌ No RL implementations found in JSON data.")
        return False
    
    # Create output directory
    models_path = BASE_PATH / MODELS_DIR
    models_path.mkdir(exist_ok=True)
    
    # Clear existing files
    for existing_file in models_path.glob('*.md'):
        existing_file.unlink()
        print(f"🗑️  Removed old file: {existing_file.name}")
    
    # Filter out obvious non-algorithm directories but keep all with README content
    main_implementations = []
    for impl in implementations:
        name_lower = impl['name'].lower()
        path_lower = impl['path'].lower()
        
        # Skip obvious non-algorithm folders
        skip_patterns = ['images', 'assets', 'docs', '__pycache__', 'node_modules', '.git']
        
        # Skip if it's clearly a non-algorithm folder
        if any(skip in name_lower for skip in skip_patterns):
            continue
            
        # Skip if path contains obvious non-algorithm patterns
        if any(skip in path_lower for skip in skip_patterns):
            continue
            
        # Include if it has substantial README content (likely an algorithm)
        if impl['readme_content'] and len(impl['readme_content'].strip()) > 50:
            main_implementations.append(impl)
        # Also include top-level directories even without README
        elif impl['path'].count('/') == 0:
            main_implementations.append(impl)
    
    # Generate markdown files
    successful_count = 0
    for index, impl in enumerate(main_implementations, 1):
        try:
            name = impl['display_name']
            slug = re.sub(r'[^\w\s-]', '', name.lower())
            slug = re.sub(r'[-\s]+', '-', slug).strip('-')
            
            filename = f"{index:02d}-{slug}.md"
            filepath = models_path / filename
            
            # Generate markdown content
            content = f"""---
title: "{name}"
excerpt: "{impl['description']}"
collection: rl
layout: rl-implementation
category: "{impl['category']}"
categories: [{', '.join([f'\"{c}\"' for c in impl.get('categories', [impl['category']])])}]
framework: "{impl['framework']}"
environment: "{impl['environment']}"
github_url: "{impl['github_url']}"
date: {impl['github_date']}
---

## Overview
{impl['description']}

## Technical Details
- **Framework**: {impl['framework']}
- **Environment**: {impl['environment']}
- **Category**: {impl['category']}

## Implementation Details

{impl['readme_content'] if impl['readme_content'] else f'''
# {name}

This implementation demonstrates {impl['category'].lower()} using {impl['framework']} framework on {impl['environment']} environment.

## Features
- Clean and well-documented code
- Easy to understand implementation
- Comprehensive training and evaluation scripts

## Usage
Please refer to the GitHub repository for detailed usage instructions and training procedures.
'''}

## Source Code
📁 **GitHub Repository**: [{name}]({impl['github_url']})

View the complete implementation, training scripts, and documentation on GitHub.
"""
            
            # Write the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Generated: {filename}")
            successful_count += 1
            
        except Exception as e:
            print(f"❌ Error generating file for {impl['name']}: {e}")
    
    print(f"\n🎉 Successfully generated {successful_count} RL markdown files")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Refresh RL implementations data')
    parser.add_argument('--json-only', action='store_true', help='Only generate JSON data')
    parser.add_argument('--md-only', action='store_true', help='Only generate markdown files')
    args = parser.parse_args()
    
    if args.md_only:
        generate_markdown_files()
    elif args.json_only:
        generate_rl_json()
    else:
        # Generate both
        if generate_rl_json():
            generate_markdown_files()

if __name__ == "__main__":
    main()
