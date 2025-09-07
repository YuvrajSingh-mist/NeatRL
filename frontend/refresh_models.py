#!/usr/bin/env python3
"""
Script to refresh models from the Paper-Replications GitHub repository.
This script clones the repo, reads folder structure and README files,
then generates Jekyll model files.
"""

import os
import shutil
import subprocess
import glob
import re
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/YuvrajSingh-mist/Paper-Replications.git"
TEMP_DIR = "_temp_paper_replications"
MODELS_DIR = "_models"
BASE_PATH = Path(__file__).parent

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        return None

def clean_temp_dir():
    """Remove temporary directory if it exists."""
    temp_path = BASE_PATH / TEMP_DIR
    if temp_path.exists():
        shutil.rmtree(temp_path)

def clone_repository():
    """Clone the Paper-Replications repository."""
    print("Cloning repository...")
    clean_temp_dir()
    
    command = f"git clone {REPO_URL} {TEMP_DIR}"
    result = run_command(command, cwd=BASE_PATH)
    
    if result is None:
        print("Failed to clone repository")
        return False
    
    print("Repository cloned successfully")
    return True

def read_readme_content(readme_path):
    """Read and clean README content for use as excerpt."""
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown headers and clean up
        content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Remove markdown links
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Remove bold formatting
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Remove italic formatting
        content = re.sub(r'`([^`]+)`', r'\1', content)  # Remove code formatting
        content = re.sub(r'\n+', ' ', content)  # Replace newlines with spaces
        content = content.strip()
        
        # Truncate to reasonable length for excerpt
        if len(content) > 200:
            content = content[:200] + "..."
        
        return content if content else "AI model for advanced machine learning applications."
    
    except Exception as e:
        print(f"Error reading README: {e}")
        return "AI model for advanced machine learning applications."

def get_model_folders():
    """Get list of model folders from the cloned repository."""
    temp_path = BASE_PATH / TEMP_DIR
    
    if not temp_path.exists():
        print("Temporary directory not found")
        return []
    
    # Get all directories (excluding .git and other hidden folders)
    folders = []
    for item in temp_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            folders.append(item.name)
    
    return sorted(folders)

def clean_title(folder_name):
    """Clean folder name to create a nice title."""
    # Replace hyphens and underscores with spaces
    title = folder_name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize each word
    title = ' '.join(word.capitalize() for word in title.split())
    
    return title

def create_model_file(folder_name, index):
    """Create a Jekyll model file from folder information."""
    temp_path = BASE_PATH / TEMP_DIR / folder_name
    models_path = BASE_PATH / MODELS_DIR
    
    # Ensure models directory exists
    models_path.mkdir(exist_ok=True)
    
    # Look for README files
    readme_files = list(temp_path.glob("README.md")) + list(temp_path.glob("readme.md")) + list(temp_path.glob("README.txt"))
    
    title = clean_title(folder_name)
    
    if readme_files:
        description = read_readme_content(readme_files[0])
        # Read full README content for the detail page
        with open(readme_files[0], 'r', encoding='utf-8') as f:
            full_content = f.read()
    else:
        description = f"Advanced AI model implementation for {title.lower()}."
        full_content = f"""## {title}

This is an advanced AI model implementation for {title.lower()}.

### Features
- State-of-the-art architecture
- High performance and accuracy
- Well-documented codebase
- Easy to use and extend

### Usage
Please refer to the original repository for detailed usage instructions and implementation details.

### Repository
This model is part of the Paper-Replications project available on GitHub.
"""
    
    # Create filename
    filename = f"model-{index+1:02d}-{folder_name.lower().replace('_', '-').replace(' ', '-')}.md"
    filepath = models_path / filename
    
    # Create the markdown content
    markdown_content = f"""---
title: "{title}"
excerpt: "{description}<br/><img src='/images/500x300.png'>"
collection: models
repository_folder: "{folder_name}"
---

{full_content}
"""
    
    # Write the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Created model file: {filename}")

def clear_existing_models():
    """Remove existing model files."""
    models_path = BASE_PATH / MODELS_DIR
    
    if models_path.exists():
        for model_file in models_path.glob("model-*.md"):
            model_file.unlink()
            print(f"Removed existing model: {model_file.name}")

def refresh_models():
    """Main function to refresh models from repository."""
    print("Starting model refresh process...")
    
    # Step 1: Clone repository
    if not clone_repository():
        return False
    
    # Step 2: Get model folders
    folders = get_model_folders()
    print(f"Found {len(folders)} model folders: {folders}")
    
    if not folders:
        print("No model folders found in repository")
        clean_temp_dir()
        return False
    
    # Step 3: Clear existing models
    clear_existing_models()
    
    # Step 4: Create new model files
    for index, folder_name in enumerate(folders):
        create_model_file(folder_name, index)
    
    # Step 5: Clean up
    clean_temp_dir()
    
    print(f"Successfully created {len(folders)} model files")
    print("Model refresh completed!")
    return True

if __name__ == "__main__":
    refresh_models()
