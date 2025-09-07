#!/usr/bin/env python3
"""
Auto-detect and create model files for new implementations
This script monitors the Paper-Replications repository and automatically
creates markdown files for any new model implementations found.
"""

import json
import requests
import os
import sys
import argparse
from datetime import datetime
import subprocess

def get_github_folders(owner, repo, token=None):
    """Get all folders from the GitHub repository"""
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        folders = []
        for item in response.json():
            if item['type'] == 'dir' and not item['name'].startswith('.'):
                folders.append(item['name'])
        
        return sorted(folders)
    except Exception as e:
        print(f"Error fetching repository contents: {e}")
        return []

def get_existing_models():
    """Get list of models that already have markdown files"""
    models_file = "_data/models.json"
    if not os.path.exists(models_file):
        return []
    
    with open(models_file, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (array) and new format (object with models key)
    if isinstance(data, list):
        models_data = data
    elif isinstance(data, dict) and 'models' in data:
        models_data = data['models']
    else:
        print(f"Warning: Unexpected format in {models_file}")
        return []
    
    # Extract folder names, trying different possible field names
    folder_names = []
    for model in models_data:
        if isinstance(model, dict):
            # Try different possible field names for the folder/directory name
            folder_name = (
                model.get('folder_name') or 
                model.get('name') or 
                model.get('display_name') or 
                ''
            )
            if folder_name:
                folder_names.append(folder_name)
    
    return folder_names

def detect_new_models(owner="YuvrajSingh-mist", repo="Paper-Replications", token=None):
    """Detect new models that don't have markdown files yet"""
    print(f"Checking for new models in {owner}/{repo}...")
    
    # Get all folders from repository
    github_folders = get_github_folders(owner, repo, token)
    print(f"Found {len(github_folders)} folders in repository: {github_folders}")
    
    # Get existing models
    existing_models = get_existing_models()
    print(f"Found {len(existing_models)} existing models: {existing_models}")
    
    # Find new models
    new_models = [folder for folder in github_folders if folder not in existing_models]
    
    if new_models:
        print(f"🆕 Found {len(new_models)} new models: {new_models}")
        return new_models
    else:
        print("✅ No new models found. All models are up to date.")
        return []

def create_model_entry(folder_name, owner="YuvrajSingh-mist", repo="Paper-Replications", token=None):
    """Create a new model entry for the models.json file"""
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Get README content
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_name}/README.md"
    readme_content = "Implementation from scratch"
    
    try:
        response = requests.get(readme_url, headers=headers)
        if response.status_code == 200:
            import base64
            readme_data = response.json()
            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
    except Exception as e:
        print(f"Could not fetch README for {folder_name}: {e}")
    
    # Get commit date
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={folder_name}&per_page=1"
    commit_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        response = requests.get(commits_url, headers=headers)
        if response.status_code == 200:
            commits = response.json()
            if commits:
                commit_date = commits[0]['commit']['author']['date'][:10]  # YYYY-MM-DD
    except Exception as e:
        print(f"Could not fetch commit date for {folder_name}: {e}")
    
    # Create model entry
    model_entry = {
        "name": folder_name.replace('-', ' ').replace('_', ' ').title(),
        "description": f"Implementation of {folder_name.replace('-', ' ').replace('_', ' ')} from scratch",
        "github_url": f"https://github.com/{owner}/{repo}/tree/master/{folder_name}",
        "folder_name": folder_name,
        "readme_content": readme_content,
        "date": commit_date
    }
    
    return model_entry

def add_new_models_to_json(new_models, token=None):
    """Add new models to the models.json file"""
    models_file = "_data/models.json"
    
    # Load existing models
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (array) and new format (object with models key)
        if isinstance(data, list):
            models_data = data
        elif isinstance(data, dict) and 'models' in data:
            models_data = data['models']
        else:
            print(f"Warning: Unexpected format in {models_file}, creating new structure")
            models_data = []
    else:
        models_data = []
    
    # Add new models
    for folder_name in new_models:
        print(f"📝 Creating entry for {folder_name}...")
        model_entry = create_model_entry(folder_name, token=token)
        models_data.append(model_entry)
    
    # Save updated models.json - maintain the original structure
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            original_data = json.load(f)
        
        if isinstance(original_data, dict) and 'models' in original_data:
            # Maintain the object structure
            original_data['models'] = models_data
            final_data = original_data
        else:
            # Use array structure
            final_data = models_data
    else:
        # Default to array structure for new files
        final_data = models_data
    
    with open(models_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"✅ Added {len(new_models)} new models to {models_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Auto-detect new models from Paper-Replications repository')
    parser.add_argument('--source', default='paper-replications', 
                       help='Source repository type (default: paper-replications)')
    args = parser.parse_args()
    
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Warning: No GITHUB_TOKEN found. API requests may be rate limited.")
    
    print(f"🔧 Running auto-detection for source: {args.source}")
    
    # Detect new models
    new_models = detect_new_models(token=token)
    
    if new_models:
        # Add to models.json
        add_new_models_to_json(new_models, token=token)
        
        # Regenerate all markdown files with latest guidelines
        print("🔄 Regenerating all model markdown files with latest guidelines...")
        print("  ✅ Using model-implementation layout")
        print("  ✅ Converting images to GitHub hyperlinks")
        print("  ✅ Using exact GitHub API data")
        try:
            subprocess.run(['python', 'regenerate_models.py'], check=True)
            print("  ✅ Model files regenerated successfully!")
            
            # Ensure images are converted to links
            print("🖼️ Converting any images to GitHub hyperlinks...")
            subprocess.run(['python', 'fix_images_direct.py'], check=True)
            print("  ✅ Image conversion completed!")
            
            print("✅ Successfully regenerated all model files!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error regenerating model files: {e}")
            return False
        
        print(f"🎉 Successfully processed {len(new_models)} new models!")
        return True
    else:
        # No new models found is a successful outcome, not an error
        print("✅ Repository is up to date. No action needed.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
