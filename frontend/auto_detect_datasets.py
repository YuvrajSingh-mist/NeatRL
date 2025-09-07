#!/usr/bin/env python3
"""
Auto-detect and create dataset files for new dataset implementations
This script monitors the Datasets-Collection repository and automatically
creates markdown files for any new dataset implementations found.
"""

import json
import requests
import os
import sys
from datetime import datetime
import subprocess

def get_github_folders(owner, repo, token=None):
    """Get all folders from the GitHub repository"""
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            contents = response.json()
            folders = [item['name'] for item in contents if item['type'] == 'dir' and not item['name'].startswith('.')]
            return folders
        else:
            print(f"Error fetching repository contents: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_existing_datasets():
    """Get list of existing datasets from datasets.json"""
    datasets_file = "_data/datasets.json"
    if not os.path.exists(datasets_file):
        return []
    
    try:
        with open(datasets_file, 'r') as f:
            data = json.load(f)
            return [dataset.get('name', '') for dataset in data.get('datasets', [])]
    except Exception as e:
        print(f"Error reading datasets.json: {e}")
        return []

def detect_new_datasets(owner="YuvrajSingh-mist", repo="Datasets-Collection", token=None):
    """Detect new datasets in the repository"""
    print(f"Checking for new datasets in {owner}/{repo}...")
    
    # Get all folders from repository
    github_folders = get_github_folders(owner, repo, token)
    print(f"Found {len(github_folders)} folders in repository: {github_folders}")
    
    # Get existing datasets
    existing_datasets = get_existing_datasets()
    print(f"Found {len(existing_datasets)} existing datasets: {existing_datasets}")
    
    # Find new datasets
    new_datasets = [folder for folder in github_folders if folder not in existing_datasets]
    
    if new_datasets:
        print(f"🆕 Found {len(new_datasets)} new datasets: {new_datasets}")
        return new_datasets
    else:
        print("✅ No new datasets found. All datasets are up to date.")
        return []

def get_dataset_info(folder_name, owner="YuvrajSingh-mist", repo="Datasets-Collection", token=None):
    """Get dataset information from README and folder contents"""
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Get README content
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_name}/README.md"
    readme_content = f"Dataset implementation: {folder_name}"
    description = f"A comprehensive dataset for {folder_name.replace('-', ' ').replace('_', ' ').lower()}"
    
    try:
        response = requests.get(readme_url, headers=headers)
        if response.status_code == 200:
            import base64
            readme_data = response.json()
            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
            
            # Extract description from README (first paragraph after title)
            lines = readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and not line.startswith('|') and len(line.strip()) > 50:
                    description = line.strip()
                    break
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
    
    # Try to detect dataset format and size from folder contents
    contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_name}"
    dataset_format = "Unknown"
    size = "Unknown"
    samples = "Unknown"
    
    try:
        response = requests.get(contents_url, headers=headers)
        if response.status_code == 200:
            contents = response.json()
            files = [item['name'] for item in contents if item['type'] == 'file']
            
            # Detect format based on file extensions
            if any(f.endswith('.csv') for f in files):
                dataset_format = "CSV"
            elif any(f.endswith('.json') for f in files):
                dataset_format = "JSON"
            elif any(f.endswith('.parquet') for f in files):
                dataset_format = "Parquet"
            elif any(f.endswith(('.txt', '.text')) for f in files):
                dataset_format = "Text"
            
            # Try to get size info from file sizes
            total_size = sum(item.get('size', 0) for item in contents if item['type'] == 'file')
            if total_size > 0:
                if total_size > 1024*1024*1024:  # > 1GB
                    size = f"{total_size/(1024*1024*1024):.1f} GB"
                elif total_size > 1024*1024:  # > 1MB
                    size = f"{total_size/(1024*1024):.1f} MB"
                else:
                    size = f"{total_size/1024:.1f} KB"
    except Exception as e:
        print(f"Could not fetch contents for {folder_name}: {e}")
    
    return {
        "readme_content": readme_content,
        "description": description,
        "commit_date": commit_date,
        "format": dataset_format,
        "size": size,
        "samples": samples
    }

def create_dataset_entry(folder_name, owner="YuvrajSingh-mist", repo="Datasets-Collection", token=None):
    """Create a new dataset entry for the datasets.json file"""
    print(f"📝 Creating entry for {folder_name}...")
    
    # Get dataset information
    info = get_dataset_info(folder_name, owner, repo, token)
    
    # Infer tags and tasks from folder name and description
    name_lower = folder_name.lower()
    tags = []
    tasks = []
    
    # Common tag patterns
    if 'qna' in name_lower or 'question' in name_lower:
        tags.extend(['QnA', 'Question-Answering'])
        tasks.extend(['NLP', 'Question Answering'])
    if 'image' in name_lower or 'vision' in name_lower:
        tags.extend(['Computer Vision', 'Images'])
        tasks.extend(['Computer Vision', 'Image Classification'])
    if 'text' in name_lower or 'nlp' in name_lower:
        tags.extend(['Text', 'NLP'])
        tasks.extend(['NLP', 'Text Classification'])
    if 'disease' in name_lower:
        tags.extend(['Medical', 'Disease', 'Healthcare'])
        tasks.extend(['Classification', 'Detection'])
    if 'plant' in name_lower or 'agriculture' in name_lower:
        tags.extend(['Agriculture', 'Plant'])
        tasks.extend(['Classification', 'Detection'])
    
    # Default tags if none found
    if not tags:
        tags = [folder_name.replace('-', ' ').replace('_', ' ').title()]
    if not tasks:
        tasks = ['Machine Learning', 'Data Analysis']
    
    # Create dataset entry
    dataset_entry = {
        "name": folder_name,
        "display_name": f"{folder_name.replace('-', ' ').replace('_', ' ').title()} Dataset",
        "description": info["description"],
        "size": info["size"],
        "format": info["format"],
        "samples": info["samples"],
        "license": "Open Source",
        "tasks": tasks,
        "tags": tags,
        "github_url": f"https://github.com/{owner}/{repo}/tree/main/{folder_name}",
        "download_url": None,
        "paper_url": None,
        "created_date": info["commit_date"],
        "last_updated": datetime.now().isoformat(),
        "readme_content": info["readme_content"]
    }
    
    return dataset_entry

def add_new_datasets_to_json(new_datasets, token=None):
    """Add new datasets to the datasets.json file"""
    datasets_file = "_data/datasets.json"
    
    # Load existing datasets
    if os.path.exists(datasets_file):
        with open(datasets_file, 'r') as f:
            datasets_data = json.load(f)
    else:
        datasets_data = {
            "last_updated": "",
            "total_datasets": 0,
            "repository": "YuvrajSingh-mist/Datasets-Collection",
            "datasets": []
        }
    
    # Add new datasets
    for folder_name in new_datasets:
        dataset_entry = create_dataset_entry(folder_name, token=token)
        datasets_data["datasets"].append(dataset_entry)
    
    # Update metadata
    datasets_data["total_datasets"] = len(datasets_data["datasets"])
    datasets_data["last_updated"] = datetime.now().isoformat()
    
    # Save updated datasets.json
    with open(datasets_file, 'w') as f:
        json.dump(datasets_data, f, indent=2)
    
    print(f"✅ Added {len(new_datasets)} new datasets to {datasets_file}")

def main():
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Warning: No GITHUB_TOKEN found. API requests may be rate limited.")
    
    # Detect new datasets
    new_datasets = detect_new_datasets(token=token)
    
    if new_datasets:
        # Add to datasets.json
        add_new_datasets_to_json(new_datasets, token=token)
        
        # Generate markdown files
        print("🔄 Generating dataset markdown files...")
        try:
            subprocess.run(['python', 'generate_dataset_files.py'], check=True)
            print("  ✅ Dataset markdown files generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating dataset files: {e}")
            return False
        except FileNotFoundError:
            print("⚠️ generate_dataset_files.py not found, skipping markdown generation")
        
        print(f"🎉 Successfully processed {len(new_datasets)} new datasets!")
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
