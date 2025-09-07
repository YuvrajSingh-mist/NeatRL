#!/usr/bin/env python3
"""
Fetch GitHub creation/commit dates for models and datasets
This script fetches the actual creation dates from GitHub API and updates
the JSON files with accurate date information.
"""

import json
import os
import argparse
import requests
from datetime import datetime
import time

def get_github_commit_date(repo_owner, repo_name, folder_path, github_token=None):
    """Get the first commit date for a specific folder from GitHub API"""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'SmolHub-Website'
    }
    
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    try:
        # Get commits for the specific folder
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        params = {
            'path': folder_path,
            'per_page': 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            commits = response.json()
            if commits:
                # Get the first commit date
                first_commit = commits[0]
                commit_date = first_commit['commit']['author']['date']
                # Convert to just date (YYYY-MM-DD format)
                date_obj = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                return date_obj.strftime('%Y-%m-%d')
        elif response.status_code == 409:
            # Repository is empty
            print(f"Warning: Repository {repo_owner}/{repo_name} is empty for path {folder_path}")
            return None
        else:
            print(f"Warning: Could not get commit date for {folder_path} (status: {response.status_code})")
            return None
        
    except Exception as e:
        print(f"Error getting commit date for {folder_path}: {e}")
        return None

def update_models_with_github_dates(github_token=None):
    """Update models.json with GitHub commit dates"""
    json_path = '_data/models.json'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} not found!")
        return False
    
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🔄 Fetching GitHub commit dates for models...")
    updated_count = 0
    
    for i, model in enumerate(data.get('models', []), 1):
        model_name = model.get('name', '')
        print(f"Processing {i}/{len(data['models'])}: {model_name}")
        
        # Get actual commit date from GitHub
        commit_date = get_github_commit_date(
            'YuvrajSingh-mist', 
            'Paper-Replications', 
            model_name, 
            github_token
        )
        
        if commit_date:
            model['created_date'] = commit_date
            model['github_date'] = commit_date
            print(f"  ✅ Found date: {commit_date}")
            updated_count += 1
        else:
            # Keep existing date or use fallback
            if 'created_date' not in model:
                model['created_date'] = '2024-01-01'
                model['github_date'] = '2024-01-01'
            print(f"  ⚠️  Using fallback date: {model.get('created_date', '2024-01-01')}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    # Add last_updated timestamp
    data['last_updated'] = datetime.now().isoformat()
    
    # Write back the updated data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated models.json with GitHub commit dates! ({updated_count} models updated)")
    return True

def update_datasets_with_github_dates(github_token=None):
    """Update datasets.json with GitHub commit dates"""
    json_path = '_data/datasets.json'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} not found!")
        return False
    
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🔄 Fetching GitHub commit dates for datasets...")
    updated_count = 0
    
    for i, dataset in enumerate(data.get('datasets', []), 1):
        dataset_name = dataset.get('name', '')
        print(f"Processing {i}/{len(data['datasets'])}: {dataset_name}")
        
        # Get actual commit date from GitHub
        commit_date = get_github_commit_date(
            'YuvrajSingh-mist', 
            'Datasets-Collection', 
            dataset_name, 
            github_token
        )
        
        if commit_date:
            dataset['created_date'] = commit_date
            dataset['github_date'] = commit_date
            print(f"  ✅ Found date: {commit_date}")
            updated_count += 1
        else:
            # Keep existing date or use fallback
            if not dataset.get('created_date'):
                dataset['created_date'] = '2024-01-01'
                dataset['github_date'] = '2024-01-01'
            print(f"  ⚠️  Using fallback date: {dataset.get('created_date', '2024-01-01')}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    # Add last_updated timestamp
    data['last_updated'] = datetime.now().isoformat()
    
    # Write back the updated data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated datasets.json with GitHub commit dates! ({updated_count} datasets updated)")
    return True

def update_model_markdown_files():
    """Update model markdown files to use github_date from JSON"""
    models_dir = '_models'
    json_path = '_data/models.json'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} not found!")
        return False
    
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found!")
        return False
    
    # Load the JSON data to get github dates
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping of model names to github dates
    name_to_date = {}
    for model in data.get('models', []):
        name = model.get('name', '')
        github_date = model.get('github_date', model.get('created_date', '2024-01-01'))
        name_to_date[name] = github_date
    
    print("🔄 Updating model markdown files with GitHub dates...")
    updated_count = 0
    
    # Get all markdown files
    for filename in os.listdir(models_dir):
        if filename.endswith('.md'):
            file_path = os.path.join(models_dir, filename)
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from frontmatter
            import re
            title_match = re.search(r'title:\s*["\']([^"\']+)["\']', content)
            if title_match:
                title = title_match.group(1)
                
                # Find matching date in our mapping
                github_date = None
                for model_name, date in name_to_date.items():
                    if model_name.lower().replace(' ', '').replace('-', '') in title.lower().replace(' ', '').replace('-', ''):
                        github_date = date
                        break
                
                if github_date:
                    # Update the date in frontmatter
                    content = re.sub(
                        r'date:\s*\d{4}-\d{2}-\d{2}',
                        f'date: {github_date}',
                        content
                    )
                    
                    # Write back to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  ✅ Updated {filename} -> {github_date}")
                    updated_count += 1
                else:
                    print(f"  ⚠️  No GitHub date found for {filename} (title: {title})")
    
    print(f"\n✅ Updated {updated_count} markdown files with GitHub dates!")
    return True

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch GitHub creation dates for models and datasets')
    parser.add_argument('--source', default='paper-replications', 
                       help='Source repository type (default: paper-replications)')
    parser.add_argument('--only', choices=['models', 'datasets', 'both'], default='both',
                       help='Limit updates to only models, only datasets, or both (default)')
    args = parser.parse_args()
    
    # You can pass a GitHub token as environment variable for higher rate limits
    github_token = os.environ.get('GITHUB_TOKEN')
    
    if github_token:
        print("🔑 Using GitHub token for higher rate limits")
    else:
        print("⚠️  No GitHub token found, using anonymous access (may hit rate limits)")
        print("💡 Set GITHUB_TOKEN environment variable for better performance")
    
    print("=" * 60)
    print("🚀 Fetching GitHub Creation Dates")
    print(f"🔧 Running for source: {args.source}")
    print("=" * 60)
    
    models_success = False
    datasets_success = False

    # Update models.json with GitHub dates (if requested)
    if args.only in ('models', 'both'):
        print("\n📁 UPDATING MODELS...")
        models_success = update_models_with_github_dates(github_token)

    # Update datasets.json with GitHub dates (if requested)
    if args.only in ('datasets', 'both'):
        print("\n📊 UPDATING DATASETS...")
        datasets_success = update_datasets_with_github_dates(github_token)

    # Update model markdown files only if models were updated successfully
    if models_success:
        print("\n📝 UPDATING MODEL MARKDOWN FILES...")
        update_model_markdown_files()
    
    print("\n" + "=" * 60)
    print("🎉 GitHub dates update completed!")
    print("=" * 60)
    
    if models_success or datasets_success:
        print("✅ JSON files updated with actual GitHub creation dates")
        print("✅ Model pages will now show authentic creation dates")
        print("✅ Dataset pages will now show authentic creation dates")
    else:
        print("❌ No updates were made - check your GitHub token and network connection")

if __name__ == '__main__':
    main()
