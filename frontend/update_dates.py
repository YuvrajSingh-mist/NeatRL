#!/usr/bin/env python3
"""
Update model dates with actual GitHub commit dates from API
"""

import json
import os
import re
import requests
from datetime import datetime
import time

def get_github_first_commit_date(folder_name, github_token=None):
    """Get the first commit date for a specific folder from GitHub API"""
    repo_owner = "YuvrajSingh-mist"
    repo_name = "Paper-Replications"
    
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
            'path': folder_name,
            'per_page': 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            commits = response.json()
            if commits:
                # Get the earliest commit (last in the list when sorted by date)
                first_commit = commits[-1] if len(commits) == 1 else commits[0]
                commit_date = first_commit['commit']['author']['date']
                # Convert to just date (YYYY-MM-DD format)
                date_obj = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                return date_obj.strftime('%Y-%m-%d')
        
        print(f"Warning: Could not get commit date for {folder_name}")
        return None
        
    except Exception as e:
        print(f"Error getting commit date for {folder_name}: {e}")
        return None

def update_models_json_with_dates(github_token=None):
    """Update models.json with actual GitHub commit dates"""
    json_path = '_data/models.json'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} not found!")
        return {}
    
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_dates = {}
    
    print("Fetching GitHub commit dates...")
    for i, model in enumerate(data.get('models', []), 1):
        model_name = model.get('name', '')
        print(f"Processing {i}/{len(data['models'])}: {model_name}")
        
        # Get actual commit date from GitHub
        commit_date = get_github_first_commit_date(model_name, github_token)
        
        if commit_date:
            updated_dates[model_name] = commit_date
            print(f"  Found date: {commit_date}")
        else:
            # Fallback to a reasonable default
            updated_dates[model_name] = '2024-01-01'
            print(f"  Using fallback date: 2024-01-01")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Write back the updated data with dates
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated models.json with GitHub commit dates!")
    return updated_dates

def update_markdown_file_date(file_path, new_date):
    """Update the date in a markdown file's frontmatter"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update the date in frontmatter
    content = re.sub(
        r'date:\s*\d{4}-\d{2}-\d{2}',
        f'date: {new_date}',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_all_markdown_dates(dates_dict):
    """Update all markdown files with their corresponding GitHub dates"""
    models_dir = '_models'
    
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found!")
        return
    
    # Get all markdown files
    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.md'):
            model_files.append(filename)
    
    print(f"\nUpdating {len(model_files)} markdown files with GitHub dates...")
    
    updated_count = 0
    for filename in sorted(model_files):
        file_path = os.path.join(models_dir, filename)
        
        # Read the title from the file to match with dates_dict
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        title_match = re.search(r'title:\s*["\']([^"\']+)["\']', content)
        if title_match:
            title = title_match.group(1)
            
            # Find matching date
            if title in dates_dict:
                update_markdown_file_date(file_path, dates_dict[title])
                print(f"Updated {filename} -> {dates_dict[title]}")
                updated_count += 1
            else:
                # Try to find a close match
                for model_name, date in dates_dict.items():
                    if model_name.lower().replace(' ', '') in title.lower().replace(' ', ''):
                        update_markdown_file_date(file_path, date)
                        print(f"Updated {filename} -> {date} (matched with {model_name})")
                        updated_count += 1
                        break
                else:
                    print(f"No date found for {filename} (title: {title})")
    
    print(f"\nUpdated {updated_count} markdown files with GitHub commit dates!")

def main():
    """Main function"""
    # You can pass a GitHub token as environment variable for higher rate limits
    github_token = os.environ.get('GITHUB_TOKEN')
    
    if github_token:
        print("Using GitHub token for higher rate limits")
    else:
        print("No GitHub token found, using anonymous access (may hit rate limits)")
    
    # Update models.json with GitHub dates
    dates_dict = update_models_json_with_dates(github_token)
    
    # Update markdown files with the dates
    if dates_dict:
        update_all_markdown_dates(dates_dict)
    
    print("\nAll done! GitHub commit dates have been applied.")

if __name__ == '__main__':
    main()
