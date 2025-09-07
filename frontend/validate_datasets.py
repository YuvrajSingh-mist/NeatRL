#!/usr/bin/env python3
"""
Validate dataset markdown files and ensure they follow guidelines
This script checks that all dataset files follow the established standards.
"""

import os
import json
import yaml
from pathlib import Path
import re

def validate_frontmatter(filepath, content):
    """Validate YAML frontmatter"""
    issues = []
    
    try:
        # Extract frontmatter
        if not content.startswith('---'):
            issues.append("Missing frontmatter delimiter")
            return issues
        
        end_idx = content.find('---', 3)
        if end_idx == -1:
            issues.append("Missing closing frontmatter delimiter")
            return issues
        
        frontmatter_str = content[3:end_idx].strip()
        frontmatter = yaml.safe_load(frontmatter_str)
        
        # Required fields
        required_fields = ['title', 'excerpt', 'collection', 'github_url']
        for field in required_fields:
            if field not in frontmatter:
                issues.append(f"Missing required field: {field}")
        
        # Collection should be 'datasets'
        if frontmatter.get('collection') != 'datasets':
            issues.append(f"Collection should be 'datasets', found: {frontmatter.get('collection')}")
        
        # GitHub URL validation
        github_url = frontmatter.get('github_url', '')
        if github_url and 'github.com' not in github_url:
            issues.append("Invalid GitHub URL")
        
        # Tags and tasks should be lists
        for field in ['tags', 'tasks']:
            if field in frontmatter:
                if not isinstance(frontmatter[field], list):
                    issues.append(f"{field} should be a list")
        
    except yaml.YAMLError as e:
        issues.append(f"Invalid YAML frontmatter: {e}")
    except Exception as e:
        issues.append(f"Error parsing frontmatter: {e}")
    
    return issues

def validate_content_structure(filepath, content):
    """Validate content structure"""
    issues = []
    
    # Required sections
    required_sections = [
        "## Dataset Overview",
        "## Technical Details", 
        "## Applications",
        "## Access and Usage"
    ]
    
    for section in required_sections:
        if section not in content:
            issues.append(f"Missing required section: {section}")
    
    # Check for local image references
    local_image_patterns = [
        r'!\[[^\]]*\]\((?!http)[^)]+\.(?:png|jpg|jpeg|gif|svg)\)',
        r'<img[^>]+src=["\'](?!http)[^"\']+\.(?:png|jpg|jpeg|gif|svg)["\']'
    ]

    for pattern in local_image_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append(f"Found local image references that should use GitHub URLs: {matches}")

    # Check for incorrect GitHub image links
    incorrect_github_pattern = r'!\[([^\]]+)\]\((https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^)]+\.(jpg|jpeg|png|gif|svg))\)'
    matches = re.findall(incorrect_github_pattern, content, re.IGNORECASE)
    for match in matches:
        if not match[1].startswith("https://raw.githubusercontent.com/YuvrajSingh-mist/"):
            issues.append(f"Incorrect GitHub image link: {match[1]}")
    
    return issues

def validate_json_consistency():
    """Validate that markdown files match datasets.json entries"""
    issues = []
    
    # Load datasets.json
    datasets_file = "_data/datasets.json"
    if not os.path.exists(datasets_file):
        issues.append("datasets.json file not found")
        return issues
    
    try:
        with open(datasets_file, 'r') as f:
            datasets_data = json.load(f)
        
        datasets = datasets_data.get('datasets', [])
        dataset_names = [d.get('name', '') for d in datasets]
        
        # Check markdown files exist for all datasets
        datasets_dir = Path("_datasets")
        if not datasets_dir.exists():
            issues.append("_datasets directory not found")
            return issues
        
        markdown_files = list(datasets_dir.glob("*.md"))
        
        # Check count consistency
        if len(markdown_files) != len(datasets):
            issues.append(f"Mismatch: {len(datasets)} datasets in JSON, {len(markdown_files)} markdown files")
        
        return issues
        
    except Exception as e:
        issues.append(f"Error validating JSON consistency: {e}")
        return issues

def validate_single_file(filepath):
    """Validate a single dataset markdown file"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Validate frontmatter
        frontmatter_issues = validate_frontmatter(filepath, content)
        issues.extend(frontmatter_issues)
        
        # Validate content structure
        content_issues = validate_content_structure(filepath, content)
        issues.extend(content_issues)
        
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return issues

def main():
    """Main validation function"""
    print("🔍 Validating dataset files...")
    
    total_issues = 0
    
    # Validate JSON consistency
    print("\n📊 Checking JSON consistency...")
    json_issues = validate_json_consistency()
    if json_issues:
        print("❌ JSON consistency issues:")
        for issue in json_issues:
            print(f"  - {issue}")
        total_issues += len(json_issues)
    else:
        print("✅ JSON consistency check passed")
    
    # Validate individual markdown files
    print("\n📝 Validating markdown files...")
    datasets_dir = Path("_datasets")
    
    if not datasets_dir.exists():
        print("❌ _datasets directory not found")
        return False
    
    markdown_files = list(datasets_dir.glob("*.md"))
    if not markdown_files:
        print("⚠️ No markdown files found in _datasets directory")
        return True
    
    file_issues = 0
    for filepath in markdown_files:
        issues = validate_single_file(filepath)
        if issues:
            print(f"❌ {filepath.name}:")
            for issue in issues:
                print(f"  - {issue}")
            file_issues += len(issues)
        else:
            print(f"✅ {filepath.name}")
    
    total_issues += file_issues
    
    # Summary
    print(f"\n📋 Validation Summary:")
    print(f"  - Files checked: {len(markdown_files)}")
    print(f"  - Total issues: {total_issues}")
    
    if total_issues == 0:
        print("🎉 All validations passed!")
        return True
    else:
        print("❌ Validation failed - please fix the issues above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)