#!/usr/bin/env python3

"""
SmolHub Playground Validation Script
Validates that all SmolHub components are working correctly.
"""

import json
import os
from pathlib import Path

def validate_smolhub():
    """Validate SmolHub playground setup"""
    print("🎮 SmolHub Playground Validation")
    print("================================")
    
    errors = []
    warnings = []
    
    # Check data file
    data_file = Path('_data/smolhub_playground.json')
    if not data_file.exists():
        errors.append("❌ SmolHub data file missing: _data/smolhub_playground.json")
    else:
        print("✅ SmolHub data file found")
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"   📊 Projects in data: {data.get('total_projects', 0)}")
            
            # Validate data structure
            if 'projects' not in data:
                errors.append("❌ No 'projects' key in data file")
            else:
                projects = data['projects']
                for i, project in enumerate(projects):
                    required_fields = ['name', 'display_name', 'description', 'github_url']
                    for field in required_fields:
                        if field not in project:
                            warnings.append(f"⚠️  Project {i+1} missing field: {field}")
                            
        except json.JSONDecodeError as e:
            errors.append(f"❌ Invalid JSON in data file: {e}")
    
    # Check markdown files
    smolhub_dir = Path('_smolhub')
    if not smolhub_dir.exists():
        errors.append("❌ SmolHub directory missing: _smolhub/")
    else:
        print("✅ SmolHub directory found")
        
        # Count files
        playground_files = list(smolhub_dir.glob('playground-*.md'))
        project_files = list(smolhub_dir.glob('project-*.md'))
        
        print(f"   📄 Playground files: {len(playground_files)}")
        print(f"   📄 Manual project files: {len(project_files)}")
        
        # Validate markdown frontmatter
        for md_file in playground_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.startswith('---'):
                    warnings.append(f"⚠️  {md_file.name} missing frontmatter")
                    
                # Check for required frontmatter fields
                required_fm = ['title:', 'collection: smolhub', 'github_url:']
                for field in required_fm:
                    if field not in content:
                        warnings.append(f"⚠️  {md_file.name} missing: {field}")
                        
            except Exception as e:
                errors.append(f"❌ Error reading {md_file.name}: {e}")
    
    # Check page file
    page_file = Path('_pages/smolhub.html')
    if not page_file.exists():
        errors.append("❌ SmolHub page missing: _pages/smolhub.html")
    else:
        print("✅ SmolHub page found")
        try:
            with open(page_file, 'r') as f:
                content = f.read()
            
            required_elements = [
                'smolhub-github-data',
                'site.data.smolhub_playground',
                'initializeSmolHubEnhancement'
            ]
            
            for element in required_elements:
                if element not in content:
                    warnings.append(f"⚠️  SmolHub page missing: {element}")
                    
        except Exception as e:
            errors.append(f"❌ Error reading SmolHub page: {e}")
    
    # Check scripts
    scripts = [
        'generate-smolhub-data.js',
        'generate-smolhub-markdowns.py',
        'refresh_smolhub.sh'
    ]
    
    for script in scripts:
        if not Path(script).exists():
            warnings.append(f"⚠️  Script missing: {script}")
        else:
            print(f"✅ Script found: {script}")
    
    # Summary
    print("\n📋 Validation Summary:")
    print(f"   ❌ Errors: {len(errors)}")
    print(f"   ⚠️  Warnings: {len(warnings)}")
    
    if errors:
        print("\n❌ Errors found:")
        for error in errors:
            print(f"   {error}")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   {warning}")
    
    if not errors and not warnings:
        print("\n🎉 All SmolHub playground components are valid!")
        print("🚀 Your playground is ready to use!")
    elif not errors:
        print("\n✅ SmolHub playground setup is functional with minor warnings.")
    else:
        print("\n❌ SmolHub playground has errors that need to be fixed.")
        return False
    
    return True

if __name__ == "__main__":
    validate_smolhub()
