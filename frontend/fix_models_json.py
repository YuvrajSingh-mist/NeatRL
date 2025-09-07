#!/usr/bin/env python3
"""
Fix models.json file to remove 'Paper Replications' references
"""

import json
import os

def fix_models_json():
    """Fix the models.json file"""
    json_path = '_data/models.json'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} not found!")
        return
    
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_count = 0
    
    # Fix each model entry
    for model in data.get('models', []):
        # Fix description
        if 'description' in model:
            old_desc = model['description']
            new_desc = old_desc.replace('from the Paper Replications repository', 'from scratch')
            new_desc = new_desc.replace('Paper Replications repository', 'From Scratch Implementation')
            if new_desc != old_desc:
                model['description'] = new_desc
                fixed_count += 1
                print(f"Fixed description for: {model.get('name', 'Unknown')}")
        
        # Fix readme_content if it contains references
        if 'readme_content' in model:
            old_content = model['readme_content']
            new_content = old_content.replace('Paper Replications repository', 'From Scratch Implementation')
            new_content = new_content.replace('Paper-Replications', 'From-Scratch-Implementation')
            if new_content != old_content:
                model['readme_content'] = new_content
                print(f"Fixed readme content for: {model.get('name', 'Unknown')}")
    
    # Write back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFixed models.json file! Updated {fixed_count} descriptions.")

if __name__ == '__main__':
    fix_models_json()
