#!/usr/bin/env python3
"""
Validation script to ensure all model markdown files follow the established guidelines
Run this to verify that all automation is working correctly
"""

import os
import re
import json
import argparse
from pathlib import Path

def check_layout_usage():
    """Check that all model files use the correct layout"""
    print("🔍 Checking layout usage...")
    models_dir = Path("_models")
    issues = []
    
    for md_file in models_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        layout_match = re.search(r'layout:\s*([^\n]+)', content)
        if layout_match:
            layout = layout_match.group(1).strip().strip('"')
            if layout != "model-implementation":
                issues.append(f"❌ {md_file.name}: uses '{layout}' instead of 'model-implementation'")
        else:
            issues.append(f"❌ {md_file.name}: no layout specified")
    
    if issues:
        print(f"  Found {len(issues)} layout issues:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("  ✅ All files use correct 'model-implementation' layout")
        return True

def check_local_images():
    """Check for any remaining local image references"""
    print("🖼️ Checking for local image references...")
    models_dir = Path("_models")
    issues = []
    
    image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(jpg|jpeg|png|gif|svg))\)'
    
    for md_file in models_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = re.findall(image_pattern, content)
        if matches:
            for match in matches:
                image_path = match[1]
                if not image_path.startswith('http'):
                    issues.append(f"❌ {md_file.name}: local image '{image_path}'")
    
    if issues:
        print(f"  Found {len(issues)} local image references:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("  ✅ No local image references found")
        return True

def check_github_links():
    """Check for proper GitHub image links using raw.githubusercontent.com"""
    print("🔗 Checking GitHub image links...")
    models_dir = Path("_models")
    issues = []

    # Updated pattern to check for proper raw.githubusercontent.com image links
    github_image_pattern = r'!\[([^\]]+)\]\((https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^)]+\.(jpg|jpeg|png|gif|svg|webp))\)'

    for md_file in models_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Flag any concatenated malformed URLs like github.com/.../raw/.../https://raw.githubusercontent.com/...
        malformed_concat = re.findall(r'https://github\.com/[^)]+/raw/[^)]+https://raw\.githubusercontent\.com/[^)]+', content)
        for bad in malformed_concat:
            issues.append(f"❌ {md_file.name}: malformed concatenated URL '{bad}'")

        matches = re.findall(github_image_pattern, content)
        for match in matches:
            if not match[1].startswith("https://raw.githubusercontent.com/YuvrajSingh-mist/"):
                issues.append(f"❌ {md_file.name}: incorrect GitHub image link '{match[1]}'")

    if issues:
        print(f"  Found {len(issues)} incorrect GitHub image links:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("  ✅ All GitHub image links are correct")
        return True

def check_title_consistency():
    """Check title consistency with models.json"""
    print("📝 Checking title consistency...")
    
    # Load models.json
    models_file = Path("_data/models.json")
    if not models_file.exists():
        print("  ❌ models.json not found")
        return False
    
    with open(models_file, 'r') as f:
        data = json.load(f)
    
    models_list = data.get('models', data) if isinstance(data, dict) else data
    json_titles = {model.get('name', '') for model in models_list if isinstance(model, dict)}
    
    # Check markdown files
    models_dir = Path("_models")
    issues = []
    
    for md_file in models_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        title_match = re.search(r'title:\s*"([^"]+)"', content)
        if title_match:
            title = title_match.group(1)
            if title not in json_titles:
                issues.append(f"❌ {md_file.name}: title '{title}' not in models.json")
        else:
            issues.append(f"❌ {md_file.name}: no title found")
    
    if issues:
        print(f"  Found {len(issues)} title inconsistencies:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("  ✅ All titles match models.json")
        return True

def check_file_count():
    """Check that file count matches models.json"""
    print("📊 Checking file counts...")
    
    # Count markdown files
    models_dir = Path("_models")
    md_count = len(list(models_dir.glob("*.md")))
    
    # Count models in JSON
    models_file = Path("_data/models.json")
    if models_file.exists():
        with open(models_file, 'r') as f:
            data = json.load(f)
        
        models_list = data.get('models', data) if isinstance(data, dict) else data
        json_count = len([m for m in models_list if isinstance(m, dict)])
        
        print(f"  📄 Markdown files: {md_count}")
        print(f"  📋 JSON entries: {json_count}")
        
        if md_count == json_count:
            print("  ✅ File count matches JSON entries")
            return True
        else:
            print(f"  ❌ Mismatch: {md_count} files vs {json_count} JSON entries")
            return False
    else:
        print("  ❌ models.json not found")
        return False

def check_automation_setup():
    """Check that automation templates and scripts are in place"""
    print("🤖 Checking automation setup...")
    
    template_path = Path("_templates/model_template.md")
    script_path = Path("create_new_model.py")
    
    issues = []
    
    if not template_path.exists():
        issues.append("❌ Template missing: _templates/model_template.md")
    else:
        print("  ✅ Template found: _templates/model_template.md")
    
    if not script_path.exists():
        issues.append("❌ Script missing: create_new_model.py")
    else:
        print("  ✅ Script found: create_new_model.py")
    
    if issues:
        for issue in issues:
            print(f"    {issue}")
        print("  💡 Run: python standardize_markdowns.py to recreate automation")
        return False
    else:
        print("  ✅ Automation setup complete")
        return True

def main():
    """Run all validation checks"""
    print("🔍 SmolHub Model Generation Validation")
    print("=" * 50)
    
    checks = [
        ("Layout Usage", check_layout_usage),
        ("Local Images", check_local_images),
        ("GitHub Links", check_github_links),
        ("Title Consistency", check_title_consistency),
        ("File Count", check_file_count),
        ("Automation Setup", check_automation_setup),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All guidelines are being followed correctly!")
        print("✅ The automated system is working perfectly!")
        print("🤖 Automation is ready for new model creation!")
        return True
    else:
        print("⚠️  Some issues found. Fixes available:")
        print("   - Layout issues: python standardize_markdowns.py")
        print("   - Image issues: python fix_images_direct.py")
        print("   - Content issues: python regenerate_models.py")
        print("   - Create new models: python create_new_model.py 'Name' 'Category' 'Dataset' 'URL'")
        return False
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All guidelines are being followed correctly!")
        print("✅ The automated system is working perfectly!")
        return True
    else:
        print("⚠️  Some issues found. Please run the appropriate scripts to fix them:")
        print("   - Layout issues: find _models -name '*.md' -exec sed -i 's/layout: paper-replication/layout: model-implementation/g' {} \\;")
        print("   - Image issues: python fix_images_direct.py")
        print("   - Content issues: python regenerate_models.py")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate markdown files against guidelines')
    parser.add_argument('--source', default='paper-replications', 
                       help='Source repository type (default: paper-replications)')
    args = parser.parse_args()
    
    print(f"🔧 Validating guidelines for source: {args.source}")
    
    success = main()
    exit(0 if success else 1)
