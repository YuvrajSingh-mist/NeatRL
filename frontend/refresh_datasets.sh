#!/bin/bash

# Refresh Datasets Script
# This script refreshes all dataset-related files and validates them

echo "🔄 Refreshing datasets..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${2}${1}${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_status "❌ Python 3 is required but not installed" $RED
    exit 1
fi

# Check if required files exist
required_files=("auto_detect_datasets.py" "generate_dataset_files.py" "validate_datasets.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_status "❌ Required file missing: $file" $RED
        exit 1
    fi
done

# Step 1: Auto-detect new datasets
print_status "🔍 Auto-detecting new datasets..." $BLUE
if python3 auto_detect_datasets.py; then
    print_status "✅ Dataset detection completed" $GREEN
else
    print_status "❌ Dataset detection failed" $RED
    exit 1
fi

# Step 2: Generate markdown files
print_status "📝 Generating dataset markdown files..." $BLUE
if python3 generate_dataset_files.py; then
    print_status "✅ Markdown generation completed" $GREEN
else
    print_status "❌ Markdown generation failed" $RED
    exit 1
fi

# Step 3: Validate datasets
print_status "🔍 Validating dataset files..." $BLUE
if python3 validate_datasets.py; then
    print_status "✅ Dataset validation passed" $GREEN
else
    print_status "⚠️ Dataset validation found issues (check output above)" $YELLOW
fi

# Step 4: Check for changes
if git diff --quiet _data/datasets.json _datasets/; then
    print_status "ℹ️ No changes detected" $BLUE
else
    print_status "📋 Changes detected in dataset files:" $YELLOW
    git diff --name-only _data/datasets.json _datasets/ | while read file; do
        print_status "  - $file" $YELLOW
    done
    
    # Ask if user wants to commit changes
    read -p "Do you want to commit these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add _data/datasets.json _datasets/
        git commit -m "🔄 Refresh datasets: $(date '+%Y-%m-%d %H:%M:%S')"
        print_status "✅ Changes committed" $GREEN
    else
        print_status "ℹ️ Changes not committed" $BLUE
    fi
fi

print_status "🎉 Dataset refresh completed!" $GREEN
