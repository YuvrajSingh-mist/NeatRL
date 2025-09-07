#!/bin/bash

# Deployment verification script
echo "🔍 Checking deployment requirements..."

# Check if _data/models.json exists
if [ -f "_data/models.json" ]; then
    echo "✅ _data/models.json exists"
    
    # Check file size
    file_size=$(stat -f%z "_data/models.json" 2>/dev/null || stat -c%s "_data/models.json" 2>/dev/null)
    echo "📏 File size: $file_size bytes"
    
    # Check if it's valid JSON
    if jq empty _data/models.json 2>/dev/null; then
        echo "✅ Valid JSON format"
        
        # Count models
        model_count=$(jq '.models | length' _data/models.json 2>/dev/null)
        echo "📊 Models count: $model_count"
    else
        echo "❌ Invalid JSON format"
    fi
else
    echo "❌ _data/models.json not found"
fi

# Check if static model files exist
model_files_count=$(ls _models/*.md 2>/dev/null | wc -l)
echo "📁 Static model files: $model_files_count"

# Check if Jekyll can build
echo "🔧 Testing Jekyll build..."
if bundle exec jekyll build --quiet --destination _test_site; then
    echo "✅ Jekyll build successful"
    rm -rf _test_site
else
    echo "❌ Jekyll build failed"
fi

echo "🏁 Verification complete"
