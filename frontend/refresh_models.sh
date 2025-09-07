#!/bin/bash

# Script to refresh models and rebuild Jekyll site
# Run this script when you want to update models from the repository

echo "🔄 Refreshing models from Paper-Replications repository..."

# Run the Python refresh script
python3 refresh_models.py

if [ $? -eq 0 ]; then
    echo "✅ Models refreshed successfully!"
    
    # Check if Jekyll is running and restart it
    if pgrep -f "jekyll serve" > /dev/null; then
        echo "🔄 Restarting Jekyll server..."
        pkill -f "jekyll serve"
        sleep 2
        bundle exec jekyll serve --livereload &
        echo "🚀 Jekyll server restarted with live reload"
    else
        echo "📝 Models updated. Restart your Jekyll server to see changes."
    fi
else
    echo "❌ Failed to refresh models"
    exit 1
fi
