#!/bin/bash

# SmolHub Playground Refresh Script
# This script fetches SmolHub playground data from GitHub and regenerates markdown files

echo "🎮 SmolHub Playground Refresh Script"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "📋 Checking dependencies..."
if ! command_exists node; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

if ! command_exists python; then
    echo "❌ Python is required but not installed."
    exit 1
fi

echo "✅ Dependencies check passed."

# Step 1: Generate SmolHub data from GitHub
echo ""
echo "📡 Step 1: Fetching SmolHub playground data from GitHub..."
if node generate-smolhub-data.js; then
    echo "✅ SmolHub data generated successfully"
else
    echo "❌ Failed to generate SmolHub data"
    exit 1
fi

# Step 2: Generate markdown files
echo ""
echo "📝 Step 2: Generating SmolHub playground markdown files..."
if python generate-smolhub-markdowns.py; then
    echo "✅ SmolHub markdown files generated successfully"
else
    echo "❌ Failed to generate SmolHub markdown files"
    exit 1
fi

echo ""
echo "🎉 SmolHub playground refresh completed successfully!"
echo ""
echo "📋 Summary:"
echo "   📄 Data file: _data/smolhub_playground.json"
echo "   📁 Markdown files: _smolhub/playground-*.md"
echo ""
echo "🚀 You can now commit and push the changes to deploy the updates."
echo "   git add _data/smolhub_playground.json _smolhub/playground-*.md"
echo "   git commit -m \"🎮 Update SmolHub playground data and markdown files\""
echo "   git push"
