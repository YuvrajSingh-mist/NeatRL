#!/bin/bash

echo "========================================"
echo "   Portfolio Models Data Generator"
echo "========================================"
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed or not in PATH"
    echo "💡 Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if token is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Error: GITHUB_TOKEN environment variable is required"
    echo
    echo "💡 Usage:"
    echo "   export GITHUB_TOKEN=your_token_here"
    echo "   ./update-models.sh"
    echo
    echo "Or run it in one line:"
    echo "   GITHUB_TOKEN=your_token_here ./update-models.sh"
    echo
    exit 1
fi

echo "🚀 Generating models data with authenticated GitHub API..."
echo

# Run the Node.js script
node generate-models-data.js

if [ $? -eq 0 ]; then
    echo
    echo "✅ Success! Models data has been updated."
    echo "💡 Next steps:"
    echo "   1. Review the generated _data/models.json file"
    echo "   2. Commit and push the changes to GitHub"
    echo "   3. Your Jekyll site will rebuild automatically"
else
    echo
    echo "❌ Failed to generate models data. Check the error messages above."
fi

echo
