#!/bin/bash

# Paper Replications Update Script
# Usage: ./update_models.sh [--fetch] [--token YOUR_TOKEN]

set -e

echo "🚀 Paper Replications Update Script"
echo "=================================="

# Parse arguments
FETCH_FLAG=""
TOKEN_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --fetch)
      FETCH_FLAG="--fetch"
      echo "🔄 Will fetch latest data from GitHub"
      shift
      ;;
    --token)
      TOKEN_FLAG="--github-token $2"
      echo "🔑 Using provided GitHub token"
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--fetch] [--token YOUR_TOKEN]"
      echo ""
      echo "Options:"
      echo "  --fetch         Fetch latest data from GitHub API"
      echo "  --token TOKEN   Use GitHub token for authentication"
      echo "  -h, --help      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if required files exist
if [ ! -f "update_models.py" ]; then
    echo "❌ update_models.py not found!"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
if ! python3 -c "import requests" &> /dev/null; then
    echo "Installing required Python packages..."
    pip3 install requests
fi

# Create backup of current state
echo "💾 Creating backup..."
if [ -d "_models" ]; then
    cp -r _models _models_backup_$(date +%Y%m%d_%H%M%S)
fi

# Run the update script
echo "🔄 Running update script..."
python3 update_models.py $FETCH_FLAG $TOKEN_FLAG

# Check if any files were generated
if [ -d "_models" ] && [ "$(ls -A _models)" ]; then
    MODEL_COUNT=$(ls -1 _models/*.md | wc -l)
    echo "✅ Successfully generated $MODEL_COUNT model files"
else
    echo "⚠️  No model files were generated"
fi

# Show status
echo ""
echo "📊 Current Status:"
echo "=================="
if [ -d "_models" ]; then
    echo "📁 Models folder: EXISTS"
    echo "📄 Model files: $(ls -1 _models/*.md 2>/dev/null | wc -l)"
else
    echo "📁 Models folder: MISSING"
fi

if [ -f "_data/models.json" ]; then
    echo "📄 models.json: EXISTS"
else
    echo "📄 models.json: MISSING"
fi

echo ""
echo "🎉 Update complete!"
echo ""
echo "Next steps:"
echo "1. Review the generated files in _models/"
echo "2. Test locally with: bundle exec jekyll serve"
echo "3. Commit and push changes to deploy"

# Offer to run Jekyll serve if available
if command -v bundle &> /dev/null; then
    echo ""
    read -p "Would you like to start Jekyll development server? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🌐 Starting Jekyll development server..."
        bundle exec jekyll serve --host 0.0.0.0 --port 4000
    fi
fi
