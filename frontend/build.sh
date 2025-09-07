#!/bin/bash

# Build script for Render deployment
set -o errexit

echo "🚀 Starting Render build process..."

# Install Bundler if not available
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler..."
    gem install bundler --no-document
fi

echo "Ruby version:"
ruby --version

echo "Bundler version:"
bundle --version

echo "Node.js version:"
node --version

# Configure bundle for Render environment
echo "🔧 Configuring Bundle for Render..."
bundle config set --local deployment 'false'
bundle config set --local path 'vendor/bundle'
bundle config set --local without 'development test'

echo "📦 Installing Ruby dependencies..."
bundle install

echo "🏗️ Building Jekyll site for production..."
JEKYLL_ENV=production bundle exec jekyll build

echo "✅ Build completed successfully!"
bundle update

echo "Installing Node.js dependencies..."
npm install

echo "Generating models data..."
if [ -n "$GITHUB_TOKEN" ]; then
    echo "✅ GitHub token found, generating models data..."
    if node generate-models-data.js; then
        echo "✅ Models data generated successfully"
        echo "📊 Models data file info:"
        ls -la _data/models.json
        echo "🔍 Models count:"
        jq '.models | length' _data/models.json 2>/dev/null || echo "Could not parse models count"
    else
        echo "❌ Failed to generate models data, using existing data"
    fi
else
    echo "⚠️ No GitHub token found, using existing models data..."
    if [ -f "_data/models.json" ]; then
        echo "✅ Existing models data found"
        ls -la _data/models.json
    else
        echo "❌ No models data available - models page may not work correctly"
    fi
fi

echo "Building Jekyll site..."
JEKYLL_ENV=production bundle exec jekyll build --verbose

echo "Build completed successfully!"
echo "Site files:"
ls -la _site/
