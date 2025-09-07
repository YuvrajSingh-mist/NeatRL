#!/bin/bash
# Start script for Render Web Service

echo "🌐 Starting Jekyll server for production..."

# Set the port (Render provides this automatically)
PORT=${PORT:-4000}

echo "📡 Starting server on port $PORT..."

# Ensure gems are available
export BUNDLE_PATH="vendor/bundle"

# Start Jekyll server for production
bundle exec jekyll serve \
  --host 0.0.0.0 \
  --port $PORT \
  --config _config.yml \
  --skip-initial-build \
  --verbose

echo "🚀 Jekyll server started successfully!"
