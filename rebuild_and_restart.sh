#!/bin/bash
# Script to rebuild the Docker image and restart the container

set -e  # Exit on error

echo "🔄 Stopping existing containers..."
docker-compose down

echo "🏗️ Building new Docker image..."
docker build -t simple-api:latest -f Dockerfile.simple .

echo "🚀 Starting containers..."
docker-compose up -d

echo "📋 Checking container status..."
docker-compose ps

echo "📜 Viewing logs (press Ctrl+C to exit)..."
docker-compose logs -f simple-api 