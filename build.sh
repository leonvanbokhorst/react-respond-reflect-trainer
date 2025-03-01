#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building model API Docker image...${NC}"

# Copy necessary files for the build
echo -e "${YELLOW}Copying API files...${NC}"
cp api_*.py .env requirements.txt deployment-requirements.txt healthcheck.py ./

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t model:latest -f Dockerfile.model .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker image built successfully!${NC}"
    echo -e "${GREEN}You can now run the service with:${NC}"
    echo -e "  docker-compose up -d"
else
    echo -e "${RED}Docker build failed!${NC}"
    exit 1
fi 