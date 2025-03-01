#!/bin/bash
# Cleanup script to remove unused Ray implementation files

echo "Cleaning up unused Ray implementation files..."

# Create a backup directory
mkdir -p backup_ray_files

# Move Ray implementation files to backup directory
mv api_serve_deployment.py backup_ray_files/ 2>/dev/null
mv api_config.py backup_ray_files/ 2>/dev/null
mv api_model_deployment.py backup_ray_files/ 2>/dev/null
mv api_main.py backup_ray_files/ 2>/dev/null
mv api_healthcheck.py backup_ray_files/ 2>/dev/null
mv api_client.py backup_ray_files/ 2>/dev/null
mv api_models.py backup_ray_files/ 2>/dev/null
mv healthcheck.py backup_ray_files/ 2>/dev/null
mv Dockerfile.model backup_ray_files/ 2>/dev/null

# Move test files created during debugging
mv test_ray_client.py backup_ray_files/ 2>/dev/null
mv test_api_httpx.py backup_ray_files/ 2>/dev/null
mv test_server.py backup_ray_files/ 2>/dev/null

echo "Files moved to backup_ray_files/ directory."
echo "You can delete this directory when you're sure you don't need these files anymore."
echo "To delete the backup directory, run: rm -rf backup_ray_files"

echo "Cleanup complete!" 