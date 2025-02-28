#!/bin/bash
# Quick script to upload the RRR model to Hugging Face from Mac

# Default repo name
DEFAULT_REPO="leonvanbokhorst/react-respond-reflect-model"

# Check if virtual environment exists
if [ ! -d "venv_mac" ]; then
  echo "Virtual environment not found. Please run setup_mac_demo.sh first."
  exit 1
fi

# Activate virtual environment
source venv_mac/bin/activate

# Check if upload_to_hf.py exists
if [ ! -f "upload_to_hf.py" ]; then
  echo "upload_to_hf.py not found. Please make sure it's in the current directory."
  exit 1
fi

# Check if .env exists and contains HF_TOKEN
if [ ! -f ".env" ] || ! grep -q "HF_TOKEN" .env; then
  echo "HF_TOKEN not found in .env file."
  read -p "Enter your Hugging Face token: " hf_token
  echo "HF_TOKEN=$hf_token" >> .env
  echo "Token saved to .env"
fi

# Ask for repo name
read -p "Enter Hugging Face repository name (default: $DEFAULT_REPO): " repo_name
repo_name=${repo_name:-$DEFAULT_REPO}

# Run upload script
echo "Uploading model to $repo_name..."
python upload_to_hf.py --repo_name "$repo_name"

if [ $? -eq 0 ]; then
  echo "✅ Model uploaded successfully!"
  echo "Model available at: https://huggingface.co/$repo_name"
  
  # Update demo script
  echo "Updating demo script to use the uploaded model..."
  sed -i '' "s|default=\"leonvanbokhorst/react-respond-reflect-model\"|default=\"$repo_name\"|g" rrr_demo.py
  
  echo "✅ Setup complete! You can now run the demo with:"
  echo "python rrr_demo.py"
else
  echo "❌ Failed to upload model."
  echo "Please check the error messages above."
fi 