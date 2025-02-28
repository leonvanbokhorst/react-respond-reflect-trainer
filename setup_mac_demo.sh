#!/bin/bash
# Setup script for React-Respond-Reflect demo on Mac M3

# Print colorful messages
print_step() {
  echo -e "\n\033[1;36m==== $1 ====\033[0m"
}

print_success() {
  echo -e "\033[1;32m$1\033[0m"
}

print_warning() {
  echo -e "\033[1;33m$1\033[0m"
}

print_error() {
  echo -e "\033[1;31m$1\033[0m"
}

# Check if Python 3.10+ is installed
print_step "Checking Python version"
if command -v python3 >/dev/null 2>&1; then
  python_version=$(python3 --version | cut -d' ' -f2)
  python_major=$(echo $python_version | cut -d'.' -f1)
  python_minor=$(echo $python_version | cut -d'.' -f2)
  
  if [ "$python_major" -ge 3 ] && [ "$python_minor" -ge 10 ]; then
    print_success "Python $python_version detected ✓"
  else
    print_warning "Python 3.10+ recommended. You have $python_version"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi
else
  print_error "Python 3 not found. Please install Python 3.10+"
  exit 1
fi

# Create virtual environment
print_step "Creating virtual environment"
if [ -d "venv_mac" ]; then
  print_warning "Virtual environment 'venv_mac' already exists"
  read -p "Recreate? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf venv_mac
    python3 -m venv venv_mac
    print_success "Virtual environment created ✓"
  fi
else
  python3 -m venv venv_mac
  print_success "Virtual environment created ✓"
fi

# Activate virtual environment
print_step "Activating virtual environment"
source venv_mac/bin/activate
print_success "Virtual environment activated ✓"

# Upgrade pip
print_step "Upgrading pip"
pip install --upgrade pip
print_success "Pip upgraded ✓"

# Install dependencies
print_step "Installing dependencies (this may take a while)"
pip install -r requirements_mac.txt
print_success "Dependencies installed ✓"

# Check if model exists locally
print_step "Checking for model files"
if [ -d "rrr_model" ] && [ -f "rrr_model/adapter_model.safetensors" ]; then
  print_success "Local model files found ✓"
  
  # Ask if user wants to upload to HF
  read -p "Do you want to upload the model to Hugging Face? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check for HF token
    if [ -f ".env" ] && grep -q "HF_TOKEN" .env; then
      print_success "Hugging Face token found in .env ✓"
    else
      print_warning "No Hugging Face token found in .env"
      read -p "Enter your Hugging Face token: " hf_token
      echo "HF_TOKEN=$hf_token" >> .env
      print_success "Token saved to .env ✓"
    fi
    
    # Ask for repo name
    read -p "Enter Hugging Face repository name (default: leonvanbokhorst/react-respond-reflect-model): " repo_name
    repo_name=${repo_name:-leonvanbokhorst/react-respond-reflect-model}
    
    # Run upload script
    print_step "Uploading model to Hugging Face"
    python upload_to_hf.py --repo_name "$repo_name"
    
    if [ $? -eq 0 ]; then
      print_success "Model uploaded to Hugging Face ✓"
      print_success "Model available at: https://huggingface.co/$repo_name"
    else
      print_error "Failed to upload model to Hugging Face"
    fi
  fi
else
  print_warning "Local model files not found in ./rrr_model"
  print_warning "Will use the model from Hugging Face instead"
fi

# Instructions for running the demo
print_step "Setup complete!"
print_success "To run the demo:"
echo "1. Activate the virtual environment: source venv_mac/bin/activate"
echo "2. Run the demo script: python rrr_demo.py"
echo "   - For CPU-only mode: python rrr_demo.py --device cpu"
echo "   - To specify a different model path: python rrr_demo.py --model_path /path/to/model"
echo "   - To use a specific HF model: python rrr_demo.py --model_path username/repo-name"

print_warning "Note: First run may take longer as models are downloaded and cached" 