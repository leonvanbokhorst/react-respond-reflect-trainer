#!/bin/bash
# Universal script to upload the RRR model to Hugging Face (works on Linux, WSL, and Mac)

# Default repo name
DEFAULT_REPO="leonvanbokhorst/react-respond-reflect-model"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

# Check if we're in a virtual environment already
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Not in a virtual environment.${NC}"
    
    # Check for existing virtual environments
    if [ -d "venv" ]; then
        echo -e "${BLUE}Found 'venv' directory. Activating...${NC}"
        source venv/bin/activate
    elif [ -d "venv_mac" ]; then
        echo -e "${BLUE}Found 'venv_mac' directory. Activating...${NC}"
        source venv_mac/bin/activate
    else
        echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        
        echo -e "${BLUE}Installing required packages...${NC}"
        pip install --upgrade pip
        pip install huggingface_hub python-dotenv
    fi
else
    echo -e "${GREEN}Already in virtual environment: $VIRTUAL_ENV${NC}"
fi

# Check if upload_to_hf.py exists
if [ ! -f "upload_to_hf.py" ]; then
    echo -e "${RED}upload_to_hf.py not found. Please make sure it's in the current directory.${NC}"
    exit 1
fi

# Check if .env exists and contains HF_TOKEN
if [ ! -f ".env" ] || ! grep -q "HF_TOKEN" .env; then
    echo -e "${YELLOW}HF_TOKEN not found in .env file.${NC}"
    read -p "Enter your Hugging Face token: " hf_token
    echo "HF_TOKEN=$hf_token" >> .env
    echo -e "${GREEN}Token saved to .env${NC}"
fi

# Ask for repo name
read -p "Enter Hugging Face repository name (default: $DEFAULT_REPO): " repo_name
repo_name=${repo_name:-$DEFAULT_REPO}

# Run upload script
echo -e "${BLUE}Uploading model to $repo_name...${NC}"
python upload_to_hf.py --repo_name "$repo_name"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Model uploaded successfully!${NC}"
    echo -e "${GREEN}Model available at: https://huggingface.co/$repo_name${NC}"
    
    # Update demo script - handle both Mac and Linux sed syntax
    echo -e "${BLUE}Updating demo script to use the uploaded model...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OS X
        sed -i '' "s|default=\"leonvanbokhorst/react-respond-reflect-model\"|default=\"$repo_name\"|g" rrr_demo.py
    else
        # Linux/WSL
        sed -i "s|default=\"leonvanbokhorst/react-respond-reflect-model\"|default=\"$repo_name\"|g" rrr_demo.py
    fi
    
    echo -e "${GREEN}✅ Setup complete! You can now run the demo with:${NC}"
    echo "python rrr_demo.py"
else
    echo -e "${RED}❌ Failed to upload model.${NC}"
    echo -e "${RED}Please check the error messages above.${NC}"
fi 