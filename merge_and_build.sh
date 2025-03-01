#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage information
function show_usage {
    echo "Usage: $0 --base-model PATH --adapter PATH [--output-path PATH] [--dtype TYPE]"
    echo
    echo "Options:"
    echo "  --base-model PATH   Path to the base model (required)"
    echo "  --adapter PATH      Path to the LoRA adapter (required)"
    echo "  --output-path PATH  Path to save merged model (default: ./rrr_model)"
    echo "  --dtype TYPE        Model dtype: float16, bfloat16, or float32 (default: bfloat16)"
    echo "  --skip-build        Skip Docker build after merging"
    echo "  --help              Show this help message"
    exit 1
}

# Default values
OUTPUT_PATH="./rrr_model"
DTYPE="bfloat16"
SKIP_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$BASE_MODEL" ] || [ -z "$ADAPTER_PATH" ]; then
    echo -e "${RED}Error: Base model and adapter paths are required.${NC}"
    show_usage
fi

echo -e "${YELLOW}Setting up full-precision model deployment pipeline...${NC}"

# Install required packages if needed
echo -e "${YELLOW}Checking for required packages...${NC}"
if ! pip show peft &>/dev/null; then
    echo -e "${YELLOW}Installing required packages for merging...${NC}"
    pip install -r merge_requirements.txt
fi

# Step 1: Merge model
echo -e "${YELLOW}Merging LoRA adapter with base model (using ${DTYPE} precision)...${NC}"
python merge_model.py \
    --base-model-path "$BASE_MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --output-path "$OUTPUT_PATH" \
    --torch-dtype "$DTYPE"

# Check if merge was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Model merging failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Model successfully merged to $OUTPUT_PATH${NC}"
echo -e "${GREEN}This is a full-precision model setup for higher quality inference.${NC}"

# Step 2: Build Docker image (unless skipped)
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    ./build.sh
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All done! You can now run the service with:${NC}"
        echo -e "  docker-compose up -d"
    else
        echo -e "${RED}Docker build failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping Docker build as requested.${NC}"
    echo -e "${GREEN}To build the Docker image later, run:${NC}"
    echo -e "  ./build.sh"
fi 