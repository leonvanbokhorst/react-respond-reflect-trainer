# Deploying React-Respond-Reflect Model with VLLM

This guide walks you through the process of deploying your fine-tuned React-Respond-Reflect (RRR) model using VLLM for high-performance inference.

## Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Model Export](#model-export)
4. [Running Inference](#running-inference)
5. [API Server](#api-server)
6. [Docker Deployment](#docker-deployment)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

## Overview

The React-Respond-Reflect format provides a structured approach for AI responses with:

- **React**: Internal thought process (invisible in some deployments)
- **Respond**: Direct response to the user
- **Reflect**: Reflection on the interaction (invisible in some deployments)

VLLM is a high-performance inference engine that enables:

- **Fast inference**: 2-5x faster than Hugging Face Transformers
- **Efficient batching**: Optimized for high-throughput scenarios
- **Low memory usage**: Efficient memory management with PagedAttention
- **Tensor parallelism**: Distributed inference across multiple GPUs

## Setup Instructions

### Prerequisites

- CUDA-compatible GPU with at least 12GB VRAM (24GB+ recommended)
- Python 3.8 or later
- Trained RRR model from Unsloth

### Installation

1. Install the required packages:

```bash
pip install vllm peft transformers torch
```

2. Clone or download the provided scripts:

```bash
# vllm-rrr-demo.py - Main script for export and inference
# vllm-client.py - Client for interacting with the VLLM API
# Dockerfile - For containerized deployment
```

## Model Export

Before deploying with VLLM, you need to export and merge the LoRA weights:

```bash
python vllm-rrr-demo.py --mode export \
    --model_path /path/to/rrr_model \
    --merged_path /path/to/rrr_model_merged
```

This process:
1. Loads your fine-tuned RRR model
2. Merges the LoRA weights into the base model
3. Saves the resulting model in a VLLM-compatible format

## Running Inference

Test your model with sample prompts:

```bash
python vllm-rrr-demo.py --mode inference \
    --merged_path /path/to/rrr_model_merged \
    --max_new_tokens 1024 \
    --temperature 0.7
```

This will:
1. Load the merged model with VLLM
2. Run sample prompts through the model
3. Display the React, Respond, and Reflect sections

## API Server

Start a local API server with OpenAI-compatible endpoints:

```bash
python vllm-rrr-demo.py --mode server \
    --merged_path /path/to/rrr_model_merged \
    --port 8000 \
    --tensor_parallel_size 1
```

To interact with the server, use the provided client:

```bash
python vllm-client.py --api_url http://localhost:8000/v1/chat/completions
```

The client provides a command-line interface for:
- Interactive chat with the model
- Toggling the display of React and Reflect sections
- Viewing token usage and generation time

## Docker Deployment

For containerized deployment:

1. Build the Docker image:

```bash
docker build -t rrr-vllm -f Dockerfile .
```

2. Run the container:

```bash
# Mount your model directory
docker run --gpus all -p 8000:8000 \
    -v /path/to/models:/home/user/models \
    -e TENSOR_PARALLEL_SIZE=1 \
    rrr-vllm serve
```

3. For export mode:

```bash
docker run --gpus all \
    -v /path/to/models:/home/user/models \
    rrr-vllm export
```

## Usage Examples

### API Request Format

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "rrr_model",
        "messages": [
            {"role": "system", "content": "You are an empathetic AI assistant..."},
            {"role": "user", "content": "I'm feeling stressed about my presentation"}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
)

print(response.json())
```

### Parsing Responses

```python
import re

def extract_rrr_sections(text):
    sections = {}
    
    react_match = re.search(r'<react>\s*(.*?)\s*</react>', text, re.DOTALL)
    respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', text, re.DOTALL)
    reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', text, re.DOTALL)
    
    sections['react'] = react_match.group(1).strip() if react_match else ""
    sections['respond'] = respond_match.group(1).strip() if respond_match else ""
    sections['reflect'] = reflect_match.group(1).strip() if reflect_match else ""
    
    return sections

# Example usage
response_text = response.json()['choices'][0]['message']['content']
sections = extract_rrr_sections(response_text)

print("React:", sections['react'])
print("Respond:", sections['respond'])
print("Reflect:", sections['reflect'])
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Use a lower tensor_parallel_size
   - Reduce batch size or max sequence length
   - Try running with 8-bit quantization

2. **Missing RRR Format**
   - Ensure your model was properly fine-tuned with the RRR format
   - Check the system prompt includes format instructions
   - Verify the regex patterns for extraction

3. **Slow Performance**
   - Enable CUDA with `export CUDA_VISIBLE_DEVICES=0,1,...`
   - Increase tensor parallelism if multiple GPUs are available
   - Check GPU utilization with `nvidia-smi`

### Verifying VLLM Installation

```bash
python -c "from vllm import LLM; print('VLLM is correctly installed')"
```

If you encounter any issues, check the VLLM documentation or open an issue on the GitHub repository.

---

## Advanced Configuration

### Multi-GPU Deployment

For large models or high-throughput scenarios, distribute the model across multiple GPUs:

```bash
python vllm-rrr-demo.py --mode server \
    --merged_path /path/to/rrr_model_merged \
    --tensor_parallel_size 2 \
    --port 8000
```

### Quantization

VLLM supports various quantization methods:

```bash
# With AWQ quantization
python vllm-rrr-demo.py --mode server \
    --merged_path /path/to/rrr_model_merged \
    --quantization awq
```

### Performance Tuning

Adjust these parameters for optimal performance:

```bash
python vllm-rrr-demo.py --mode server \
    --merged_path /path/to/rrr_model_merged \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --tensor_parallel_size 1
```

---

For more information on VLLM, visit the [official VLLM documentation](https://github.com/vllm-project/vllm).
