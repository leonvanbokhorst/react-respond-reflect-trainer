#!/usr/bin/env python3
"""Push GGUF model to Hugging Face Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

def push_gguf_to_hub():
    """Upload GGUF model to HF Hub."""
    # Load environment variables
    load_dotenv()
    
    # Set up HF credentials
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    
    if not all([hf_token, hf_username]):
        raise ValueError("Missing HF_TOKEN or HF_USERNAME in .env")
    
    # Initialize HF API
    api = HfApi()
    
    # Set up paths
    work_dir = Path(__file__).parent.absolute()
    model_path = work_dir / "rrr-mistral-gguf" / "q4_k_m" / "unsloth.Q4_K_M.gguf"
    
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found at {model_path}")
    
    # Create repo name
    repo_id = f"{hf_username}/rrr-mistral-gguf"
    
    print(f"\n🚀 Creating repository {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        private=False,
        exist_ok=True
    )
    
    print(f"\n📤 Uploading GGUF model to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="rrr-mistral.Q4_K_M.gguf",
        repo_id=repo_id,
        token=hf_token
    )
    
    # Create a simple README
    readme_content = """# RRR-Mistral GGUF

This is the quantized GGUF version of the React-Respond-Reflect Mistral model. 

## Model Details
- Base Model: Mistral 7B
- Quantization: Q4_K_M
- Format: GGUF (compatible with llama.cpp)

## Usage
This model is designed to follow the React-Respond-Reflect format:
1. React: Internal thought process
2. Respond: Direct response to the user
3. Reflect: Learning from the interaction

## Example with llama.cpp
```bash
./main -m rrr-mistral.Q4_K_M.gguf -n 1024
```
"""
    
    print("\n📝 Creating README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=hf_token
    )
    
    print(f"\n✨ Done! Model uploaded to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_gguf_to_hub() 