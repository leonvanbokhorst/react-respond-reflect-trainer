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
    
    # Create a detailed README
    readme_content = """# RRR-Mistral GGUF 🤖

This is the quantized GGUF version of the React-Respond-Reflect Mistral model, designed to provide structured responses following a three-step format.

## Model Details 📋

### Base Configuration
- **Base Model**: Mistral 7B
- **Architecture**: Mistral (same as base model)
- **Quantization**: Q4_K_M (4-bit, medium precision)
- **Format**: GGUF (compatible with llama.cpp)
- **Context Length**: 2048 tokens

### Training Configuration
- **Framework**: Unsloth (optimized training)
- **LoRA Parameters**:
  - Rank: 16
  - Alpha: 8
  - Target Modules: QKV projections, Output, Gate, Up/Down projections
  - Dropout: 0.05
- **Training Parameters**:
  - Learning Rate: 1e-4
  - Epochs: 3
  - Batch Size: 2 (effective 16 with gradient accumulation)
  - Weight Decay: 0.01
  - Optimizer: AdamW 8-bit
  - Scheduler: Cosine with warmup

## The React-Respond-Reflect Format 🎯

This model is trained to structure its responses in three distinct steps:

1. **React** `<react>*thought process*</react>`
   - Internal reasoning
   - Analysis of the situation
   - Planning the response

2. **Respond** `<respond>direct response</respond>`
   - Clear communication to the user
   - Implementation of the planned response
   - Focused on addressing the user's needs

3. **Reflect** `<reflect>learning & improvement</reflect>`
   - Self-evaluation
   - Lessons learned
   - Areas for improvement

## Usage with llama.cpp 🛠️

1. **Setup**:
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# Download and run
wget https://huggingface.co/leonvanbokhorst/rrr-mistral-gguf/resolve/main/rrr-mistral.Q4_K_M.gguf
./main -m rrr-mistral.Q4_K_M.gguf -n 1024
```

2. **Example Prompt**:
```
<|im_start|>system
You are an empathetic AI assistant.
<|im_end|>
<|im_start|>user
How can I improve my focus while working?
<|im_end|>
<|im_start|>assistant
```

## Dataset 📚

The model was trained on the [React-Respond-Reflect Dialogues v2](https://huggingface.co/datasets/leonvanbokhorst/react-respond-reflect-dialogues-v2) dataset, which contains curated conversations demonstrating the three-step response format.

## License and Usage 📜

This model is intended for research and development in conversational AI. Please use responsibly and in accordance with the base model's license terms.

## Acknowledgments 🙏

- [Mistral AI](https://mistral.ai/) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF support
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