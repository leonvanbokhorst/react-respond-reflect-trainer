"""
Upload React-Respond-Reflect model to Hugging Face Hub

This script uploads the fine-tuned RRR model to Hugging Face Hub
for easy sharing and downloading.
"""

import os
import argparse
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import shutil
import json

def create_model_card(output_path, model_name):
    """
    Create a README.md model card for the Hugging Face Hub.
    
    Args:
        output_path: Path to save the model card
        model_name: Name of the model on HF Hub
    """
    model_card = f"""---
language: en
license: apache-2.0
tags:
- react-respond-reflect
- mistral-7b
- lora
- conversational
- empathy
- coaching
datasets:
- leonvanbokhorst/react-respond-reflect-dialogues-v2
---

# React-Respond-Reflect Model 🎭

This is a fine-tuned version of Mistral-7B that follows the React-Respond-Reflect framework for more empathetic and structured conversations.

## Model Description

- **Base Model**: [unsloth/mistral-7b-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-bnb-4bit)
- **Training Method**: LoRA fine-tuning with Unsloth
- **Framework**: React-Respond-Reflect (RRR)

## Usage

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_name}",
    max_seq_length=2048,
    load_in_4bit=True,  # Set to False for Mac compatibility
)

# Configure tokenizer with ChatML template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    map_eos_token=True,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

# Generate a response
messages = [{{"role": "user", "content": "I'm feeling anxious about my job interview tomorrow. Any advice?"}}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=2048,
    temperature=0.7,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## Response Format

The model responses follow the React-Respond-Reflect format:

- **React**: Physical/emotional reactions expressed through actions and body language
- **Respond**: The actual verbal response to the user
- **Reflect**: Internal thoughts and analysis of the conversation

Example:
```
<react>*leans forward with an understanding nod, eyes showing empathy*</react>
<respond>It's completely normal to feel anxious before a job interview. Your body is actually helping you prepare by getting your energy up! Have you thought about what specific parts of the interview are making you most nervous?</respond>
<reflect>They're experiencing anticipatory anxiety, which is healthy but can be overwhelming. I should validate their feelings while offering practical strategies. I'll check what specific concerns they have to provide targeted advice.</reflect>
```

## Mac M3 Compatibility

For Apple Silicon (M3) users:
- Set `load_in_4bit=False` when loading the model
- Use `device_map="mps"` for Metal Performance Shaders acceleration
- If you encounter issues, fall back to CPU with `device_map="cpu"`

## Example Prompts

Try these prompts to see the model in action:

- "I'm feeling anxious about my job interview tomorrow. Any advice?"
- "How can I improve my focus when working from home?"
- "I'm struggling with imposter syndrome in my new role."
- "What are some strategies for managing my time better?"
- "I feel overwhelmed by all the tasks I need to complete."
"""

    with open(output_path, "w") as f:
        f.write(model_card)
    
    print(f"Created model card at {output_path}")

def upload_model(model_path, repo_name, token=None, create_pr=False):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to the model directory
        repo_name: Name of the repository on HF Hub
        token: HF API token (if not provided, will look for HF_TOKEN env var)
        create_pr: Whether to create a PR instead of pushing directly
    """
    # Load token from .env if not provided
    if token is None:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token is None:
            # Try alternative env var names
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token is None:
                raise ValueError("No Hugging Face token provided. Please set HF_TOKEN in .env or pass --token")
    
    # Login to Hugging Face
    login(token=token)
    api = HfApi()
    
    # Create a temporary directory for the model
    temp_dir = "temp_model_upload"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copy model files to temp directory
        for file in os.listdir(model_path):
            # Skip checkpoint directories
            if file.startswith("checkpoint-"):
                continue
            
            src = os.path.join(model_path, file)
            dst = os.path.join(temp_dir, file)
            
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Copied {file}")
        
        # Create model card
        create_model_card(os.path.join(temp_dir, "README.md"), repo_name)
        
        # Create config.json if it doesn't exist
        config_path = os.path.join(temp_dir, "config.json")
        if not os.path.exists(config_path):
            # Create a minimal config
            config = {
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "torch_dtype": "float16",
                "_name_or_path": "unsloth/mistral-7b-bnb-4bit",
                "transformers_version": "4.37.2"
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("Created config.json")
        
        # Upload to Hugging Face
        print(f"Uploading model to {repo_name}...")
        if create_pr:
            api.create_repo(repo_id=repo_name, exist_ok=True)
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                commit_message="Upload React-Respond-Reflect model",
                create_pr=True
            )
            print(f"Created PR for {repo_name}")
        else:
            api.create_repo(repo_id=repo_name, exist_ok=True)
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                commit_message="Upload React-Respond-Reflect model"
            )
            print(f"Uploaded model to {repo_name}")
        
        print(f"Model available at: https://huggingface.co/{repo_name}")
        
        # Update the demo script to use the HF model
        update_demo_script(repo_name)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def update_demo_script(repo_name):
    """
    Update the demo script to use the HF model.
    
    Args:
        repo_name: Name of the repository on HF Hub
    """
    demo_path = "rrr_demo.py"
    if os.path.exists(demo_path):
        with open(demo_path, "r") as f:
            content = f.read()
        
        # Replace the model loading code
        updated_content = content.replace(
            '        model_name="unsloth/mistral-7b-bnb-4bit",  # Base model',
            f'        model_name="{repo_name}",  # HF model'
        )
        
        with open(demo_path, "w") as f:
            f.write(updated_content)
        
        print(f"Updated {demo_path} to use {repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload React-Respond-Reflect model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, default="rrr_model",
                        help="Path to the model directory")
    parser.add_argument("--repo_name", type=str, default="leonvanbokhorst/react-respond-reflect-model",
                        help="Name of the repository on HF Hub (username/repo-name)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token (if not provided, will look for HF_TOKEN env var)")
    parser.add_argument("--create_pr", action="store_true",
                        help="Create a PR instead of pushing directly")
    
    args = parser.parse_args()
    
    upload_model(args.model_path, args.repo_name, args.token, args.create_pr)

if __name__ == "__main__":
    main() 