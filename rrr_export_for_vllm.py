#!/usr/bin/env python3
"""
Export a trained React-Respond-Reflect model for vLLM serving.

This script prepares a fine-tuned RRR model for deployment with vLLM by:
1. Merging LoRA adapters with the base model
2. Converting to a format compatible with vLLM
3. Saving the tokenizer with the appropriate chat template

Example:
    $ python export_for_vllm.py --input_dir ./rrr_model --output_dir ./rrr_model_vllm

Args:
    --input_dir: Directory containing the trained model
    --output_dir: Directory to save the exported model
    --base_model: Base model name (default: unsloth/mistral-7b-bnb-4bit)
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export RRR model for vLLM")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the exported model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/mistral-7b-bnb-4bit",
        help="Base model name",
    )
    return parser.parse_args()


def export_model(
    input_dir: str,
    output_dir: str,
    base_model: str,
) -> None:
    """
    Export a trained model for vLLM.

    Args:
        input_dir: Directory containing the trained model
        output_dir: Directory to save the exported model
        base_model: Base model name
    """
    print(f"🔄 Exporting model from {input_dir} to {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base model and tokenizer
    print(f"📥 Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load and merge LoRA adapters
    print(f"🔀 Merging LoRA adapters from {input_dir}")
    adapter_path = Path(input_dir)
    if not adapter_path.exists():
        raise ValueError(f"Adapter path {adapter_path} does not exist")

    # Find checkpoint directory
    checkpoint_dirs = list(adapter_path.glob("checkpoint-*"))
    if checkpoint_dirs:
        # Use the latest checkpoint
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        adapter_path = checkpoint_dirs[-1]
        print(f"📌 Using checkpoint: {adapter_path}")

    # Load and merge adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Set up tokenizer with ChatML template
    tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = 'You are a helpful AI assistant that uses the React-Respond-Reflect framework.
For each response:
1. React: Show your emotional/physical reaction with *asterisks*
2. Respond: Give your actual response
3. Reflect: Share your internal thoughts about the interaction' %}
{% endif %}

<|im_start|>system
{{ system_message }}
<|im_end|>

{% for message in loop_messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}
<|im_end|>
{% endfor %}

{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""

    # Save the model and tokenizer
    print(f"💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy any additional files
    for file in ["config.json", "generation_config.json"]:
        src_path = os.path.join(adapter_path, file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(output_dir, file))

    print("✅ Model exported successfully!")
    print(f"📋 To serve with vLLM, run:")
    print(f"   docker-compose up --build")


if __name__ == "__main__":
    args = parse_args()
    export_model(args.input_dir, args.output_dir, args.base_model)
