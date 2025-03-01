#!/usr/bin/env python
"""
Merge a LoRA adapter with a base model for deployment.
This script merges a LoRA adapter with a base model and saves the result.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to the base model or model ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./rrr_model_merged",
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype for the merged model",
    )
    return parser.parse_args()

def get_torch_dtype(dtype_str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]

def main():
    """Main function to merge the model."""
    args = parse_args()
    
    print(f"Loading base model from {args.base_model_path}...")
    torch_dtype = get_torch_dtype(args.torch_dtype)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    # Load PEFT model
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )
    
    print("Merging adapter with base model...")
    # Merge LoRA adapter with base model
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Saving merged model to {args.output_path}...")
    # Save merged model
    model.save_pretrained(
        args.output_path,
        safe_serialization=True,
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_path)
    
    print("Model successfully merged and saved!")

if __name__ == "__main__":
    main() 