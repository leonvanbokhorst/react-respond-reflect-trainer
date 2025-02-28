#!/usr/bin/env python3
"""Merge LoRA weights and export to GGUF format using Unsloth."""

import os
from unsloth import FastLanguageModel
from peft import PeftConfig
from pathlib import Path

def merge_lora_weights(
    base_model="unsloth/mi",
    lora_path="lora_model",  # Relative to work_dir
):
    """Merge LoRA weights and convert to GGUF format."""
    # Set up working directory
    work_dir = Path(__file__).parent.absolute()
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")
    
    # Convert path to absolute
    lora_path = work_dir / lora_path
    output_dir = work_dir / "phi-4-reluctance"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nLoading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype="bfloat16",  # Keep bfloat16 as model was trained with it
        load_in_4bit=True,  # Keep 4-bit as trained
        device_map="auto",
    )
    
    print("\nLoading LoRA weights...")
    config = PeftConfig.from_pretrained(lora_path)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,  
    )
    model.load_adapter(lora_path, adapter_name="default")
    
    print("\nMerging weights and saving to GGUF...")
    # Save directly to GGUF formats
    output_dir = work_dir / "phi-4-reluctance"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    quantizations = [
        "q4_k_m",  # Recommended balance of size/speed
        "q5_k_m",  # Higher quality than q4_k_m
        "q8_0",    # High resource but high quality
    ]
    
    for quant in quantizations:
        print(f"\nCreating {quant} version...")
        try:
            output_path = output_dir / quant
            output_path.mkdir(exist_ok=True)
            model.save_pretrained_gguf(
                str(output_path),
                tokenizer,
                quantization_method=quant
            )
            # Rename the output file to match Unsloth's naming
            gguf_path = output_path / "model.gguf"
            new_path = output_path / f"unsloth.{quant.upper()}.gguf"
            if gguf_path.exists():
                gguf_path.rename(new_path)
            print(f"Saved {quant} version to {new_path}")
        except Exception as e:
            print(f"Failed to create {quant}: {str(e)}")
            continue
    
    print("\nDone! GGUF versions have been created.")

if __name__ == "__main__":
    merge_lora_weights() 