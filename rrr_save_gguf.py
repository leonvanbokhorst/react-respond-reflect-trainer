#!/usr/bin/env python3
"""Convert React-Respond-Reflect model to GGUF format using Unsloth."""

import os
from unsloth import FastLanguageModel
from peft import PeftConfig
from pathlib import Path

def convert_to_gguf(
    base_model="unsloth/mistral-7b-bnb-4bit",
    lora_path="rrr_model",  # Our trained model directory
):
    """Convert fine-tuned RRR model to GGUF format."""
    # Set up working directory
    work_dir = Path(__file__).parent.absolute()
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")
    
    # Convert paths to absolute
    lora_path = work_dir / lora_path
    output_dir = work_dir / "rrr-mistral-gguf"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n🤖 Loading base Mistral model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype="bfloat16",  # Keep bfloat16 as model was trained with it
        load_in_4bit=True,  # Keep 4-bit as trained
        device_map="auto",
    )
    
    print("\n🔄 Loading RRR LoRA weights...")
    config = PeftConfig.from_pretrained(lora_path)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Match training config
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=8,  # Match training config
    )
    model.load_adapter(lora_path, adapter_name="default")
    
    print("\n💾 Merging weights and saving to GGUF...")
    # Save in different quantization formats
    quantizations = [
        "q4_k_m",  # Good balance of size/speed
        "q5_k_m",  # Higher quality
        "q8_0",    # Highest quality but larger size
    ]
    
    for quant in quantizations:
        print(f"\n📦 Creating {quant} version...")
        try:
            output_path = output_dir / quant
            output_path.mkdir(exist_ok=True)
            model.save_pretrained_gguf(
                str(output_path),
                tokenizer,
                quantization_method=quant
            )
            # Rename for clarity
            gguf_path = output_path / "model.gguf"
            new_path = output_path / f"rrr-mistral.{quant.upper()}.gguf"
            if gguf_path.exists():
                gguf_path.rename(new_path)
            print(f"✅ Saved {quant} version to {new_path}")
        except Exception as e:
            print(f"❌ Failed to create {quant}: {str(e)}")
            continue
    
    print("\n🎉 Done! GGUF versions have been created in the rrr-mistral-gguf directory.")

if __name__ == "__main__":
    convert_to_gguf() 