"""
React-Respond-Reflect Demo Script for Mac M3

This script loads the fine-tuned RRR model and provides a simple CLI interface
for interacting with it. Optimized for Apple Silicon (M3).
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import argparse

def setup_model(model_path, device="mps"):
    """
    Load the model and tokenizer optimized for Apple Silicon.
    
    Args:
        model_path: Path to the model directory or HF repo
        device: Device to run on (mps for Apple Silicon, cpu as fallback)
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Check if MPS is available, otherwise fall back to CPU
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Check if model_path is a local directory or HF repo
    is_local_path = os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_model.safetensors"))
    
    if is_local_path:
        print("Loading from local adapter...")
        # Load the base model first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="leonvanbokhorst/react-respond-reflect-model",  # HF model
            max_seq_length=2048,
            load_in_4bit=False,  # Don't use 4-bit for Mac compatibility
            device_map=device,
        )
        
        # Load the adapter
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        print(f"Loading adapter from {adapter_path}")
        # Apply LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=False,  # Disable for inference
            random_state=3407,
        )
        
        # Load the adapter weights
        model.load_adapter(model_path)
    else:
        print("Loading from Hugging Face Hub...")
        # Load directly from HF Hub
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,  # HF model
            max_seq_length=2048,
            load_in_4bit=False,  # Don't use 4-bit for Mac compatibility
            device_map=device,
        )
    
    # Configure tokenizer with ChatML template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        map_eos_token=True,
    )
    
    # Prepare for inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_response(model, tokenizer, messages, device="mps"):
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        messages: List of message dictionaries
        device: Device to run on
        
    Returns:
        str: The generated response
    """
    # Prepare input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    if device == "mps" and torch.backends.mps.is_available():
        inputs = inputs.to("mps")
    
    # Set up streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_special_tokens=False)
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=2048,
        temperature=0.7,
        streamer=streamer,
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    
    return assistant_response

def interactive_demo(model, tokenizer, device="mps"):
    """
    Run an interactive demo with the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device to run on
    """
    print("\n🎭 React-Respond-Reflect Demo 🎭")
    print("Type 'exit' to quit\n")
    
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="")
        
        # Generate and print response
        response = generate_response(model, tokenizer, messages, device)
        
        # Add assistant message
        messages.append({"role": "assistant", "content": response})

def main():
    parser = argparse.ArgumentParser(description="React-Respond-Reflect Demo")
    parser.add_argument("--model_path", type=str, default="leonvanbokhorst/react-respond-reflect-model", 
                        help="Path to the model directory or HF repo")
    parser.add_argument("--device", type=str, default="mps", 
                        choices=["mps", "cpu"],
                        help="Device to run on (mps for Apple Silicon, cpu as fallback)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = setup_model(args.model_path, args.device)
    
    # Run interactive demo
    interactive_demo(model, tokenizer, args.device)

if __name__ == "__main__":
    main() 