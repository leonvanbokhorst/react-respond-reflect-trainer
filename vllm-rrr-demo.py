"""
VLLM Deployment for React-Respond-Reflect Model

This script demonstrates how to:
1. Export a fine-tuned RRR model from Unsloth to a VLLM-compatible format
2. Set up a VLLM inference server with the RRR model
3. Run inference with proper RRR formatting and parsing

Requirements:
pip install vllm unsloth peft transformers torch
"""

import os
import argparse
import time
import re
from typing import List, Dict, Optional, Union, Tuple

# Model export and merging
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# VLLM imports
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="VLLM Demo for React-Respond-Reflect Model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="rrr_model",
        help="Path to the trained RRR model"
    )
    parser.add_argument(
        "--merged_path", 
        type=str, 
        default="rrr_model_merged",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["export", "inference", "server"], 
        default="inference",
        help="Mode to run: export, inference, or server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for serving the model API"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature for generation"
    )
    return parser.parse_args()


def export_model(model_path: str, save_path: str) -> None:
    """
    Export and merge LoRA weights into the base model for VLLM compatibility.
    
    Args:
        model_path: Path to the trained LoRA model
        save_path: Path to save the merged model
    """
    print(f"🔄 Loading model from {model_path}")
    
    # Load base model and LoRA adapter
    base_model_id = "mistralai/Mistral-7B-v0.1"  # Base model used for Unsloth training
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the base model in fp16 for faster loading
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the LoRA model
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    
    print("🔄 Merging LoRA weights with base model")
    # Merge LoRA weights with base model
    merged_model = peft_model.merge_and_unload()
    
    # Save the merged model
    print(f"💾 Saving merged model to {save_path}")
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,  # Save in safetensors format
    )
    tokenizer.save_pretrained(save_path)
    
    print("✅ Model export complete!")
    print(f"📊 Model size: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B parameters")
    print(f"🔍 Model saved to: {save_path}")


def extract_rrr_sections(text: str) -> Dict[str, str]:
    """
    Extract React, Respond, and Reflect sections from generated text.
    
    Args:
        text: The generated text
        
    Returns:
        dict: Dictionary with 'react', 'respond', and 'reflect' keys
    """
    sections = {}
    
    # Extract each section with regex
    react_match = re.search(r'<react>\s*(.*?)\s*</react>', text, re.DOTALL)
    respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', text, re.DOTALL)
    reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', text, re.DOTALL)
    
    # Add matches to sections dict
    if react_match:
        sections['react'] = react_match.group(1).strip()
    else:
        sections['react'] = ""
        
    if respond_match:
        sections['respond'] = respond_match.group(1).strip()
    else:
        sections['respond'] = ""
        
    if reflect_match:
        sections['reflect'] = reflect_match.group(1).strip()
    else:
        sections['reflect'] = ""
    
    return sections


def format_chatml_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format messages in ChatML format for VLLM.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        list: Formatted messages for VLLM
    """
    formatted_messages = []
    
    # Add system message if not present
    if not messages or messages[0]['role'] != 'system':
        formatted_messages.append({
            'role': 'system',
            'content': 'You are an empathetic AI assistant. Always structure your response with: '
                      '<react>*your internal thought process*</react> first, followed by '
                      '<respond>your direct response to the user</respond>, and finally '
                      '<reflect>your reflection on this interaction</reflect>.'
        })
    
    # Add user messages
    formatted_messages.extend(messages)
    
    return formatted_messages


def run_inference(args):
    """Run inference with the merged model using VLLM"""
    print(f"🚀 Loading model from {args.merged_path} with VLLM")
    
    # Initialize VLLM
    llm = LLM(
        model=args.merged_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",  # Use float16 for most models
        trust_remote_code=True,
    )
    
    # Test prompts
    test_prompts = [
        "I'm feeling really stressed about my upcoming presentation. Can you help?",
        "Tell me about a time you learned something new. How did it feel?",
        "I'm not sure if I'm making the right career choice. Any advice?",
    ]
    
    # Sample parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["<|im_start|>", "<|im_end|>"],
    )
    
    # Run inference
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Test Prompt {i+1}: {prompt}")
        
        # Format messages for ChatML
        messages = format_chatml_prompt([{'role': 'user', 'content': prompt}])
        
        # Time the generation
        start_time = time.time()
        
        # Generate with VLLM
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        
        # Get the generated text
        response = outputs[0].messages[-1]['content']
        
        # Extract RRR sections
        sections = extract_rrr_sections(response)
        
        # Display timing and results
        elapsed = time.time() - start_time
        print(f"⏱️  Generation time: {elapsed:.2f} seconds")
        
        # Print sections
        print("\n🧠 React:")
        print(sections['react'])
        print("\n💬 Respond:")
        print(sections['respond'])
        print("\n🔄 Reflect:")
        print(sections['reflect'])
        print("\n" + "-"*50)


def start_server(args):
    """Start a VLLM API server"""
    print(f"🚀 Starting VLLM API server on port {args.port}")
    print(f"📦 Loading model from {args.merged_path}")
    
    # Create and start the OpenAI-compatible server
    server = OpenAIServingChat(
        model=args.merged_path,
        tensor_parallel_size=args.tensor_parallel_size,
        host="0.0.0.0",
        port=args.port,
        chat_template="chatml",  # Use ChatML template
        dtype="float16",
    )
    
    # Start the server
    server.serve()
    

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.mode == "export":
        export_model(args.model_path, args.merged_path)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "server":
        start_server(args)


if __name__ == "__main__":
    main()
