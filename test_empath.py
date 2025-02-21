"""Interactive test script for the EMPATH model."""

import torch
from unsloth import FastLanguageModel
import asyncio
from typing import List, Dict

def load_model(model_path: str = "empath-model"):
    """Load the trained EMPATH model."""
    print("🤖 Loading EMPATH model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
        fast_inference = True,
    )
    
    # Add our special tokens (needed even after training)
    special_tokens = {
        "additional_special_tokens": [
            "<react>", "</react>",
            "<respond>", "</respond>",
            "<reflect>", "</reflect>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print("✨ Special tokens loaded")
    return model, tokenizer

async def get_response(model, tokenizer, prompt: str) -> str:
    """Get a response from the model."""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = await model.fast_generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)  # Keep special tokens

async def interactive_chat():
    """Run an interactive chat session with EMPATH."""
    model, tokenizer = load_model()
    
    print("\n🎭 Welcome to EMPATH Interactive Chat! 🎭")
    print("Type 'quit' to exit\n")
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Build the prompt with conversation history and system prompt
        full_prompt = SYSTEM_PROMPT + "\n\n"  # Add the system prompt
        for turn in conversation_history[-3:]:  # Keep last 3 turns for context
            full_prompt += f"{turn}\n"
        full_prompt += f"User: {user_input}\nEMPATH:"
        
        try:
            response = await get_response(model, tokenizer, full_prompt)
            print("\nEMPATH:", response)
            
            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"EMPATH: {response}")
            
        except Exception as e:
            print(f"\n❌ Oops! Something went wrong: {str(e)}")
            print("Let's try again!")

if __name__ == "__main__":
    print("\n🚀 Starting EMPATH test environment...")
    asyncio.run(interactive_chat()) 