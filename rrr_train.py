"""
Fine-tune a model on the React-Respond-Reflect dataset using Unsloth.

This script implements:
- Multi-turn conversation handling
- React-Respond-Reflect format validation
- Optimized training configuration
- Progress monitoring and validation
"""

import multiprocessing
# Set spawn method for WSL compatibility
multiprocessing.set_start_method('spawn', force=True)

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from typing import List, Dict, Optional
import re
from tqdm import tqdm

def validate_rrr_format(text: str) -> bool:
    """
    Validate that assistant responses follow React-Respond-Reflect format.
    
    Args:
        text: The text to validate
        
    Returns:
        bool: True if format is valid
    """
    # Split into turns
    turns = text.split("<|im_start|>assistant\n")
    
    for turn in turns[1:]:  # Skip first split (user/system messages)
        # Check for all three tags in order
        react_match = re.search(r'<react>\*(.*?)\*</react>', turn)
        respond_match = re.search(r'<respond>(.*?)</respond>', turn)
        reflect_match = re.search(r'<reflect>(.*?)</reflect>', turn)
        
        if not all([react_match, respond_match, reflect_match]):
            return False
            
        # Check order
        react_pos = turn.find('<react>')
        respond_pos = turn.find('<respond>')
        reflect_pos = turn.find('<reflect>')
        
        if not (react_pos < respond_pos < reflect_pos):
            return False
    
    return True

def format_conversation(conversation: Dict) -> str:
    """
    Format a multi-turn conversation into ChatML format.
    
    Args:
        conversation: Dictionary containing messages and metadata
        
    Returns:
        str: Formatted conversation text
    """
    formatted_text = [
        "<|im_start|>system\n"
        "You are an empathetic AI assistant.\n"
        "<|im_end|>"
    ]
    
    for msg in conversation["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            formatted_text.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            formatted_text.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    return "\n".join(formatted_text)

def formatting_prompts_func(example):
    """Format dataset examples into ChatML format for training.
    
    Args:
        example: A dictionary containing conversation data with 'messages' list
        
    Returns:
        dict: Formatted text for training
    """
    # Start with system message
    formatted_text = "<|im_start|>system\nYou are an empathetic AI assistant<|im_end|>\n"
    
    # Add each message in the conversation
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        # Add message with ChatML tags
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return {
        "text": formatted_text
    }

def prepare_model_and_tokenizer(
    max_seq_length: int = 2048,  # Keep at 2048 for multi-turn conversations
    load_in_4bit: bool = False,
) -> tuple:
    """
    Initialize and prepare the model and tokenizer.
    
    Args:
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length=max_seq_length,
        #load_in_4bit=load_in_4bit,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                # Keep rank for learning capacity
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,       # Halved alpha for more conservative updates
        lora_dropout=0.05,  # Small dropout for stability
        bias="none",
        use_gradient_checkpointing="True",
        random_state=3407,
    )
    
    # Configure tokenizer with ChatML template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Using ChatML as shown in Unsloth example
        map_eos_token=True,
    )
    
    return model, tokenizer

def validate_generations(
    model: torch.nn.Module,
    tokenizer,
    validation_prompts: List[str],
    max_new_tokens: int = 2048,
) -> None:
    """
    Validate model generations follow RRR format.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        validation_prompts: List of prompts to test
        max_new_tokens: Maximum tokens to generate
    """
    print("\n=== Validation Generation Examples ===")
    
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)
    
    for prompt in validation_prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        print(f"\nPrompt: {prompt}")
        print("Generation:")
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            streamer=text_streamer,
            use_cache=True
        )
        
        # Validate format
        output_text = tokenizer.decode(outputs[0])
        is_valid = validate_rrr_format(output_text)
        print(f"\nValid RRR format: {'✅' if is_valid else '❌'}")

def main():
    print("🚀 Starting React-Respond-Reflect training pipeline...")
    
    # Load dataset
    print("📚 Loading dataset...")
    full_dataset = load_dataset("leonvanbokhorst/react-respond-reflect-dialogues-v2")
    # Split into train/validation (95/5 split for smaller validation set)
    full_dataset = full_dataset["train"].train_test_split(test_size=0.1, seed=3407)
    print(f"Dataset split into {len(full_dataset['train'])} train and {len(full_dataset['test'])} validation examples")
    
    # Prepare model and tokenizer
    print("🤖 Preparing model and tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer(
        max_seq_length=2048,  # Keep at 2048 for multi-turn conversations
        load_in_4bit=True,    # Using 4-bit quantization for memory efficiency
    )
    
    # Format datasets
    print("🔄 Formatting datasets...")
    formatted_train = full_dataset["train"].map(
        formatting_prompts_func,
        remove_columns=full_dataset["train"].column_names,
        num_proc=1  # Disable parallel mapping for WSL compatibility
    )
    formatted_valid = full_dataset["test"].map(
        formatting_prompts_func,
        remove_columns=full_dataset["test"].column_names,
        num_proc=1
    )
    
    # Configure training arguments - Optimized for 4090
    training_args = TrainingArguments(
        output_dir="rrr_model",
        num_train_epochs=5,               # Reduced epochs to prevent overfitting
        per_device_train_batch_size=16,    # Optimized for 4090
        per_device_eval_batch_size=16,     # Smaller eval batch size to prevent OOM
        gradient_accumulation_steps=1,    # Effective batch size of 16
        learning_rate=1e-4,               # Reduced learning rate for more stability
        warmup_ratio=0.1,            
        logging_steps=10,      
        weight_decay=0.01,              # Increased weight decay to combat overfitting
        save_steps=50,                    # Save checkpoints every 50 steps
        eval_steps=50,                    # Evaluate every 50 steps
        save_total_limit=3,              # Keep last 3 checkpoints
        eval_strategy="steps",           # Evaluate during training
        save_strategy="steps",           # Save during training
        load_best_model_at_end=True,     # Load best model at end of training
        bf16=True,                       # Using bfloat16 for compatibility
        optim="adamw_8bit",
        lr_scheduler_type="cosine",      
        seed=3407,
        report_to="none",               
        dataloader_num_workers=0,        # Disable multiprocessing for WSL compatibility
        group_by_length=True,   
    )
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # Unsloth needs this for token fixing
        train_dataset=formatted_train,
        eval_dataset=formatted_valid,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=1,  # Disable parallel processing for WSL compatibility
        packing=False,  # Can make training 5x faster for short sequences
        args=training_args,
    )
    
    # Train
    print("🏃 Starting training...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    model.save_pretrained("rrr_model")
    tokenizer.save_pretrained("rrr_model")
    
    # Validate generations
    print("✨ Testing model with validation prompts...")
    validation_prompts = [
        "I'm uhm... feeling really stressed about my upcoming presentation. Can you help?",
        "Hi... Tell me about a time you learned something new. Like, how did it feel?",
        "Hmm... I'm not sure if I'm making the right career choice. So... Any advice?",
        "I'm not sure if I'm making the right career choice. So... Any advice?",
    ]
    
    validate_generations(model, tokenizer, validation_prompts)
    print("🎉 Training complete!")

if __name__ == "__main__":
    main() 