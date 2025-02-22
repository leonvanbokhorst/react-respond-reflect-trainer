"""Train EMPATH (Embodied Multi-modal Personality And Thought Handler) model using GRPO.

This script implements training for a virtual human model that uses a three-stage
response pattern designed around natural human interaction latencies:

1. REACT - Immediate emotional and physical reactions (microseconds)
   - Natural facial expressions and body language
   - Creates immediate emotional connection
   - Bridges the gap before verbal response

2. RESPOND - Natural conversation with expected latency (milliseconds)
   - Thoughtful verbal responses
   - Personality and style in communication
   - Matches human expectations for dialogue timing

3. REFLECT - Background reflection for future turns (seconds)
   - Deep processing of conversation
   - Prediction of user's next moves
   - Invisible to user but enriches future interactions

The model learns to balance these timescales to create natural and engaging
interactions that feel responsive while maintaining depth.
"""

from typing import List, Dict
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from reward_rules import (  # Import all our reward functions
    format_reward,
    content_reward, 
    consistency_reward,
    reward_func
)

# Patch FastRL before all functions
PatchFastRL("GRPO", FastLanguageModel)

SYSTEM_PROMPT = """You are a virtual human who connects deeply with others through three key behaviors:

1. REACT - Show immediate emotional presence through:
   - Natural facial expressions that reflect your emotional state
   - Authentic body language and posture changes
   Format: [expression/emotion] + [body language/posture]
   Examples:
   - warm smile, eyes crinkling + relaxed, open posture
   - thoughtful expression, slight head tilt + leaning forward attentively
   - concerned look, furrowed brow + gentle, supportive stance

2. RESPOND - Engage naturally in conversation with:
   - Thoughtful verbal responses
   - Authentic personality and style
   - Natural conversational timing
   - Appropriate emotional depth

3. REFLECT - Process deeply in the background:
   - Consider multiple perspectives
   - Show appropriate uncertainty
   - Think about underlying goals
   - Plan future interactions

For EVERY response, use these EXACT tags:
<react>
[expression/emotion] + [body language/posture]
</react>

<respond>
Your natural response here
</respond>

<reflect>
Your thoughtful reflection here
</reflect>"""

def extract_tags(text: str, tag: str) -> str:
    """Extract content between XML-style tags."""
    try:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    except:
        return ""

def prepare_conversation_dataset(split="train", tokenizer=None):
    """Load and prepare conversation dataset for GRPO training."""
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
        
    # Load the SmolTalk dataset
    data = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split=split)
    
    def transform_to_grpo_format(example):
        # Only process if we have at least 2 messages
        messages = example.get("messages", [])
        if len(messages) < 2:
            return None
            
        # Get the last two messages for our training pair
        user_message = messages[-2]["content"]
        assistant_message = messages[-1]["content"]
        
        # Format target response with our special tags
        target_response = f"""<react>
thoughtful expression, attentive gaze + leaning forward slightly
</react>

<respond>
{assistant_message}
</respond>

<reflect>
User seems engaged and interested. Should maintain this level of engagement while exploring the topic further.
</reflect>"""

        # Create prompt using chat template
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # Return ONLY the exact keys GRPO expects
        return {
            "prompt": prompt,  # Raw text after chat template
            "chosen": target_response,  # What the model should generate
            "rejected": None  # Required by GRPO
        }
    
    # Transform dataset with explicit key removal
    print("🔄 Transforming dataset...")
    data = data.map(
        transform_to_grpo_format,
        remove_columns=data.column_names  # Explicitly remove all original columns
    )
    
    # Filter out empty conversations
    print("🧹 Filtering empty conversations...")
    data = data.filter(lambda x: x is not None)
    
    # Verify dataset format
    print("\n🔍 Verifying dataset format...")
    example = data[0]
    expected_keys = {"prompt", "chosen", "rejected"}
    actual_keys = set(example.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"Dataset has wrong keys! Expected {expected_keys}, got {actual_keys}")
    
    # Log some examples
    print("\n📝 Example conversation format:")
    print("Keys:", list(example.keys()))
    print("Prompt:", example["prompt"][:100] + "...")
    print("Chosen:", example["chosen"][:100] + "...")
    
    return data

def main():
    # Adjust memory allocation for two models
    max_seq_length = 2048  # For main model
    lora_rank = 64  # Reduced for memory
    
    print("🤖 Loading Qwen 7B model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-7B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = True,
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7,  # Adjusted for better balance
    )

    # Add our special tokens
    special_tokens = {
        "additional_special_tokens": [
            "<react>", "</react>",
            "<respond>", "</respond>",
            "<reflect>", "</reflect>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"✨ Added {len(special_tokens['additional_special_tokens'])} special tokens")

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("📚 Loading and preparing dataset...")
    dataset = prepare_conversation_dataset("train", tokenizer=tokenizer)

    print("⚙️ Configuring training...")
    training_args = GRPOConfig(
        use_vllm = True,
        learning_rate = 5e-6,  # Increased slightly for better exploration
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,  # Reduced to match docs
        gradient_accumulation_steps = 4,  # Increased for stability
        num_generations = 8,  # Keep 8 for good exploration
        max_prompt_length = 512,  # Reduced to save memory
        max_completion_length = 1024,  # Plenty of room for full responses
        max_steps = 2000,  # Increased for better learning
        save_steps = 500,  # Save less frequently
        max_grad_norm = 0.1,
        output_dir = "outputs",
    )

    print("🚀 Setting up trainer...")
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward,     # Structure and tag ordering
            content_reward,    # Quality within sections
            consistency_reward # Flow between sections
        ],
        args = training_args,
        train_dataset = dataset,
    )

    print("🎯 Resuming training from checkpoint...")
    trainer.train(resume_from_checkpoint="outputs/checkpoint-500")

    print("💾 Saving model...")
    model.save_pretrained_merged(
        "empath-model",
        tokenizer,
        save_method = "merged_16bit",
    )

if __name__ == "__main__":
    main() 