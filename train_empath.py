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
interactions that feel responsive while maintaining depth. Training uses
GPT-4o-mini as a reward model to evaluate response quality and provide
nuanced feedback for more human-like behavior.
"""

import re
import asyncio
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
import random
from dataclasses import dataclass
from vllm import SamplingParams

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

@dataclass
class StageEvaluation:
    """Evaluation criteria for each stage"""
    score: float
    feedback: str
    sub_scores: Dict[str, float]

class EMPATHRewardModel:
    """Local Qwen 1.5B based reward model for evaluating EMPATH responses"""
    
    def __init__(self):
        print("🤖 Initializing local reward model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=1024,  # Shorter for reward model
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.35,  # Increased from 0.3
        )
        # Initialize cache with size limit
        self.max_cache_size = 10000  # Limit cache entries
        self.cache = {
            "react": {},
            "respond": {},
            "reflect": {}
        }
        self.cache_hits = 0
        self.total_queries = 0
        
    def _prune_cache(self, stage: str):
        """Remove oldest entries if cache exceeds size limit"""
        if len(self.cache[stage]) > self.max_cache_size:
            # Remove 20% of oldest entries
            num_to_remove = int(self.max_cache_size * 0.2)
            self.cache[stage] = dict(list(self.cache[stage].items())[num_to_remove:])

    def evaluate_batch(self, stage: str, contents: List[str], batch_size: int = 4) -> List[StageEvaluation]:
        """Evaluate multiple responses in batches"""
        results = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            # Check cache first
            cached_results = []
            uncached_indices = []
            uncached_contents = []
            
            for j, content in enumerate(batch):
                if content in self.cache[stage]:
                    self.cache_hits += 1
                    cached_results.append((j, self.cache[stage][content]))
                else:
                    uncached_indices.append(j)
                    uncached_contents.append(content)
            
            self.total_queries += len(batch)
            
            if uncached_contents:
                # Create batched prompt
                batch_prompt = self.create_batch_evaluation_prompt(stage, uncached_contents)
                
                try:
                    # Generate evaluations using local model
                    sampling_params = SamplingParams(
                        temperature=0.3,
                        max_tokens=200,
                    )
                    outputs = self.model.generate(
                        batch_prompt,
                        #sampling_params=sampling_params
                    )
                    response_text = outputs[0].outputs[0].text
                    
                    # Parse batch response
                    evaluations = self.parse_batch_response(response_text, len(uncached_contents))
                    
                    # Cache new results and prune if needed
                    for content, eval in zip(uncached_contents, evaluations):
                        self.cache[stage][content] = eval
                        self._prune_cache(stage)
                    
                    # Combine cached and new results
                    batch_results = [None] * len(batch)
                    for idx, eval in cached_results:
                        batch_results[idx] = eval
                    for uncached_idx, eval in zip(uncached_indices, evaluations):
                        batch_results[uncached_idx] = eval
                    
                    results.extend(batch_results)
                except Exception as e:
                    #print(f"⚠️ Batch evaluation error: {str(e)}")
                    # Fallback: evaluate individually
                    for content in uncached_contents:
                        results.append(self.evaluate_stage(stage, content))
            else:
                # All results were cached
                batch_results = [None] * len(batch)
                for idx, eval in cached_results:
                    batch_results[idx] = eval
                results.extend(batch_results)
            
            # Log cache stats periodically
            if random.random() < 0.01:  # 1% of batches
                hit_rate = (self.cache_hits / self.total_queries) * 100 if self.total_queries > 0 else 0
                print(f"\n📊 Cache Stats:")
                print(f"Hit rate: {hit_rate:.1f}%")
                print(f"Cache size: {sum(len(c) for c in self.cache.values())} entries")
        
        return results

    def create_batch_evaluation_prompt(self, stage: str, contents: List[str]) -> str:
        """Create evaluation prompt for multiple responses"""
        base_prompt = self.create_evaluation_prompt(stage, "").split("Reaction to evaluate:")[0]
        batch_prompt = base_prompt + "\nEvaluate these responses:\n\n"
        for i, content in enumerate(contents, 1):
            batch_prompt += f"Response {i}:\n{content}\n\n"
        batch_prompt += "Provide evaluations in this format for EACH response:\n"
        batch_prompt += "RESPONSE {n}:\nSCORE: (number)\nFEEDBACK: (brief)\nSUB-SCORES:\n(criteria): (number)\n\n"
        return batch_prompt

    def parse_batch_response(self, response_text: str, expected_count: int) -> List[StageEvaluation]:
        """Parse response containing multiple evaluations"""
        evaluations = []
        sections = response_text.split("RESPONSE")[1:]  # Skip first split
        
        for section in sections[:expected_count]:  # Only process expected number
            try:
                lines = section.strip().split('\n')
                score_line = [l for l in lines if "SCORE:" in l][0]
                score = float(score_line.split(':')[1].strip())
                
                feedback_line = [l for l in lines if "FEEDBACK:" in l][0]
                feedback = feedback_line.split(':')[1].strip()
                
                sub_scores = {}
                in_sub_scores = False
                for line in lines:
                    if "SUB-SCORES:" in line:
                        in_sub_scores = True
                    elif in_sub_scores and ":" in line:
                        key, value = line.split(':')
                        sub_scores[key.strip()] = float(value.strip())
                
                evaluations.append(StageEvaluation(score, feedback, sub_scores))
            except:
                evaluations.append(StageEvaluation(0.0, "Failed to parse batch evaluation", {}))
        
        # Fill with fallback if needed
        while len(evaluations) < expected_count:
            evaluations.append(StageEvaluation(0.0, "Missing batch evaluation", {}))
        
        return evaluations

    def evaluate_stage(self, stage: str, content: str) -> StageEvaluation:
        """Evaluate a single stage using local Qwen 1.5B"""
        prompt = self.create_evaluation_prompt(stage, content)
        
        try:
            # Generate evaluation using local model
            sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=100,
            )
            outputs = self.model.generate(
                prompt,
                #sampling_params=sampling_params
            )
            response_text = outputs[0].outputs[0].text
            
            # Parse response
            try:
                # Extract overall score
                score_line = [l for l in response_text.split('\n') if l.startswith('SCORE:')][0]
                score = float(score_line.split(':')[1].strip())
                
                # Extract feedback
                feedback_line = [l for l in response_text.split('\n') if l.startswith('FEEDBACK:')][0]
                feedback = feedback_line.split(':')[1].strip()
                
                # Extract sub-scores
                sub_scores = {}
                sub_score_lines = response_text.split('SUB-SCORES:\n')[1].split('\n')
                for line in sub_score_lines:
                    if ':' in line:
                        key, value = line.split(':')
                        sub_scores[key.strip()] = float(value.strip())
                
                return StageEvaluation(score, feedback, sub_scores)
            except:
                # Fallback if parsing fails
                return StageEvaluation(0.0, "Failed to parse evaluation", {})
        except Exception as e:
            #print(f"⚠️ Evaluation error: {str(e)}")
            return StageEvaluation(0.0, f"Model error: {str(e)}", {})

    def create_evaluation_prompt(self, stage: str, content: str) -> str:
        """Create stage-specific evaluation prompt"""
        prompts = {
            "react": """You are an expert in evaluating human-like emotional reactions.
Rate this reaction (0.0-1.0) on how well it:

1. Shows authentic emotional state through:
   - Facial expressions (e.g., smiling, frowning, raised eyebrows)
   - Body language (e.g., leaning in, sitting back, posture changes)
   
2. Matches these criteria:
   - Instinctive (feels immediate and natural)
   - Clear (easy to understand the emotional signal)
   - Appropriate (fits the context)
   - Brief (quick and to the point)
   - Authentic (not performative or exaggerated)

The reaction should be in format: [expression/emotion] + [body language/posture]
Example good reactions:
- slight smile, eyes brightening + leaning forward attentively
- concerned frown, gentle eyes + supportive posture
- thoughtful expression, raised eyebrow + relaxed but engaged stance
- warm gaze, soft smile + open, welcoming posture
- curious tilt of head + alert, receptive stance

Bad examples:
- happy + sitting (too vague)
- EXTREMELY EXCITED!!! + JUMPING UP AND DOWN (too exaggerated)
- neutral face + neutral posture (too generic)
- smiling warmly while maintaining eye contact and nodding slightly + leaning forward with shoulders relaxed and hands clasped gently (too verbose)

Reaction to evaluate:
{content}

Respond in this exact format:
SCORE: (number between 0.0 and 1.0)
FEEDBACK: (brief explanation)
SUB-SCORES:
emotional_clarity: (number)
body_language: (number)
naturalness: (number)
appropriateness: (number)
authenticity: (number)""",

            "respond": """You are an expert in evaluating natural conversation.
Rate this response (0.0-1.0) on how well it demonstrates:

1. Conversational Quality:
   - Natural flow and rhythm
   - Appropriate detail level
   - Clear but casual language
   - Good use of conversational markers

2. Emotional Intelligence:
   - Shows understanding of user's state
   - Maintains appropriate emotional depth
   - Balances support and engagement
   - Authentic personality

3. Engagement:
   - Active listening signals
   - Relevant follow-up points
   - Personal but professional
   - Maintains conversation momentum

Response to evaluate:
{content}

Respond in this exact format:
SCORE: (number between 0.0 and 1.0)
FEEDBACK: (brief explanation)
SUB-SCORES:
naturalness: (number)
engagement: (number)
emotional_intelligence: (number)
clarity: (number)
personality: (number)""",

            "reflect": """You are an expert in evaluating thoughtful reflection.
Rate this reflection (0.0-1.0) on how well it shows:

1. Depth of Processing:
   - Multiple viewpoint consideration
   - Connection to broader context
   - Integration of past interactions
   - Future interaction planning

2. Emotional Understanding:
   - Recognition of emotional undercurrents
   - Appreciation of unstated needs
   - Sensitivity to social dynamics
   - Awareness of potential concerns

3. Growth Orientation:
   - Learning from interaction
   - Adapting approach as needed
   - Setting interaction goals
   - Planning improvements

Example good reflections:
- "They seem hesitant about opening up. Might need to build more trust gradually."
- "Their enthusiasm suggests a deeper interest. Could explore related topics next."
- "Sensing some underlying worry. Should maintain supportive presence while respecting boundaries."

Reflection to evaluate:
{content}

Respond in this exact format:
SCORE: (number between 0.0 and 1.0)
FEEDBACK: (brief explanation)
SUB-SCORES:
depth: (number)
emotional_insight: (number)
adaptability: (number)
usefulness: (number)
authenticity: (number)""",
        }
        return prompts[stage].format(content=content)

def extract_tags(text: str, tag: str) -> str:
    """Extract content between XML-style tags."""
    try:
        content = text.split(f"<{tag}>")[-1]
        content = content.split(f"</{tag}>")[0]
        return content.strip()
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

def format_reward_func(completions, **kwargs) -> list[float]:
    """Check if response has all required tags in right order"""
    pattern = r"<react>.*?</react>\s*<respond>.*?</respond>\s*<reflect>.*?</reflect>"
    # Handle completions as direct strings
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in completions]

def tag_count_reward_func(completions, **kwargs) -> list[float]:
    """Ensure exactly one of each tag pair"""
    scores = []
    for r in completions:  # Handle completions as direct strings
        score = 0.0
        if r.count("<react>") == 1 and r.count("</react>") == 1: score += 0.2
        if r.count("<respond>") == 1 and r.count("</respond>") == 1: score += 0.2
        if r.count("<reflect>") == 1 and r.count("</reflect>") == 1: score += 0.2
        scores.append(score)
    return scores

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
        gpu_memory_utilization = 0.45,  # Adjusted for better balance
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

    print("🎭 Initializing reward model...")
    reward_model = EMPATHRewardModel()

    def reward_func(prompts=None, completions=None, **kwargs) -> list[float]:
        """Uses local Qwen model to evaluate response quality with batching"""
        if completions is None:
            return []
        
        # Handle completions as direct strings
        responses = completions
        rewards = []
        
        try:
            # Extract stages for all responses
            react_contents = [extract_tags(r, "react") for r in responses]
            respond_contents = [extract_tags(r, "respond") for r in responses]
            reflect_contents = [extract_tags(r, "reflect") for r in responses]
            
            # Evaluate in batches
            react_evals = reward_model.evaluate_batch("react", react_contents)
            respond_evals = reward_model.evaluate_batch("respond", respond_contents)
            reflect_evals = reward_model.evaluate_batch("reflect", reflect_contents)
            
            # Calculate final scores
            for react_eval, respond_eval, reflect_eval in zip(react_evals, respond_evals, reflect_evals):
                final_score = (
                    0.3 * react_eval.score +    # Quick reaction
                    0.4 * respond_eval.score +  # Core conversation
                    0.3 * reflect_eval.score    # Deep reflection
                )
                
                # Log detailed feedback periodically
                if random.random() < 0.0:  # 5% of responses
                    print("\n🔍 Detailed Evaluation:")
                    print(f"⚡ REACT ({react_eval.score:.2f}): {react_eval.feedback}")
                    print(f"💬 RESPOND ({respond_eval.score:.2f}): {respond_eval.feedback}")
                    print(f"🤔 REFLECT ({reflect_eval.score:.2f}): {reflect_eval.feedback}")
                    print("\n📊 Sub-scores:")
                    print("React:", react_eval.sub_scores)
                    print("Respond:", respond_eval.sub_scores)
                    print("Reflect:", react_eval.sub_scores)
                
                rewards.append(final_score)
        except Exception as e:
            print(f"⚠️ Reward calculation error: {str(e)}")
            print(f"Response causing error: {responses[0][:100]}...")  # Print start of problematic response
            rewards = [0.0] * len(responses)  # Fallback
        
        return rewards

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
        max_prompt_length = 256,  # Reduced to save memory
        max_completion_length = 256,  # Reduced to save memory
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
            reward_func,  # Our main quality reward
            format_reward_func,  # Check overall format
            tag_count_reward_func,  # Check tag counts
        ],
        args = training_args,
        train_dataset = dataset,
    )

    print("🎯 Starting training...")
    trainer.train()

    print("💾 Saving model...")
    model.save_pretrained_merged(
        "empath-model",
        tokenizer,
        save_method = "merged_16bit",
    )

if __name__ == "__main__":
    main() 