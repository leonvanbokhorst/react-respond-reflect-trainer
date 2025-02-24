"""
Generate seed dialogues for training using GPT-4-mini.

This script generates a set of seed dialogues following the React-Respond-Reflect format,
using curated examples as seeds. Each dialogue demonstrates natural conversation flow
with structured responses containing <react>, <respond>, and <reflect> tags.

The script implements:
- Batch-based generation with progress tracking
- Automatic validation of dialogue structure
- Temperature-based randomization for response variety
- Configurable batch sizes and total dialogue count
- Automatic backup and versioning

Example:
    $ python seed_dialogues_generate_dataset.py

Environment Variables:
    OPENAI_API_KEY (str): OpenAI API key for GPT-4-mini access

Dependencies:
    - openai
    - python-dotenv
    - tqdm
    - pathlib
"""

import json
import os
import time
import random
from typing import List, Dict, Optional, Set
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

def load_curated_dialogs(dialog_dir: str = "curated_seed_dialogs") -> List[str]:
    """
    Load all curated dialog examples from the specified directory.
    
    Each dialog file should contain a header section followed by the actual dialog,
    separated by a line of equal signs.
    
    Args:
        dialog_dir: Directory containing the dialog txt files
        
    Returns:
        List of dialog strings with headers removed
        
    Raises:
        FileNotFoundError: If dialog_dir doesn't exist
        ValueError: If dialog files are improperly formatted
    """
    dialogs = []
    dialog_path = Path(dialog_dir)
    
    for file in dialog_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            # Skip the header lines
            content = f.read()
            dialog_content = content.split("==================================================\n")[1].strip()
            dialogs.append(dialog_content)
            
    return dialogs

def get_random_example(
    curated_dialogs: List[str],
    used_examples: Set[int]
) -> Optional[str]:
    """
    Get a random dialog example that hasn't been used yet.
    
    Maintains a set of used example indices to ensure variety in the generated
    dialogues. Resets the used examples set when all examples have been used.
    
    Args:
        curated_dialogs: List of all curated dialogs
        used_examples: Set of indices of already used examples
        
    Returns:
        A dialog string or None if all examples have been used
    """
    available_indices = set(range(len(curated_dialogs))) - used_examples
    if not available_indices:
        return None
        
    chosen_idx = random.choice(list(available_indices))
    used_examples.add(chosen_idx)
    return curated_dialogs[chosen_idx]

class DialogueGenerationError(Exception):
    """Custom exception for dialogue generation errors."""
    pass

def validate_dialogue_format(dialogue: str) -> bool:
    """
    Validate that the dialogue contains the required tags in the correct order.
    
    Each Virtual Human response should have all three tags (<react>, <respond>, <reflect>)
    in the correct sequence. The function checks both presence and order of tags.
    
    Args:
        dialogue: The generated dialogue string to validate
    
    Returns:
        bool: True if the dialogue format is valid, False otherwise
        
    Example valid format:
        User: Hello!
        Virtual Human: <react>*smiles warmly*</react>
        <respond>Hi there! How can I help?</respond>
        <reflect>User seems friendly and open.</reflect>
    """
    # Split into turns
    turns = dialogue.split("Virtual Human:")
    
    # Skip the first part (user's initial message)
    for turn in turns[1:]:
        if not turn.strip():
            continue
            
        # Check for required tags
        if not all(tag in turn for tag in ['<react>', '<respond>', '<reflect>']):
            missing_tags = [tag for tag in ['<react>', '<respond>', '<reflect>'] if tag not in turn]
            print(f"Missing tags in Virtual Human response: {missing_tags}")
            print(f"Problem response:\n{turn}\n")
            return False
            
        # Check order of tags
        react_pos = turn.find('<react>')
        respond_pos = turn.find('<respond>')
        reflect_pos = turn.find('<reflect>')
        
        if not (react_pos < respond_pos < reflect_pos):
            print(f"Tags in wrong order. Expected: react -> respond -> reflect")
            print(f"Found positions: react={react_pos}, respond={respond_pos}, reflect={reflect_pos}")
            print(f"Problem response:\n{turn}\n")
            return False
    
    return True

def generate_seed_dialogue(
    seed_examples: str,
    prompt_template: str,
    max_tokens: int = 500,
    temperature: float = 0.8,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[str]:
    """
    Generate a single seed dialogue using GPT-4-mini with retry logic.
    
    Uses example dialogues and a template to guide the generation process.
    Implements retry logic with exponential backoff for reliability.
    
    Args:
        seed_examples: Example dialogues to guide the generation
        prompt_template: Template for the generation prompt
        max_tokens: Maximum tokens in the response
        temperature: Temperature for generation randomness
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        str: Generated dialogue if successful, None if all retries fail
        
    Raises:
        DialogueGenerationError: If generation fails after all retries
    """
    prompt = prompt_template.format(seed_examples=seed_examples)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using the specified model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a dialogue generator that produces high quality multi-turn human-like conversations."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            dialogue = response.choices[0].message.content.strip()
            print(f"\nGenerated dialogue (attempt {attempt + 1}):\n{dialogue}\n")
            
            # Validate the generated dialogue
            if validate_dialogue_format(dialogue):
                return dialogue
            else:
                print(f"Generated dialogue failed validation (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            print(f"Error during generation (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return None

def get_next_batch_number() -> int:
    """
    Find the next batch number by checking existing files.
    
    Scans the current directory for files matching the pattern
    'seed_dialogues_batch_*.json' and returns the next available
    batch number.
    
    Returns:
        int: Next available batch number
    """
    existing_batches = list(Path(".").glob("seed_dialogues_batch_*.json"))
    if not existing_batches:
        return 1
    
    # Extract batch numbers from filenames
    batch_numbers = [int(file.stem.split("_")[-1]) for file in existing_batches]
    return max(batch_numbers) + 1

def main():
    """
    Main function to generate and save dialogues.
    
    Implements the core dialogue generation workflow:
    1. Loads curated examples
    2. Generates dialogues in batches
    3. Validates dialogue format
    4. Saves results with automatic versioning
    5. Tracks progress and handles failures
    
    Environment variables used:
        OPENAI_API_KEY: Required for GPT-4-mini access
    """
    # Load curated dialogs
    curated_dialogs = load_curated_dialogs()
    used_examples = set()  # Track which examples we've used
    batch_size = 10
    total_dialogues = 500
    temperature = 0.85

    # Find next batch number and calculate remaining dialogues
    batch_number = get_next_batch_number()
    dialogues_generated = (batch_number - 1) * batch_size
    dialogues_remaining = total_dialogues - dialogues_generated

    print(f"Continuing from batch {batch_number}, {dialogues_remaining} dialogues remaining")

    # Define prompt template with more guidance
    prompt_template = (
        "Below is an example of a natural dialogue that shows emotional depth and gradual trust-building:\n"
        "{seed_examples}\n\n"
        "Using the same format, generate a new dialogue that:\n"
        "- Shows natural progression from hesitation to openness\n"
        "- Balances emotional support with practical help\n"
        "- Includes <react>, <respond>, and <reflect> tags in that order\n\n"
        "New dialogue:\n"
    )

    generated_dialogues: List[Dict[str, str]] = []
    failed_generations = 0

    # Generate dialogues with progress bar
    with tqdm(total=dialogues_remaining, desc="Generating dialogues") as pbar:
        while len(generated_dialogues) < dialogues_remaining:
            # Reset used examples when we've used them all
            if len(used_examples) >= len(curated_dialogs):
                print("\nResetting example pool for next batch...")
                used_examples.clear()
            
            # Get a random example from our curated set
            seed_example = get_random_example(curated_dialogs, used_examples)
            
            dialogue = generate_seed_dialogue(
                seed_examples=seed_example,
                prompt_template=prompt_template,
                temperature=random.uniform(temperature - 0.1, temperature + 0.1),
                max_tokens=2048
            )
            
            if dialogue:
                generated_dialogues.append({
                    "conversation_id": f"seed_{len(generated_dialogues)+1}",
                    "dialogue": dialogue
                })
                pbar.update(1)
                
                # Save every 10 dialogues
                if len(generated_dialogues) % batch_size == 0:
                    batch_output_file = Path(f"seed_dialogues_batch_{batch_number}.json")
                    current_batch = generated_dialogues[-batch_size:]  # Get last 10 dialogues
                    
                    with open(batch_output_file, "w", encoding="utf-8") as f:
                        json.dump(current_batch, f, ensure_ascii=False, indent=2)
                    
                    print(f"\nSaved batch {batch_number} to: {batch_output_file}")
                    batch_number += 1
            else:
                failed_generations += 1
                if failed_generations >= 3:
                    print("\nToo many failed generations. Stopping early.")
                    break

    # Save any remaining dialogues
    if len(generated_dialogues) % batch_size != 0:
        remaining = len(generated_dialogues) % batch_size
        batch_output_file = Path(f"seed_dialogues_batch_{batch_number}.json")
        final_batch = generated_dialogues[-remaining:]  # Get remaining dialogues
        
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(final_batch, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved final batch to: {batch_output_file}")

    print(f"\nGeneration complete!")
    print(f"Successfully generated: {len(generated_dialogues)} dialogues")
    print(f"Failed generations: {failed_generations}")
    print(f"Total batches saved: {batch_number}")

if __name__ == "__main__":
    main() 