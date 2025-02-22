"""
Generate seed dialogues for training using GPT-4-mini.

This script generates a set of seed dialogues following a specific format with
<react>, <respond>, and <reflect> tags. The generated dialogues are saved to a JSON file
for later refinement and use in training.
"""

import json
import os
import time
from typing import List, Dict, Optional
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

class DialogueGenerationError(Exception):
    """Custom exception for dialogue generation errors."""
    pass

def validate_dialogue_format(dialogue: str) -> bool:
    """
    Validate that the dialogue contains the required tags in the correct order.
    Each Virtual Human response should have all three tags in order.
    
    Args:
        dialogue: The generated dialogue string to validate
    
    Returns:
        bool: True if the dialogue format is valid, False otherwise
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
    
    Args:
        seed_examples: Example dialogues to guide the generation
        prompt_template: Template for the generation prompt
        max_tokens: Maximum tokens in the response
        temperature: Temperature for generation randomness
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        str: Generated dialogue if successful, None if all retries fail
    """
    prompt = prompt_template.format(seed_examples=seed_examples)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using the specified model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a dialogue generator that produces multi-turn conversations."
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

def main():
    """Main function to generate and save dialogues."""
    # Define seed examples
    seed_examples = (
        "Example dialogue:\n"
        "User: Hey, do you have a minute to help me with something?\n"
        "Virtual Human:\n"
        "  <react>smiles and nods</react>\n"
        "  <respond>Sure! What do you need help with?</respond>\n"
        "  <reflect>I sense she might be a bit nervous about sharing her problem.</reflect>\n"
    )

    # Define prompt template
    prompt_template = (
        "Below is an example seed dialogue:\n"
        "{seed_examples}\n\n"
        "Using the same format, generate a new multi-turn dialogue that follows this structure. "
        "Each turn should include a <react>, a <respond>, and a <reflect> section in that order, "
        "and should build upon the previous turn's context. The dialogue should be natural and realistic.\n\n"
        "New dialogue:\n"
    )

    # Configuration
    num_dialogues = 50  # Generate 50 seed conversations
    temperature = 0.85  # Slightly increased temperature for more diversity
    output_file = Path("seed_dialogues.json")
    generated_dialogues: List[Dict[str, str]] = []
    failed_generations = 0

    # Generate dialogues with progress bar
    with tqdm(total=num_dialogues, desc="Generating dialogues") as pbar:
        while len(generated_dialogues) < num_dialogues:
            dialogue = generate_seed_dialogue(
                seed_examples=seed_examples,
                prompt_template=prompt_template,
                temperature=temperature,  # Pass the temperature
                max_tokens=800  # Increased token limit for longer conversations
            )
            
            if dialogue:
                generated_dialogues.append({
                    "conversation_id": f"seed_{len(generated_dialogues)+1}",
                    "dialogue": dialogue
                })
                pbar.update(1)
            else:
                failed_generations += 1
                if failed_generations >= 3:  # Stop if too many failures
                    print("\nToo many failed generations. Stopping early.")
                    break

    # Save the generated dialogues
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_dialogues, f, ensure_ascii=False, indent=2)

    print(f"\nGeneration complete!")
    print(f"Successfully generated: {len(generated_dialogues)} dialogues")
    print(f"Failed generations: {failed_generations}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main() 