"""
Convert generated dialogues to HuggingFace dataset format.

This script processes dialogue JSON files and converts them into a format suitable
for uploading to the HuggingFace Hub. It handles message role assignment,
generates dataset statistics, and manages the upload process.

Features:
- Automatic message role assignment
- Dataset statistics generation
- Private dataset publishing
- Progress tracking with tqdm

Example:
    $ python seed_dialogues_convert_to_hf.py

Environment Variables:
    HF_TOKEN (str): HuggingFace API token for dataset upload

Dependencies:
    - datasets
    - huggingface_hub
    - python-dotenv
    - tqdm
"""

from datasets import Dataset
from huggingface_hub import login
import json
import glob
from tqdm import tqdm
import os
from dotenv import load_dotenv
from typing import List, Dict

def convert_dialog_to_messages(dialogue_str: str) -> List[Dict[str, str]]:
    """
    Convert a dialogue string into a list of message dictionaries.
    
    Processes raw dialogue text and structures it into a list of messages with
    appropriate role assignments (user/assistant). Handles the React-Respond-Reflect
    format while preserving all tags.
    
    Args:
        dialogue_str: Raw dialogue string to convert
        
    Returns:
        List of dictionaries, each containing:
            - role: "user" or "assistant"
            - content: The message content
            
    Example:
        >>> dialog = "User: Hi!\\n\\nVirtual Human: <react>*waves*</react>..."
        >>> convert_dialog_to_messages(dialog)
        [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "<react>*waves*</react>..."}
        ]
    """
    messages = []
    # Split on double newlines to separate turns
    turns = dialogue_str.strip().split("\n\n")
    
    for turn in turns:
        turn = turn.strip()
        if turn.startswith("User: "):
            messages.append({
                "role": "user",
                "content": turn[6:].strip()
            })
        elif turn.startswith("Virtual Human:"):
            # Keep the full response with all tags
            messages.append({
                "role": "assistant",
                "content": turn[15:].strip()
            })
    return messages

def main():
    """
    Main function to convert and upload dialogues to HuggingFace.
    
    Workflow:
    1. Authenticates with HuggingFace
    2. Loads all dialogue JSON files
    3. Converts dialogues to message format
    4. Generates dataset statistics
    5. Uploads to HuggingFace Hub
    
    Environment variables used:
        HF_TOKEN: Required for HuggingFace authentication
        
    Raises:
        ValueError: If HF_TOKEN is not set
        Exception: If dataset upload fails
    """
    # Load environment variables
    load_dotenv()
    
    # Login to HuggingFace
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file!")
    login(token)
    
    # Load all dialogues
    all_conversations = []
    json_files = glob.glob("*.json")
    
    print(f"Found {len(json_files)} JSON files to process... 📁")
    
    for json_file in tqdm(json_files, desc="Processing files 🔄"):
        with open(json_file, 'r') as f:
            dialogues = json.load(f)
            for dialogue in dialogues:
                messages = convert_dialog_to_messages(dialogue["dialogue"])
                all_conversations.append({
                    "conversation_id": dialogue["conversation_id"],
                    "messages": messages,
                    "num_turns": len(messages) // 2  # Each turn is a user+assistant pair
                })
    
    print(f"\nConverted {len(all_conversations)} conversations! 🎯")
    
    # Create dataset
    dataset = Dataset.from_list(all_conversations)
    
    # Print some stats
    print("\n=== Dataset Stats ===")
    print(f"Average turns per conversation: {sum(c['num_turns'] for c in all_conversations) / len(all_conversations):.1f}")
    print(f"Max turns in a conversation: {max(c['num_turns'] for c in all_conversations)}")
    print(f"Min turns in a conversation: {min(c['num_turns'] for c in all_conversations)}")
    
    # Push to hub
    dataset_name = "leonvanbokhorst/react-respond-reflect-dialogues-v2" 
    print(f"\nPushing to HuggingFace Hub as {dataset_name}... 🚀")
    
    dataset.push_to_hub(
        dataset_name,
        private=True
    )
    
    print("\n✨ All done! Your dataset is now on the HuggingFace Hub! 🎉")

if __name__ == "__main__":
    main() 