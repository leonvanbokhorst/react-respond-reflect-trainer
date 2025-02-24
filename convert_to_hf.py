from datasets import Dataset
from huggingface_hub import login
import json
import glob
from tqdm import tqdm
import os
from dotenv import load_dotenv

def convert_dialog_to_messages(dialogue_str):
    """Convert a dialogue string into a list of message dictionaries"""
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