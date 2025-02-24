import json
import os
from datetime import datetime

def save_curated_dialogs():
    # Load original dialogs to get structure
    with open('seed_dialogues.json', 'r') as f:
        dialogs = json.load(f)
        
    # Read curated dialogs
    work_dir = 'dialogs_to_curate'
    for i, dialog in enumerate(dialogs):
        dialog_file = os.path.join(work_dir, f"dialog_{i+1:03d}.txt")
        if os.path.exists(dialog_file):
            with open(dialog_file, 'r') as f:
                # Skip header lines
                lines = f.readlines()
                dialog_text = ''.join(lines[3:])  # Skip first 3 lines (title and separator)
                dialog['dialogue'] = dialog_text.strip()
                
    # Create backup
    backup_file = f'seed_dialogues_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(backup_file, 'w') as f:
        json.dump(dialogs, f, indent=2)
        
    # Save curated version
    with open('seed_dialogues_curated.json', 'w') as f:
        json.dump(dialogs, f, indent=2)
        
    print(f"\nSaved backup to: {backup_file}")
    print("Saved curated dialogs to: seed_dialogues_curated.json")

if __name__ == "__main__":
    save_curated_dialogs() 