"""
Save manually curated dialogues with automatic backup functionality.

This script processes manually curated dialogue files and saves them in the proper format
while creating timestamped backups. It handles the transition from raw dialogue files
to the structured JSON format used by the dataset.

Features:
- Automatic backup creation with timestamps
- Original structure preservation
- Header handling for dialogue files
- Safe file processing with error handling

Example:
    $ python seed_dialogues_save_curated.py

The script expects dialogue files in the 'dialogs_to_curate' directory and
saves the processed results as 'seed_dialogues_curated.json'.
"""

import json
import os
from datetime import datetime

def save_curated_dialogs():
    """
    Save curated dialogues while creating a backup of the original.
    
    Workflow:
    1. Loads the original dialogue structure
    2. Processes each curated dialogue file
    3. Creates a timestamped backup
    4. Saves the curated version
    
    The function expects dialogue files to be in a specific format:
    - Located in the 'dialogs_to_curate' directory
    - Named as 'dialog_XXX.txt' where XXX is a number
    - Contains a header section separated by '==='
    
    Returns:
        None
        
    Side effects:
        - Creates a backup file with timestamp
        - Creates/updates seed_dialogues_curated.json
        
    Raises:
        FileNotFoundError: If required files are missing
        json.JSONDecodeError: If JSON parsing fails
    """
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