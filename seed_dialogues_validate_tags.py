"""
Validate and fix XML-style tags in dialogue responses.

This script processes dialogue files to ensure proper formatting of React-Respond-Reflect
tags. It validates tag presence, order, and format, fixing common issues automatically.

Features:
- Tag order verification (<react> -> <respond> -> <reflect>)
- Format standardization (especially for react tags with asterisks)
- Detailed fix reporting
- Batch processing with progress tracking

Example:
    $ python seed_dialogues_validate_tags.py

The script processes all JSON files in the current directory containing dialogues.
"""

import json
import glob
import re
from tqdm import tqdm

def fix_tags(text: str) -> tuple[str, bool]:
    """
    Fix all tags to ensure proper formatting.
    
    Processes dialogue text to standardize tag format:
    - <react>*...*</react>
    - <respond>...</respond>
    - <reflect>...</reflect>
    
    Args:
        text: The dialogue text to process
        
    Returns:
        tuple containing:
            - fixed_text: The processed text with standardized tags
            - was_modified: Boolean indicating if any changes were made
            
    Example:
        >>> text = "Virtual Human: <react>smiles</react>\\n<respond>Hi!</respond>"
        >>> fixed, modified = fix_tags(text)
        >>> print(fixed)
        "Virtual Human: <react>*smiles*</react>\\n<respond>Hi!</respond>"
    """
    def fix_react(match):
        """Helper function to fix react tag content."""
        content = match.group(1).strip()
        # Remove existing asterisks if they exist
        content = content.strip('*')
        # Add asterisks properly
        return f"<react>*{content}*</react>"
    
    def fix_tag(match, tag_name):
        """Helper function to fix respond/reflect tag content."""
        content = match.group(1).strip()
        return f"<{tag_name}>{content}</{tag_name}>"
    
    original = text
    
    # Fix react tags (should have asterisks)
    text = re.sub(r'<react>(.*?)</react>', fix_react, text)
    
    # Fix respond tags (no asterisks)
    text = re.sub(r'<respond>(.*?)</respond>', lambda m: fix_tag(m, "respond"), text)
    
    # Fix reflect tags (no asterisks)
    text = re.sub(r'<reflect>(.*?)</reflect>', lambda m: fix_tag(m, "reflect"), text)
    
    # Check if any tags are missing or in wrong order
    turns = text.split("\n\n")
    fixed_turns = []
    
    for turn in turns:
        if turn.startswith("Virtual Human:"):
            # Ensure all three tags are present and in correct order
            parts = turn.split("\n")
            if len(parts) >= 2:  # Has content after "Virtual Human:"
                response_parts = []
                response_parts.append(parts[0])  # "Virtual Human:" line
                
                # Extract existing tags content or use placeholders
                react_match = re.search(r'<react>\*(.*?)\*</react>', turn)
                respond_match = re.search(r'<respond>(.*?)</respond>', turn)
                reflect_match = re.search(r'<reflect>(.*?)</reflect>', turn)
                
                react_content = f"*{react_match.group(1)}*" if react_match else "*neutral stance*"
                respond_content = respond_match.group(1) if respond_match else "Generic response"
                reflect_content = reflect_match.group(1) if reflect_match else "Basic reflection"
                
                response_parts.extend([
                    f"  <react>{react_content}</react>",
                    f"  <respond>{respond_content}</respond>",
                    f"  <reflect>{reflect_content}</reflect>"
                ])
                turn = "\n".join(response_parts)
        fixed_turns.append(turn)
    
    text = "\n\n".join(fixed_turns)
    return text, text != original

def main():
    """
    Main function to process and fix dialogue files.
    
    Workflow:
    1. Finds all JSON files in current directory
    2. Processes each file's dialogues
    3. Fixes tag formatting issues
    4. Saves modified files
    5. Reports statistics on fixes made
    
    The script maintains the original file structure while fixing:
    - Missing tags
    - Improper tag order
    - Inconsistent formatting
    - Missing asterisks in react tags
    """
    json_files = glob.glob("*.json")
    print(f"Found {len(json_files)} JSON files to process... 🔍")
    
    total_fixes = 0
    files_modified = set()
    
    for json_file in tqdm(json_files, desc="Processing files 🕵️"):
        file_modified = False
        with open(json_file, 'r') as f:
            dialogues = json.load(f)
            
        for dialogue in dialogues:
            fixed_dialogue, was_modified = fix_tags(dialogue["dialogue"])
            if was_modified:
                total_fixes += 1
                files_modified.add(json_file)
                dialogue["dialogue"] = fixed_dialogue
                file_modified = True
                print(f"\n🔧 Fixed tags in conversation {dialogue['conversation_id']}")
        
        # Save changes if file was modified
        if file_modified:
            with open(json_file, 'w') as f:
                json.dump(dialogues, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Total conversations fixed: {total_fixes}")
    print(f"Files modified: {len(files_modified)}")
    
    if files_modified:
        print("\nModified files:")
        for file in files_modified:
            print(f"  - {file}")
        print("\n✨ All tags have been fixed and standardized! 🎉")
    else:
        print("\n✨ All tags were already properly formatted! 🎉")

if __name__ == "__main__":
    main() 