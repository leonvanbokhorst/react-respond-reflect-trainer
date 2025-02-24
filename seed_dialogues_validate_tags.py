import json
import glob
import re
from tqdm import tqdm

def fix_tags(text):
    """
    Fixes all tags to ensure proper formatting:
    <react>*...*</react>
    <respond>...</respond>
    <reflect>...</reflect>
    Returns (fixed_text, was_modified)
    """
    def fix_react(match):
        content = match.group(1).strip()
        # Remove existing asterisks if they exist
        content = content.strip('*')
        # Add asterisks properly
        return f"<react>*{content}*</react>"
    
    def fix_tag(match, tag_name):
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