import json
import glob

# Get the list of JSON files
json_files = glob.glob("*.json")
if not json_files:
    print("No JSON files found!")
    exit(1)

# Open the first JSON file
with open(json_files[0], 'r') as f:
    data = json.load(f)
    print("Sample structure:")
    print(json.dumps(data[0], indent=2))  # Print first dialog 