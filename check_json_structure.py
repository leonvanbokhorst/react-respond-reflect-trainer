import json
import glob

# Get the first JSON file
json_files = glob.glob("*.json")
with open(json_files[0], 'r') as f:
    data = json.load(f)
    print("Sample structure:")
    print(json.dumps(data[0], indent=2))  # Print first dialog 