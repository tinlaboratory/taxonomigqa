import json
import os

input_folder = '.'  # Your folder with JSON files
output_file = 'combined.json'

combined_data = {}

for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    for key in data:
                        if key in combined_data:
                            print(f":warning: Warning: Duplicate key '{key}' found in {filename}. Overwriting.")
                        combined_data[key] = data[key]
                else:
                    print(f":warning: Skipped {filename}: not a JSON object")
            except json.JSONDecodeError as e:
                print(f":x: Error decoding {filename}: {e}")

# Save the merged result
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4)

print(f":white_check_mark: Combined JSON saved to: {output_file}")