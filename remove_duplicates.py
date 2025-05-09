import json

# Path to your JSON file
file_path = "/content/drive/MyDrive/squad_paraphrases.json"

# Load the JSON data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove duplicates by 'id', keeping the first occurrence
seen_ids = set()
unique_data = []
for entry in data:
    if entry["id"] not in seen_ids:
        unique_data.append(entry)
        seen_ids.add(entry["id"])

# Save cleaned data back to file
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, indent=2)

print(f"✅ Removed duplicates. Kept {len(unique_data)} unique entries.")