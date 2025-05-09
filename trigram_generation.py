import json
import spacy

# Load English tokenizer, POS tagger, etc.
nlp = spacy.load("en_core_web_sm")

# Function to extract POS trigrams
def get_pos_trigrams(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return [' '.join(pos_tags[i:i+3]) for i in range(len(pos_tags) - 2)]

# Load predictions
with open("/content/drive/MyDrive/squad2_predictions_with_paraphrases.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Add only POS trigrams with progress
output_path = "squad2_predictions_with_pos_trigrams.json"
total = len(predictions)

for i, item in enumerate(predictions, 1):
    question = item["variant_question"]
    item["pos_trigrams"] = get_pos_trigrams(question)

    # Save intermediate progress every 100 entries or at the end
    if i % 100 == 0 or i == total:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        print(f"[{i}/{total}] ✅ Progress saved to {output_path}")

print("🎉 POS trigram annotations completed and saved.")