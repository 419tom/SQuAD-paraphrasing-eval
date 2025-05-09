!pip install datasets

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
squad_data = load_dataset("squad_v2", split="validation")

# Output paths in Google Drive
save_path = "/content/drive/MyDrive/squad_paraphrases.json"
progress_path = "/content/drive/MyDrive/squad_progress.json"

# Load existing progress if files exist
if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        paraphrased_data = json.load(f)
    print(f"Loaded {len(paraphrased_data)} paraphrased items.")
else:
    paraphrased_data = []
    print("Starting fresh paraphrased data.")

if os.path.exists(progress_path):
    with open(progress_path, "r", encoding="utf-8") as f:
        progress_info = json.load(f)
    start_index = progress_info.get("last_index", 0)
    print(f"Resuming from index {start_index}.")
else:
    start_index = 0
    print("Starting from index 0.")

# Load paraphrasing model
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

# Paraphrasing function
def paraphrase(question, num_return_sequences=5, num_beams=5):
    input_ids = tokenizer(
        f"paraphrase: {question}", return_tensors="pt", truncation=True, padding="longest"
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=128,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        num_beam_groups=num_beams,
        temperature=0.7,
    )

    return list(set(tokenizer.batch_decode(outputs, skip_special_tokens=True)))

# Paraphrase and save incrementally
for idx in range(start_index, len(squad_data)):
    item = squad_data[idx]
    q = item["question"]
    ps = paraphrase(q)

    paraphrased_data.append({
        "id": item["id"],
        "original_question": q,
        "paraphrases": ps
    })

    # Save paraphrases
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(paraphrased_data, f, indent=2)

    # Save progress index
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"last_index": idx + 1}, f)  # Save next index to continue

    print(f"Processed {idx + 1} / {len(squad_data)} items. (Last ID: {item['id']})")

print("Done! Paraphrases saved to Google Drive.")