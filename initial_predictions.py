!pip install datasets

from transformers import pipeline
from datasets import load_dataset
import json
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define file paths for saving progress
progress_file = '/content/drive/MyDrive/squad_progress_test.json'  # Change this to your desired path
predictions_file = '/content/drive/MyDrive/squad2_predictions.json'

# Load the SQuAD2.0 dataset
dataset = load_dataset("squad_v2", split="validation")  # Use a small slice for demo

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load previous progress (if any)
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)  # Load the saved progress
    return {"last_processed_index": 0, "predictions": []}  # Default if no progress file is found

# Function to save progress
def save_progress(data):
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Load previous progress
progress = load_progress()
last_processed_index = progress["last_processed_index"]
predictions = progress["predictions"]

# Run inference, resuming from the last processed item
for i in range(last_processed_index, len(dataset)):
    item = dataset[i]
    question = item["question"]
    context = item["context"]
    gold_answer = item["answers"]["text"]

    # Get the prediction
    result = qa_pipeline(question=question, context=context)

    # Add the prediction to the list
    predictions.append({
        "id": item["id"],
        "question": question,
        "context": context,
        "predicted_answer": result.get("answer", ""),
        "gold_answer": gold_answer,
        "score": result.get("score", 0)
    })

    # Save progress after every item
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

# Output path in Google Drive
save_path = "/content/drive/MyDrive/squad_paraphrases.json"

# Load existing progress if file exists
if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        paraphrased_data = json.load(f)
    completed_ids = set(item["id"] for item in paraphrased_data)
    print(f"Resuming from progress. {len(completed_ids)} items already processed.")
else:
    paraphrased_data = []
    completed_ids = set()
    print("Starting fresh.")

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
for item in squad_data:
    if item["id"] in completed_ids:
        continue  # Skip already done

    q = item["question"]
    ps = paraphrase(q)

    paraphrased_data.append({
        "id": item["id"],
        "original_question": q,
        "paraphrases": ps
    })

    # Save progress after each item
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(paraphrased_data, f, indent=2)

print("Done! Paraphrases saved to Google Drive.")

# After completing, save final predictions
with open(predictions_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

print("Predictions saved to squad2_predictions.json")
