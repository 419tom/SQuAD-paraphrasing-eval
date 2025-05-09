!pip install datasets


from transformers import pipeline
from datasets import load_dataset
import json
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# File paths on Drive
paraphrase_file = "/content/drive/MyDrive/squad_paraphrases.json"
output_file = "/content/drive/MyDrive/squad2_predictions_with_paraphrases.json"

# Load the SQuAD2 validation subset
squad_data = load_dataset("squad_v2", split="validation")

# Load paraphrases
with open(paraphrase_file, "r", encoding="utf-8") as f:
    paraphrase_data = json.load(f)

# Load any existing predictions
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    processed_ids = set((p["id"], p["variant_question"]) for p in predictions)
    print(f"Resuming from progress: {len(predictions)} predictions already made.")
else:
    predictions = []
    processed_ids = set()
    print("Starting fresh.")

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Match by ID and run predictions for each paraphrase
total_inputs = len(squad_data)
for idx, item in enumerate(squad_data):
    question_id = item["id"]
    context = item["context"]
    original_question = item["question"]
    gold_answers = item["answers"]["text"]

    entry = next((q for q in paraphrase_data if q["id"] == question_id), None)
    if entry is None:
        continue

    q_variants = [original_question] + entry["paraphrases"]

    for variant in q_variants:
        if (question_id, variant) in processed_ids:
            continue  # Skip already processed variants

        print(f"🔄 Processing ID {question_id}, variant: \"{variant[:60]}...\" [{idx+1}/{total_inputs}]")

        result = qa_pipeline(question=variant, context=context)

        predictions.append({
            "id": question_id,
            "original_question": original_question,
            "variant_question": variant,
            "predicted_answer": result.get("answer", ""),
            "score": result.get("score", 0),
            "gold_answers": gold_answers
        })

        # Save progress immediately after each prediction
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        processed_ids.add((question_id, variant))  # Mark as processed

print("✅ All predictions saved to Google Drive.")