import json
from collections import defaultdict

# Load predictions
with open("/content/drive/MyDrive/squad2_predictions_with_paraphrases.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Organize predictions by question ID
grouped = defaultdict(list)
for pred in predictions:
    grouped[pred["id"]].append(pred)

# Helper: check if prediction matches any gold answer (optional, for future accuracy evals)
def is_correct(predicted, gold_list):
    predicted = predicted.strip().lower()
    return any(g.strip().lower() == predicted for g in gold_list if g)

# Analyze score differences
evaluation = []
all_ids = list(grouped.keys())

for i, qid in enumerate(all_ids, 1):
    preds = grouped[qid]
    try:
        original = next(p for p in preds if p["variant_question"] == p["original_question"])
    except StopIteration:
        print(f"[{i}/{len(all_ids)}] Skipping ID {qid}: original question not found.")
        continue

    original_score = original["score"]

    for variant in preds:
        if variant["variant_question"] == variant["original_question"]:
            continue  # Skip original

        delta = variant["score"] - original_score
        status = "improved" if delta > 0 else "worsened" if delta < 0 else "same"

        evaluation.append({
            "id": qid,
            "original_question": original["original_question"],
            "variant_question": variant["variant_question"],
            "score_diff": delta,
            "status": status,
            "original_score": original_score,
            "variant_score": variant["score"],
            "predicted_answer": variant["predicted_answer"]
        })

    # Save intermediate progress every 25 questions
    if i % 25 == 0 or i == len(all_ids):
        with open("paraphrase_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)
        print(f"[{i}/{len(all_ids)}] ✅ Progress saved at ID {qid}")

print("🎉 Evaluation completed and fully saved to paraphrase_evaluation.json")