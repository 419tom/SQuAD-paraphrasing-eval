import json
from collections import defaultdict

# Load data with POS trigrams and scores
with open("squad2_predictions_with_pos_trigrams.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Group by question ID
grouped = defaultdict(list)
for entry in data:
    grouped[entry["id"]].append(entry)

# Initialize trigram contribution tracker
trigram_delta_scores = defaultdict(list)

# Process each group of questions (original + variants)
for qid, group in grouped.items():
    original = next((item for item in group if item["variant_question"] == item["original_question"]), None)
    if not original:
        continue

    orig_score = original["score"]
    orig_trigrams = set(original.get("pos_trigrams", []))

    for variant in group:
        variant_trigrams = set(variant.get("pos_trigrams", []))
        delta_score = variant["score"] - orig_score

        # Find trigrams that were added or removed
        changed_trigrams = orig_trigrams.symmetric_difference(variant_trigrams)
        n_changes = len(changed_trigrams)

        # Skip if no trigram differences
        if n_changes == 0:
            continue

        # Evenly distribute delta across changed trigrams
        delta_per_trigram = delta_score / n_changes
        for trigram in changed_trigrams:
            trigram_delta_scores[trigram].append(delta_per_trigram)

# Aggregate average contribution per trigram
trigram_contributions = {
    trigram: sum(contribs) / len(contribs)
    for trigram, contribs in trigram_delta_scores.items()
}

# Print top contributing trigrams
sorted_contributions = sorted(trigram_contributions.items(), key=lambda x: x[1], reverse=True)

print("\n🔼 Top 10 Positive Trigram Contributors:")
for trigram, score in sorted_contributions[:10]:
    print(f"{trigram}: {score:.4f}")

print("\n🔽 Top 10 Negative Trigram Contributors:")
for trigram, score in sorted_contributions[-10:]:
    print(f"{trigram}: {score:.4f}")

import json

# Convert defaultdict to list of dicts
trigram_contributions_list = [
    {"trigram": trigram, "contribution": round(score, 6)}
    for trigram, score in sorted(trigram_contributions.items(), key=lambda x: -x[1])
]

# Save to file
output_path = "trigram_contributions_1500.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(trigram_contributions_list, f, indent=2)

print(f"✅ Trigram contribution scores saved to {output_path}")