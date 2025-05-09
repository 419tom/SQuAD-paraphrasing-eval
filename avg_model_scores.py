import json
import matplotlib.pyplot as plt

# Load prediction data
with open("squad2_predictions_with_pos_trigrams.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Separate original and variant entries
original_scores = []
variant_scores = []

for item in data:
    if item["variant_question"] == item["original_question"]:
        original_scores.append(item["score"])
    else:
        variant_scores.append(item["score"])

# Compute averages
avg_original = sum(original_scores) / len(original_scores)
avg_variant = sum(variant_scores) / len(variant_scores)

# Plot bar chart
labels = ['Original Questions', 'Variant Questions']
scores = [avg_original, avg_variant]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, scores, color=['skyblue', 'salmon'])
plt.ylabel("Average Score")
plt.title("Average Model Score: Original vs. Variant Prompts")
plt.ylim(0, 1)  # Set y-axis from 0 to 1

# Add score labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}",
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()