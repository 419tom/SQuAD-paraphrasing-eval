import json
from collections import defaultdict

# Load data
with open("squad2_predictions_with_pos_trigrams.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Group by question ID
grouped = defaultdict(list)
for entry in data:
    grouped[entry["id"]].append(entry)

# Track trigram performance
trigram_stats = defaultdict(lambda: {"improved": 0, "worsened": 0, "same": 0})

# Evaluate score deltas with progress and intermediate saving
trigram_metrics = []
question_ids = list(grouped.keys())
total = len(question_ids)

for i, qid in enumerate(question_ids, 1):
    print(f"[{i}/{total}] Processing question ID: {qid}")
    variants = grouped[qid]
    original = next((v for v in variants if v["variant_question"] == v["original_question"]), None)
    if not original:
        continue
    orig_score = original["score"]

    for v in variants:
        if v["variant_question"] == v["original_question"]:
            continue
        delta = v["score"] - orig_score
        label = "improved" if delta > 0 else "worsened" if delta < 0 else "same"
        for trigram in v.get("pos_trigrams", []):
            trigram_stats[trigram][label] += 1

    # Intermediate metric computation and saving
    if i % 100 == 0 or i == total:
        trigram_metrics = []
        for trigram, counts in trigram_stats.items():
            total_ = counts["improved"] + counts["worsened"] + counts["same"]
            if total_ == 0:
                continue
            improve_rate = counts["improved"] / total_
            worsen_rate = counts["worsened"] / total_
            net_gain = counts["improved"] - counts["worsened"]
            norm_gain = net_gain / total_
            trigram_metrics.append({
                "trigram": trigram,
                "total": total_,
                "improved": counts["improved"],
                "worsened": counts["worsened"],
                "same": counts["same"],
                "improve_rate": round(improve_rate, 3),
                "worsen_rate": round(worsen_rate, 3),
                "net_gain": net_gain,
                "normalized_gain": round(norm_gain, 3)
            })
        with open("squad_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(trigram_metrics, f, indent=2)
        print(f"✅ Saved intermediate results at question {i}")

# Sort and print final top trigrams
sorted_metrics = sorted(trigram_metrics, key=lambda x: x["normalized_gain"], reverse=True)

print("\n🔼 Top 10 Trigrams (Most Helpful):")
for m in sorted_metrics[:10]:
    print(m)

# Optionally show bottom 10
#print("\n🔽 Top 10 Trigrams (Most Harmful):")
#for m in sorted_metrics[-10:]:
#    print(m)

print("✅ Final POS trigram metrics saved to squad_test_metrics.json")
