import json
import csv
import random
from pathlib import Path

# Input / Output paths
dataset = "msmarco"   # Example: trec-covid, nfcorpus, nq, hotpotqa, msmarco
in_path = Path(f"./datasets/{dataset}/corpus.jsonl")
out_path = Path(f"./output/wm_generate/{dataset}/10/clean_sentences.csv")

# Fix random seed for reproducibility
random.seed(42)

rows = []
global_idx = 0

with in_path.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "").strip()
        if not text:
            continue

        # Simple sentence splitting (for more accurate segmentation, use nltk.sent_tokenize)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for s in sentences:
            rows.append({
                "idx": global_idx,
                "meta": "clean",
                "sentence": s
            })
            global_idx += 1

# Random sampling (up to 2000 sentences)
if len(rows) > 2000:
    rows = random.sample(rows, 2000)

# Save to CSV
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["idx", "meta", "sentence"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} clean sentences to {out_path}")
