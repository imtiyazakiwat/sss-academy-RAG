"""
Generate Cloud Training Dataset for Kaggle / Colab
Creates train_data.jsonl with full system persona and gold-standard Q&As.
"""

import json
import os
import random

SYSTEM_PROMPT = """You are an experienced Senior ETL Test Engineer in a technical interview. You have 4.2 years of experience at HCL Technologies working on the Menards retail data warehouse project. Speak naturally, authoritatively, and concisely in the 1st person. Ground your answers in real tools (Oracle, Informatica, TOAD, HP ALM, Unix) and practical SQL validation techniques (MINUS queries, duplicate checks, SCD2 history tracking, count reconciliation). Never use robotic bullet lists or markdown headers."""

def main():
    os.makedirs("kaggle_pipeline", exist_ok=True)

    with open("qa_data.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    samples = []

    # 1. Add all core gold standard Q&As
    for item in qa_data:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        })

    # 2. Add conversational variations
    for item in list(samples):
        q = item["messages"][1]["content"]
        a = item["messages"][2]["content"]
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"How would you explain {q} in an interview?"},
                {"role": "assistant", "content": a}
            ]
        })

    random.seed(42)
    random.shuffle(samples)

    out_path = "kaggle_pipeline/train_data.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(samples)} training samples for Kaggle at {out_path}")

if __name__ == "__main__":
    main()
