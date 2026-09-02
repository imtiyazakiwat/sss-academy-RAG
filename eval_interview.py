"""Run all 22 interview questions on both models and save the answers for review."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.TEMPERATURE = 0.0          # greedy, so the evaluation is reproducible
from rag.rag_system import RAGSystem

QS = json.load(open("knowledge_base/interview_questions.json"))
MODELS = ["fast-3b", "accurate-4b"]

s = RAGSystem(load_llm=True)
out = {}
for mid in MODELS:
    s.registry.get(mid)
    rows = []
    for item in QS:
        q = item["question"]
        t0 = time.time()
        r = s.answer(q, mode="fast", model_id=mid)
        rows.append({
            "id": item["id"], "question": q, "answer": r["answer"],
            "style": r["style"], "top": r["evidence"][0]["heading"] if r["evidence"] else "",
            "page": r["evidence"][0]["page_label"] if r["evidence"] else "",
            "ttft_ms": round(r["ttft_ms"]), "total_ms": round(r["total_ms"]),
            "words": len(r["answer"].split()),
        })
        print("  [%s] %2d %5dms %3dw %-9s %-28s %s"
              % (mid, item["id"], rows[-1]["total_ms"], rows[-1]["words"],
                 r["style"], rows[-1]["top"][:28], q[:40]))
    out[mid] = rows

with open("eval22.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nsaved -> eval22.json")
