"""
Regression + latency benchmark for the ETL interview RAG system.

Measures what the product actually promises:
  * time to first token (what the student perceives as responsiveness)
  * retrieval correctness (did the right section come back)
  * completeness (for multi-part topics, did every sub-point appear)
  * leakage (did the answer talk about context/notes/availability)

Run:  venv/bin/python benchmark.py                      # full suite
      venv/bin/python benchmark.py --quick              # 8 questions
      venv/bin/python benchmark.py --model <mlx-repo>   # A/B a different model
      venv/bin/python benchmark.py --json out.json      # save raw results
"""

import argparse
import json
import sys
import os
import re
import resource
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

import config
from rag.rag_system import RAGSystem


def rss_gb():
    """Resident set size of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

# (question, category, expected_page_substring_or_None, must_include_terms)
SUITE = [
    # --- direct definition questions -----------------------------------
    ("What are joins?", "direct", "35", ["inner", "non-equi", "cross", "left outer", "right outer", "full outer", "self join"]),
    # The notes define SCD2 as "maintain current and all or complete historical
    # data ... achieved by inserting as new row" (pages 7-8). They never use
    # "version" for the definition; that word only appears in the SCD2
    # validation checklist on pages 49-51. Asserting it was a bad test.
    ("What is SCD Type 2?", "direct", None, ["historical", "new row"]),
    ("Difference between TRUNCATE and DELETE", "direct", None, ["ddl", "dml", "where", "roll"]),
    ("What is a fact table?", "direct", None, ["measure"]),
    ("What is a dimension table?", "direct", None, ["de-normalized"]),
    ("What is normalization?", "direct", None, ["1nf", "2nf", "3nf"]),
    ("What is a surrogate key?", "direct", None, ["generated"]),
    ("What is a primary key?", "direct", None, ["null"]),
    ("Explain the defect life cycle", "direct", None, ["new", "open", "closed"]),
    ("What is star schema?", "direct", None, ["fact"]),
    ("What are the types of fact table?", "direct", None, []),
    ("What are constraints in SQL?", "direct", None, ["not null"]),
    ("What is a view?", "direct", None, ["logical"]),
    ("What are set operators?", "direct", None, ["union", "minus", "intersect"]),
    ("What are analytical functions?", "direct", None, ["rank", "lead", "lag"]),
    ("What is agile methodology?", "direct", None, ["sprint"]),
    ("Difference between severity and priority", "direct", None, []),
    ("What are the types of dimension table?", "direct", None, ["scd"]),
    ("What is NVL and NVL2?", "direct", None, ["nvl2"]),
    ("What are unix commands you used?", "direct", None, []),

    # --- indirect / reworded (must still hit the right notes) ----------
    ("how do i combine two tables in sql", "indirect", "35", ["join"]),
    ("how to keep old and new address history of a customer", "indirect", None, ["scd"]),
    ("which command removes all rows but keeps the table", "indirect", None, ["truncate"]),
    ("how to get rid of repeated rows", "indirect", None, ["duplicate"]),
    ("how do i get second largest salary", "indirect", None, ["max"]),
    ("what breaks a table into smaller tables to avoid repeating data", "indirect", None, ["normal"]),
    ("auto generated id column in warehouse", "indirect", None, ["surrogate"]),
    ("how to replace a null value in a column", "indirect", None, ["nvl"]),
    ("what do you call a bug's journey from raised to closed", "indirect", None, ["defect"]),
    ("who runs the daily standup", "indirect", None, ["scrum"]),

    # --- scenario / applied --------------------------------------------
    ("What happens when a customer changes their address in the dimension table?", "scenario", None, ["scd"]),
    ("If the source has 100 records and the target has 98, how would you find the missing ones?", "scenario", None, []),
    ("Suppose duplicate records loaded into target, how do you validate?", "scenario", None, []),
    ("How would you test an incremental load?", "scenario", None, ["incremental"]),
    ("The interviewer asks how you validated SCD Type 2 in your project. What do you say?", "scenario", None, []),
    ("How do you handle a defect that the developer rejects?", "scenario", None, []),

    # --- out of scope (must still answer, in the notes' voice) ---------
    ("What is Apache Airflow?", "out-of-scope", None, []),
    ("Explain database indexing and when to use it", "out-of-scope", None, []),
    ("What is the difference between a data lake and a data warehouse?", "out-of-scope", None, []),
    ("How does Kafka work?", "out-of-scope", None, []),
]

# Phrases that mean the model leaked its plumbing to the student.
LEAKS = re.compile(
    r"(based on|according to|as per|from) the (provided |given |above )?"
    r"(context|notes|document|text|knowledge|information)"
    r"|in the (provided|given) (context|notes|text)"
    r"|the (context|notes|text) (does not|doesn't|do not)"
    r"|not (available|mentioned|provided|found) in the"
    # Table cells were commenting on the material itself ("Not explicitly
    # documented"), which the pattern above missed because it has no "in the".
    r"|not (explicitly |specifically )?(documented|mentioned|specified|stated|provided|available|listed|included|found)\b"
    r"|as (documented|stated|mentioned) in the notes"
    r"|knowledge base"
    r"|i (don't|do not) have (enough |any )?(information|context)"
    r"|KNOWLEDGE:",
    re.I,
)


def stream_answer(system, question, mode="fast"):
    """Run the real streaming path.
    Returns (answer, ttft_ms, total_ms, retrieved, routed)."""
    answer, ttft, total, retrieved, routed, _ = stream_answer_full(
        system, question, mode
    )
    return answer, ttft, total, retrieved, routed


def stream_answer_full(system, question, mode="fast"):
    """As stream_answer, plus decode throughput in tokens/sec."""
    t0 = time.time()
    retrieved, _ = system.retriever.retrieve(question)
    retrieve_ms = (time.time() - t0) * 1000
    routed = system.generator.route(question, retrieved)

    pieces, ttft, first_tok_at = [], None, None
    for tok, first in system.llm.generate(
        question, retrieved, stream=True, style=routed["style"], mode=mode
    ):
        if tok == "":
            break
        if ttft is None:
            ttft = retrieve_ms + first
            first_tok_at = time.time()
        pieces.append(tok)
    total_ms = (time.time() - t0) * 1000

    # Throughput measured over the decode phase only (after the first token),
    # so it reflects generation speed rather than prefill.
    decode_s = (time.time() - first_tok_at) if first_tok_at else 0.0
    tps = (len(pieces) - 1) / decode_s if decode_s > 0.05 and len(pieces) > 1 else 0.0
    return "".join(pieces), (ttft or total_ms), total_ms, retrieved, routed, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run only 8 questions")
    ap.add_argument("--model", default=None, help="override the MLX model repo")
    ap.add_argument("--draft", default=None, help="speculative-decoding draft model")
    ap.add_argument("--json", default=None, help="write raw results to this file")
    # Greedy by default. At temperature 0.2 the same model scored 31 and 34 out
    # of 40 on consecutive runs, which is larger than the difference between
    # models, so sampling noise has to be removed for an A/B to mean anything.
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (0 = greedy, reproducible)")
    # Context knobs, exposed so the completeness/latency tradeoff can be
    # measured rather than guessed.
    ap.add_argument("--topk-context", type=int, default=None,
                    help="how many retrieved sections to send to the model")
    ap.add_argument("--focus-chars", type=int, default=None,
                    help="window size for narrow questions (0 disables trimming)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="total context character budget")
    args = ap.parse_args()
    config.TEMPERATURE = args.temp
    if args.topk_context is not None:
        config.TOP_K_CONTEXT = args.topk_context
    if args.focus_chars is not None:
        # 0 means "never trim": treat the window as effectively unlimited.
        config.CONTEXT_FOCUS_CHARS = args.focus_chars or 10 ** 9
    if args.max_context is not None:
        config.MAX_CONTEXT_CHARS = args.max_context

    # Overriding config before RAGSystem is built keeps the whole pipeline
    # (chunking, retrieval, routing, prompt cache) identical between runs, so a
    # comparison isolates model quality.
    if args.model:
        config.LOCAL_MODEL = args.model
    if args.draft:
        config.DRAFT_MODEL = args.draft

    suite = SUITE[:8] if args.quick else SUITE

    mx.reset_peak_memory()
    boot = time.time()
    system = RAGSystem(load_llm=True)
    boot_ms = (time.time() - boot) * 1000
    load_peak_gb = mx.get_peak_memory() / 1e9

    print(f"\nmodel:   {config.LOCAL_MODEL}")
    if config.DRAFT_MODEL:
        print(f"draft:   {config.DRAFT_MODEL}")
    print(f"context: top_k={config.TOP_K_CONTEXT} focus={config.CONTEXT_FOCUS_CHARS} "
          f"budget={config.MAX_CONTEXT_CHARS}")
    print(f"startup: {boot_ms / 1000:.1f}s (incl. warmup)")
    print(f"weights + warmup peak MLX memory: {load_peak_gb:.2f} GB\n")

    rows, failures = [], []
    for question, category, want_page, must in suite:
        answer, ttft, total, retrieved, routed, tps = stream_answer_full(
            system, question
        )
        low = answer.lower()

        page_ok = True
        if want_page:
            pages = " ".join(str(r.get("page_label", "")) for r in retrieved[:2])
            page_ok = want_page in pages

        missing = [t for t in must if t.lower() not in low]
        leak = LEAKS.search(answer)

        ok = page_ok and not missing and not leak
        rows.append({
            "q": question, "cat": category, "ttft": ttft, "total": total,
            "style": routed["style"], "conf": routed["confidence"],
            "words": len(answer.split()), "ok": ok, "tps": tps,
            "missing": missing, "leak": bool(leak),
            "top": retrieved[0].get("heading") if retrieved else "",
            "answer": answer,
        })
        if not ok:
            failures.append((question, category, want_page, page_ok, missing,
                             leak.group(0) if leak else None,
                             retrieved[0].get("heading") if retrieved else "-"))

        flag = "ok  " if ok else "FAIL"
        print(f"[{flag}] {ttft:6.0f}ms ttft {total:6.0f}ms tot "
              f"{routed['style']:<8} conf={routed['confidence']:.2f} "
              f"{len(answer.split()):3d}w  {question[:52]}")

    print("\n" + "=" * 74)
    ttfts = [r["ttft"] for r in rows]
    totals = [r["total"] for r in rows]
    words = [r["words"] for r in rows]
    tpss = [r["tps"] for r in rows if r["tps"] > 0]
    passed = sum(1 for r in rows if r["ok"])
    run_peak_gb = mx.get_peak_memory() / 1e9

    print(f"model:             {config.LOCAL_MODEL}")
    print(f"passed:            {passed}/{len(rows)}  ({passed / len(rows) * 100:.0f}%)")
    print(f"ttft  mean/median: {statistics.mean(ttfts):.0f} / {statistics.median(ttfts):.0f} ms")
    print(f"ttft  p90/max:     {sorted(ttfts)[int(len(ttfts) * 0.9)]:.0f} / {max(ttfts):.0f} ms")
    print(f"ttft  under 1s:    {sum(1 for t in ttfts if t < 1000)}/{len(ttfts)}")
    print(f"total mean/median: {statistics.mean(totals):.0f} / {statistics.median(totals):.0f} ms")
    print(f"throughput median: {statistics.median(tpss) if tpss else 0:.1f} tok/s")
    print(f"words mean/median: {statistics.mean(words):.0f} / {statistics.median(words):.0f}")
    print(f"peak MLX memory:   {run_peak_gb:.2f} GB")
    print(f"process peak RSS:  {rss_gb():.2f} GB")
    print(f"startup:           {boot_ms / 1000:.1f} s")
    print(f"leaks:             {sum(1 for r in rows if r['leak'])}/{len(rows)}")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r)
    print("\nby category:")
    for cat, rs in by_cat.items():
        print(f"  {cat:<14} {sum(1 for r in rs if r['ok'])}/{len(rs)} passed"
              f"   ttft med {statistics.median([r['ttft'] for r in rs]):.0f}ms"
              f"   styles: {sorted({r['style'] for r in rs})}")

    if failures:
        print("\nfailures:")
        for q, cat, want, page_ok, missing, leak, top in failures:
            print(f"  [{cat}] {q}")
            print(f"        top section: {top}")
            if not page_ok:
                print(f"        wrong page (wanted {want})")
            if missing:
                print(f"        missing terms: {missing}")
            if leak:
                print(f"        LEAKED: {leak!r}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "model": config.LOCAL_MODEL,
                "draft_model": config.DRAFT_MODEL,
                "passed": passed,
                "total_questions": len(rows),
                "ttft_median_ms": statistics.median(ttfts),
                "ttft_mean_ms": statistics.mean(ttfts),
                "ttft_p90_ms": sorted(ttfts)[int(len(ttfts) * 0.9)],
                "ttft_under_1s": sum(1 for t in ttfts if t < 1000),
                "total_median_ms": statistics.median(totals),
                "throughput_median_tps": statistics.median(tpss) if tpss else 0,
                "words_median": statistics.median(words),
                "peak_mlx_gb": run_peak_gb,
                "peak_rss_gb": rss_gb(),
                "startup_s": boot_ms / 1000,
                "leaks": sum(1 for r in rows if r["leak"]),
                "by_category": {
                    cat: {"passed": sum(1 for r in rs if r["ok"]), "n": len(rs)}
                    for cat, rs in by_cat.items()
                },
                "rows": rows,
            }, f, indent=2)
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
