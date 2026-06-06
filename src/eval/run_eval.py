"""
Golden-dataset evaluation for the RAG-grounded persona.

Mirrors the live voice path end-to-end for every golden question:

    retrieve(k=RAG_VOICE_TOP_K)  ->  format_for_voice()  ->  build_system_prompt(context)
        ->  generate answer (settings.LLM_MODEL)  ->  LLM-as-judge (--judge-model)

The judge (gpt-4.1-mini by default) scores each answer on three axes:

  * correctness   — does the answer match the gold reference?           (answerable items)
  * groundedness  — is every claim supported by the retrieved context?  (hallucination signal)
  * refusal       — did it correctly decline?                           (unanswerable items)

It also measures retrieval quality (recall@k, low-confidence rate) and latency
(retrieval + generation, sequential so the numbers mirror a single live turn).

Usage (run from src/):
    python -m eval.run_eval                         # full run, judged, mirrors voice (k=3)
    python -m eval.run_eval --k 6                   # evaluate the chat-style top-k
    python -m eval.run_eval --no-judge              # retrieval + generation only, no judge spend
    python -m eval.run_eval --limit 5               # smoke test on the first 5 questions
    python -m eval.run_eval --out ../temp/eval      # write the JSON report elsewhere
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openai import OpenAI

from agent.persona import build_system_prompt
from rag.retrieve import format_for_voice, retrieve
from utils.config import settings

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
_DEFAULT_OUT = _SRC.parent / "temp" / "eval"

# A turn that fails retrieval injects this note instead of junk — same intent as the
# voice pipeline's _NO_CONTEXT_NOTE, so the persona is told to decline honestly.
_NO_CONTEXT_NOTE = (
    "No grounded facts were retrieved for this question. If you don't already know the "
    "answer from your role, tell the caller you don't have that detail — do not guess."
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class ItemResult:
    id: str
    category: str
    answerable: bool
    question: str
    reference: str
    answer: str = ""
    context: str = ""
    retrieved_sources: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    top_score: Optional[float] = None
    low_confidence: bool = False
    source_hit: Optional[bool] = None     # expected source present in retrieved (answerable only)
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    # Judge verdicts
    correct: Optional[bool] = None
    grounded: Optional[bool] = None
    appropriate_refusal: Optional[bool] = None
    judge_reasoning: str = ""

    @property
    def passed(self) -> Optional[bool]:
        """Overall pass: answerable -> correct & grounded; unanswerable -> declined & grounded."""
        if self.correct is None and self.appropriate_refusal is None:
            return None
        if self.answerable:
            return bool(self.correct) and bool(self.grounded)
        return bool(self.appropriate_refusal) and bool(self.grounded)


# ---------------------------------------------------------------------------
# Pipeline mirror: retrieve -> ground -> answer
# ---------------------------------------------------------------------------
def _answer_question(client: OpenAI, model: str, item: dict, k: int) -> ItemResult:
    res = ItemResult(
        id=item["id"],
        category=item["category"],
        answerable=item["answerable"],
        question=item["question"],
        reference=item["reference"],
        expected_sources=item.get("expected_sources", []),
    )

    t0 = time.perf_counter()
    retrieval = retrieve(item["question"], k=k)
    res.retrieval_s = time.perf_counter() - t0

    res.context = format_for_voice(retrieval)
    res.low_confidence = retrieval.low_confidence
    res.retrieved_sources = [
        r.metadata.get("file_path") or r.metadata.get("project", "unknown")
        for r in retrieval.results
    ]
    res.top_score = retrieval.results[0].score if retrieval.results else None
    if item["answerable"] and item.get("expected_sources"):
        res.source_hit = any(s in res.retrieved_sources for s in item["expected_sources"])

    system = build_system_prompt(res.context) if res.context else build_system_prompt()
    messages = [{"role": "system", "content": system}]
    if not res.context:
        messages.append({"role": "assistant", "content": _NO_CONTEXT_NOTE})
    messages.append({"role": "user", "content": item["question"]})

    t1 = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=160,
    )
    res.generation_s = time.perf_counter() - t1
    res.answer = (completion.choices[0].message.content or "").strip()
    return res


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = (
    "You are a strict evaluator for an AI persona that answers questions about a software "
    "engineer named Ayush over the phone. You are given the QUESTION, the GOLD REFERENCE "
    "answer, the CONTEXT that was retrieved and shown to the persona, and the persona's "
    "ANSWER. Judge only what is present; do not use outside knowledge about Ayush.\n\n"
    "Return STRICT JSON with these keys:\n"
    '  "correct":  true if the ANSWER conveys the key facts of the GOLD REFERENCE without '
    "contradicting it. For an unanswerable question (gold reference says the info is not "
    "available), correct is true only if the ANSWER declines / says it lacks that detail.\n"
    '  "grounded": true if every factual claim in the ANSWER is supported by the CONTEXT '
    "(or the ANSWER makes no factual claim, e.g. a refusal). false if the ANSWER asserts a "
    "specific fact — a number, date, employer, or project — that is absent from the CONTEXT.\n"
    '  "appropriate_refusal": for an unanswerable question, true if the ANSWER declined / '
    "said it doesn't have that detail instead of inventing one; null for answerable questions.\n"
    '  "reasoning": one short sentence.\n'
    "Output only the JSON object, no markdown fences."
)


def _judge(client: OpenAI, model: str, res: ItemResult) -> None:
    payload = (
        f"QUESTION:\n{res.question}\n\n"
        f"GOLD REFERENCE:\n{res.reference}\n\n"
        f"CONTEXT SHOWN TO PERSONA:\n{res.context or '(none retrieved)'}\n\n"
        f"PERSONA ANSWER:\n{res.answer}"
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": payload},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )
        verdict = json.loads(completion.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        res.judge_reasoning = f"[judge error] {exc}"
        return

    res.correct = _as_bool(verdict.get("correct"))
    res.grounded = _as_bool(verdict.get("grounded"))
    res.appropriate_refusal = _as_bool(verdict.get("appropriate_refusal"))
    res.judge_reasoning = str(verdict.get("reasoning", ""))[:200]


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return None


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def _pct(num: int, den: int) -> Optional[float]:
    return round(100 * num / den, 1) if den else None


def _p50_p95(vals: list[float]) -> dict[str, float]:
    clean = [v for v in vals if v]
    if not clean:
        return {"p50": 0.0, "p95": 0.0}
    p95 = statistics.quantiles(clean, n=100)[94] if len(clean) >= 2 else max(clean)
    return {"p50": round(statistics.median(clean), 3), "p95": round(p95, 3)}


def _summarize(results: list[ItemResult], judged: bool) -> dict[str, Any]:
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    with_src = [r for r in ans if r.expected_sources]

    summary: dict[str, Any] = {
        "n_total": len(results),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "retrieval": {
            "recall_at_k_pct": _pct(sum(1 for r in with_src if r.source_hit), len(with_src)),
            "low_confidence_rate_pct": _pct(sum(1 for r in results if r.low_confidence), len(results)),
            "latency_secs": _p50_p95([r.retrieval_s for r in results]),
        },
        "generation_latency_secs": _p50_p95([r.generation_s for r in results]),
        "e2e_latency_secs": _p50_p95([r.retrieval_s + r.generation_s for r in results]),
    }

    if judged:
        graded = [r for r in results if r.passed is not None]
        summary["quality"] = {
            "overall_pass_pct": _pct(sum(1 for r in graded if r.passed), len(graded)),
            "answer_correctness_pct": _pct(sum(1 for r in ans if r.correct), len(ans)),
            "groundedness_pct": _pct(sum(1 for r in graded if r.grounded), len(graded)),
            "hallucination_rate_pct": _pct(sum(1 for r in graded if r.grounded is False), len(graded)),
            "refusal_accuracy_pct": _pct(sum(1 for r in una if r.appropriate_refusal), len(una)),
        }
    return summary


def _print_report(results: list[ItemResult], summary: dict[str, Any], judged: bool) -> None:
    print("\n" + "=" * 92)
    print(f"  PERSONA RAG EVAL  —  {len(results)} questions  "
          f"(LLM={settings.LLM_MODEL}, embed={settings.EMBEDDING_MODEL}, floor={settings.RAG_RELEVANCE_FLOOR})")
    print("=" * 92)

    if judged:
        header = f"{'id':<22}{'cat':<13}{'pass':<6}{'corr':<6}{'grnd':<6}{'src':<5}{'lowC':<6}{'ret_s':<7}"
    else:
        header = f"{'id':<22}{'cat':<13}{'src':<5}{'lowC':<6}{'ret_s':<7}{'top'}"
    print(header)
    print("-" * 92)
    for r in results:
        def mark(v: Optional[bool]) -> str:
            return "·" if v is None else ("✓" if v else "✗")
        if judged:
            print(f"{r.id:<22}{r.category:<13}{mark(r.passed):<6}{mark(r.correct):<6}"
                  f"{mark(r.grounded):<6}{mark(r.source_hit):<5}{('Y' if r.low_confidence else '-'):<6}"
                  f"{r.retrieval_s:<7.2f}")
        else:
            top = f"{r.top_score:.3f}" if r.top_score is not None else "—"
            print(f"{r.id:<22}{r.category:<13}{mark(r.source_hit):<5}"
                  f"{('Y' if r.low_confidence else '-'):<6}{r.retrieval_s:<7.2f}{top}")

    print("\n" + "-" * 92)
    print("  SUMMARY")
    print("-" * 92)
    rt = summary["retrieval"]
    print(f"  Retrieval recall@k (answerable) : {rt['recall_at_k_pct']}%")
    print(f"  Low-confidence rate             : {rt['low_confidence_rate_pct']}%")
    print(f"  Retrieval latency               : p50 {rt['latency_secs']['p50']}s  p95 {rt['latency_secs']['p95']}s")
    g = summary["generation_latency_secs"]
    e = summary["e2e_latency_secs"]
    print(f"  Generation latency              : p50 {g['p50']}s  p95 {g['p95']}s")
    print(f"  End-to-end (retrieval+gen)      : p50 {e['p50']}s  p95 {e['p95']}s")
    if judged:
        q = summary["quality"]
        print()
        print(f"  Overall pass                    : {q['overall_pass_pct']}%")
        print(f"  Answer correctness (answerable) : {q['answer_correctness_pct']}%")
        print(f"  Groundedness (no hallucination) : {q['groundedness_pct']}%")
        print(f"  Hallucination rate              : {q['hallucination_rate_pct']}%")
        print(f"  Refusal accuracy (unanswerable) : {q['refusal_accuracy_pct']}%")

        fails = [r for r in results if r.passed is False]
        if fails:
            print("\n  FAILURES")
            print("-" * 92)
            for r in fails:
                print(f"  [{r.id}] {r.question}")
                print(f"     answer : {r.answer[:140]}")
                print(f"     judge  : {r.judge_reasoning}")
    print("=" * 92 + "\n")


# ---------------------------------------------------------------------------
# Regression gate
#
# Thresholds for `--gate` (CI / pre-submission, e.g. before the gpt-4.1-mini -> gpt-4.1
# swap). Groundedness and refusal are the safety properties that must never regress, so
# they gate hardest; recall guards retrieval health; overall pass is a loose floor (it
# has temp-driven run-to-run variance, so it only catches a catastrophic drop). These had
# real headroom at the time of writing (grounded/refusal 100%, recall 95%, pass ~90%).
# ---------------------------------------------------------------------------
_GATE_THRESHOLDS = {
    "groundedness_pct": 95.0,        # in summary["quality"]
    "refusal_accuracy_pct": 90.0,    # in summary["quality"]
    "overall_pass_pct": 80.0,        # in summary["quality"]
    "recall_at_k_pct": 85.0,         # in summary["retrieval"]
}


def _check_gate(summary: dict[str, Any]) -> bool:
    """Print PASS/FAIL per threshold; return True only if every hard gate passes."""
    quality = summary.get("quality", {})
    actual = {**quality, "recall_at_k_pct": summary["retrieval"]["recall_at_k_pct"]}
    print("-" * 92)
    print("  REGRESSION GATE")
    print("-" * 92)
    ok = True
    for metric, floor in _GATE_THRESHOLDS.items():
        got = actual.get(metric)
        passed = got is not None and got >= floor
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {metric:<24} {got}%  (>= {floor}%)")
    print(f"\n  GATE {'PASSED' if ok else 'FAILED'}")
    print("=" * 92 + "\n")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_golden(limit: Optional[int]) -> list[dict]:
    items = [json.loads(line) for line in _GOLDEN_PATH.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-dataset persona eval.")
    parser.add_argument("--k", type=int, default=settings.RAG_VOICE_TOP_K, help="retrieval top-k (default: voice)")
    parser.add_argument("--model", default=settings.LLM_MODEL, help="answer model")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="judge model")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge (retrieval + generation only)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="directory to write the JSON report")
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero if quality/retrieval thresholds aren't met (CI / pre-submission)")
    args = parser.parse_args()

    if args.gate and args.no_judge:
        parser.error("--gate needs the judge (quality metrics); drop --no-judge")

    items = _load_golden(args.limit)
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    judged = not args.no_judge

    print(f"Evaluating {len(items)} questions  |  k={args.k}  model={args.model}  "
          f"judge={'off' if args.no_judge else args.judge_model}")

    # Retrieval + generation are run sequentially so latency mirrors a single live turn.
    results: list[ItemResult] = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['id']}…", end="\r", flush=True)
        results.append(_answer_question(client, args.model, item, args.k))

    # Judging is latency-insensitive — fan it out.
    if judged:
        print("\n  Judging…                          ")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda r: _judge(client, args.judge_model, r), results))

    summary = _summarize(results, judged)
    _print_report(results, summary, judged)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eval_{ts}.json"
    out_path.write_text(json.dumps({
        "meta": {
            "timestamp": ts,
            "k": args.k,
            "answer_model": args.model,
            "judge_model": None if args.no_judge else args.judge_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "relevance_floor": settings.RAG_RELEVANCE_FLOOR,
            "overfetch_factor": settings.RAG_OVERFETCH_FACTOR,
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"Report written to {out_path}")

    if args.gate:
        sys.exit(0 if _check_gate(summary) else 1)


if __name__ == "__main__":
    main()
