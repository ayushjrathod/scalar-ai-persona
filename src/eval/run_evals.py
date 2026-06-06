"""
Part C — unified eval suite.

Runs all five Part C measurements in one pass and writes a single combined
report to ``src/eval/results/run_{timestamp}.json`` (consumed by ``report.py``).

The five evals:

  1. Retrieval precision/recall  — precision@k, recall@k, mean relevance, by category
  2. Chat hallucination rate     — LLM-as-judge over the live chat answer chain
  3. Voice latency               — p50/p95/min/max from the metrics JSONL (responded turns)
  4. Task completion             — booking success rate (manual / flag / auto-from-metrics)
  5. Transcription proxy         — malformed-transcript rate from logs.txt (no ground truth)

This is an orchestrator, not a rewrite: evals 1+2 reuse the judge and records from
``eval.run_eval``; eval 3 reuses ``agent.monitoring``'s loader. The retrieval + answer
chain here mirrors the CHAT path (k=RAG_CHAT_TOP_K, format_for_chat, build_chat_prompt) —
that's the surface the hallucination/retrieval numbers are reported against.

Usage (run from src/):
    python -m eval.run_evals                                   # full suite, judged
    python -m eval.run_evals --no-judge                        # skip eval 2 (no judge spend)
    python -m eval.run_evals --bookings-attempted 8 --bookings-succeeded 7
    python -m eval.run_evals --limit 5                         # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openai import OpenAI

from agent.monitoring import load_turns
from agent.persona import build_chat_prompt
from eval.run_eval import ItemResult, _judge, _load_golden
from rag.retrieve import format_for_chat, retrieve
from utils.config import settings

_PROJECT_ROOT = _SRC.parent
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DEFAULT_LOGS = _PROJECT_ROOT / "logs.txt"

# Injected when retrieval returns nothing — mirrors the live chat endpoint (chat.py).
_NO_CONTEXT_HINT = (
    "No relevant facts were found for this question. "
    "If you cannot answer from your representative knowledge, "
    "say: 'I don't have specific details on that.'"
)

# A new-call gap heuristic. metrics.jsonl carries no call/room id, so we segment the
# voice turn stream into "calls" wherever the inter-turn gap exceeds this many seconds.
_SESSION_GAP_S = 120.0


# ---------------------------------------------------------------------------
# Small local stats helpers (kept here so this file stands alone)
# ---------------------------------------------------------------------------
def _pct(num: int, den: int) -> Optional[float]:
    return round(100 * num / den, 1) if den else None


def _percentiles(vals: list[float]) -> dict[str, Optional[float]]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return {"p50": None, "p95": None, "min": None, "max": None, "n": 0}
    p95 = statistics.quantiles(clean, n=100)[94] if len(clean) >= 2 else max(clean)
    return {
        "p50": round(statistics.median(clean), 3),
        "p95": round(p95, 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "n": len(clean),
    }


def _enrich_query(query: str) -> str:
    """Mirror the chat endpoint's single-turn enrichment: prepend the name if absent so
    retrieval has a strong anchor. (The chat route also folds in history for short
    follow-ups; the golden set is single-turn, so only the name prefix applies.)"""
    return query if "ayush" in query.lower() else f"Ayush Rathod {query}"


# ---------------------------------------------------------------------------
# Chat-path answer: retrieve(k) -> format_for_chat -> build_chat_prompt -> generate
# Mirrors api/routes/chat.py (minus tool-calling, irrelevant to grounding/retrieval).
# ---------------------------------------------------------------------------
def _answer_chat(client: OpenAI, model: str, item: dict, k: int) -> ItemResult:
    res = ItemResult(
        id=item["id"],
        category=item["category"],
        answerable=item["answerable"],
        question=item["question"],
        reference=item["reference"],
        expected_sources=item.get("expected_sources", []),
    )

    t0 = time.perf_counter()
    retrieval = retrieve(_enrich_query(item["question"]), k=k)
    res.retrieval_s = time.perf_counter() - t0

    res.low_confidence = retrieval.low_confidence
    res.retrieved_sources = [
        r.metadata.get("file_path") or r.metadata.get("project", "unknown")
        for r in retrieval.results
    ]
    res.top_score = retrieval.results[0].score if retrieval.results else None
    if item["answerable"] and item.get("expected_sources"):
        res.source_hit = any(s in res.retrieved_sources for s in item["expected_sources"])

    context = format_for_chat(retrieval)
    res.context = context
    system = build_chat_prompt(context or _NO_CONTEXT_HINT)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": item["question"]},
    ]

    t1 = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # deterministic for reproducible grounding/correctness scoring
        max_tokens=220,
    )
    res.generation_s = time.perf_counter() - t1
    res.answer = (completion.choices[0].message.content or "").strip()
    return res


# ---------------------------------------------------------------------------
# EVAL 1 — retrieval precision / recall
# ---------------------------------------------------------------------------
def eval_retrieval(results: list[ItemResult], k: int) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable and r.expected_sources]
    ooc = [r for r in results if not r.answerable]

    per_q = []
    for r in answerable:
        retrieved = r.retrieved_sources
        expected = set(r.expected_sources)
        n_rel = sum(1 for s in retrieved if s in expected)
        precision = (n_rel / len(retrieved)) if retrieved else 0.0
        recall = 1.0 if r.source_hit else 0.0
        per_q.append({
            "id": r.id, "category": r.category,
            "precision": round(precision, 3), "recall": recall,
            "top_score": round(r.top_score, 4) if r.top_score is not None else None,
        })

    def _cat_breakdown() -> dict[str, dict]:
        cats: dict[str, list] = {}
        for q in per_q:
            cats.setdefault(q["category"], []).append(q)
        out = {}
        for cat, qs in sorted(cats.items()):
            out[cat] = {
                "n": len(qs),
                "precision": round(statistics.mean(q["precision"] for q in qs), 3),
                "recall": round(statistics.mean(q["recall"] for q in qs), 3),
            }
        return out

    relevances = [r.top_score for r in answerable if r.top_score is not None]
    # Out-of-corpus: low_confidence=True is the correct "refuse" signal. low_confidence=False
    # means junk cleared the floor — a retrieval false positive.
    ooc_refused = sum(1 for r in ooc if r.low_confidence)
    ooc_false_pos = [r.id for r in ooc if not r.low_confidence]

    return {
        "k": k,
        "n_answerable_scored": len(answerable),
        "precision_at_k": round(statistics.mean(q["precision"] for q in per_q), 3) if per_q else None,
        "recall_at_k": round(statistics.mean(q["recall"] for q in per_q), 3) if per_q else None,
        "mean_relevance_score": round(statistics.mean(relevances), 4) if relevances else None,
        "by_category": _cat_breakdown(),
        "out_of_corpus": {
            "n": len(ooc),
            "refusal_rate": _pct(ooc_refused, len(ooc)),
            "refused": ooc_refused,
            "false_positives": ooc_false_pos,
        },
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# EVAL 2 — hallucination rate (LLM-as-judge)
# ---------------------------------------------------------------------------
def eval_hallucination(results: list[ItemResult]) -> dict[str, Any]:
    graded = [r for r in results if r.grounded is not None]
    answerable = [r for r in graded if r.answerable]
    ooc = [r for r in graded if not r.answerable]

    halluc = [r for r in graded if r.grounded is False]

    def _cat_breakdown() -> dict[str, dict]:
        cats: dict[str, list] = {}
        for r in graded:
            cats.setdefault(r.category, []).append(r)
        out = {}
        for cat, rs in sorted(cats.items()):
            out[cat] = {
                "n": len(rs),
                "hallucination_rate_pct": _pct(sum(1 for r in rs if r.grounded is False), len(rs)),
                "grounded_pct": _pct(sum(1 for r in rs if r.grounded), len(rs)),
            }
        return out

    # Worst offenders: any hallucination, or an answerable item judged incorrect.
    offenders = [
        {
            "id": r.id, "category": r.category, "question": r.question,
            "answer": r.answer[:200], "grounded": r.grounded, "correct": r.correct,
            "reasoning": r.judge_reasoning,
        }
        for r in graded
        if r.grounded is False or (r.answerable and r.correct is False)
    ]

    # End-to-end refusal: did the persona decline the out-of-corpus question without
    # fabricating? Robust to a known judge quirk where appropriate_refusal is left null on
    # a clear "the resume doesn't mention X" decline — we accept it if the judge marked it
    # correct (declined) AND grounded (no invented fact). grounded=False would be a real
    # fabrication. (Distinct from retrieval-layer low_confidence gating in eval 1 — the
    # name-magnet contact chunk often clears the floor, yet the model still declines.)
    n_refused = sum(
        1 for r in ooc
        if r.grounded is not False and (r.appropriate_refusal or r.correct)
    )
    n_ooc_fabricated = sum(1 for r in ooc if r.grounded is False)
    return {
        "n_graded": len(graded),
        "hallucination_rate_pct": _pct(len(halluc), len(graded)),
        "grounded_pct": _pct(sum(1 for r in graded if r.grounded), len(graded)),
        "answer_correctness_pct": _pct(sum(1 for r in answerable if r.correct), len(answerable)),
        "n_unanswerable": len(ooc),
        "n_refused_end_to_end": n_refused,
        "n_out_of_corpus_fabricated": n_ooc_fabricated,
        "refusal_accuracy_pct": _pct(n_refused, len(ooc)),
        "by_category": _cat_breakdown(),
        "worst_offenders": offenders,
    }


# ---------------------------------------------------------------------------
# EVAL 3 — voice latency from the metrics JSONL
# ---------------------------------------------------------------------------
def eval_voice_latency() -> dict[str, Any]:
    turns = load_turns(limit=100_000)
    voice = [t for t in turns if getattr(t, "source", "voice") == "voice" and t.responded]

    warning = None
    if len(voice) < 5:
        warning = (
            f"Only {len(voice)} responded voice turns in the metrics log — too few for "
            "reliable p50/p95. Make more test calls before quoting these percentiles."
        )

    # Segment into calls by inter-turn gap (no call id is persisted).
    ts = sorted(t.timestamp for t in voice)
    sessions = 1 if ts else 0
    for a, b in zip(ts, ts[1:]):
        if b - a > _SESSION_GAP_S:
            sessions += 1

    e2e = [t.e2e_latency for t in voice if t.e2e_latency]
    ttfb = [t.tts_ttfb for t in voice if t.tts_ttfb]
    e2e_under_2s = sum(1 for v in e2e if v < 2.0)
    ttfb_under_2s = sum(1 for v in ttfb if v < 2.0)

    return {
        "responded_voice_turns": len(voice),
        "approx_call_sessions": sessions,
        "session_gap_seconds": _SESSION_GAP_S,
        # Full perceived turn: user stops speaking -> first agent audio. Includes the
        # semantic end-of-turn wait + transcription + retrieval + LLM + TTS.
        "e2e_latency_secs": _percentiles(e2e),
        # First audio out of the agent once it starts responding — this is the
        # assignment's "<2s first response" metric (greeting TTFB ~0.21s).
        "tts_ttfb_secs": _percentiles(ttfb),
        "llm_ttft_secs": _percentiles([t.llm_ttft for t in voice if t.llm_ttft]),
        "retrieval_latency_secs": _percentiles([t.retrieval_latency for t in voice if t.retrieval_latency]),
        "transcription_delay_secs": _percentiles([t.transcription_delay for t in voice if t.transcription_delay]),
        "pct_e2e_under_2s": _pct(e2e_under_2s, len(e2e)),
        "pct_first_audio_under_2s": _pct(ttfb_under_2s, len(ttfb)),
        "interruption_rate_pct": _pct(sum(1 for t in voice if t.interrupted), len(voice)),
        "latency_note": (
            "E2E is the full perceived turn (user stops -> agent speaks); it is dominated by "
            "cross-region RAG retrieval (~1.3s, in the critical path) and the semantic "
            "end-of-turn wait, not by synthesis. First audio out (TTS TTFB) is the <2s "
            "first-response figure."
        ),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# EVAL 4 — task completion (booking success rate)
# ---------------------------------------------------------------------------
def eval_task_completion(attempted: Optional[int], succeeded: Optional[int]) -> dict[str, Any]:
    # Auto-hint from metrics: count distinct booking-flow tool calls. This is a lower
    # bound and a sanity check, not the headline number — confirmed bookings are best
    # tallied by hand from real test calls.
    turns = load_turns(limit=100_000)
    slot_lookups = sum(1 for t in turns if getattr(t, "tool_name", None) == "get_available_slots")
    bookings_logged = sum(1 for t in turns if getattr(t, "tool_name", None) == "book_slot")
    auto_hint = {"slot_lookups_logged": slot_lookups, "bookings_logged": bookings_logged}

    if attempted is None and sys.stdin.isatty():
        print("\n  EVAL 4 — Booking test results (press Enter to skip)")
        print(f"  (metrics hint: {slot_lookups} slot lookups, {bookings_logged} confirmed bookings logged)")
        try:
            a = input("    Attempts (N): ").strip()
            s = input("    Confirmed Cal.com bookings: ").strip()
            attempted = int(a) if a else None
            succeeded = int(s) if s else None
        except (ValueError, EOFError):
            attempted = succeeded = None

    if attempted is not None:
        succeeded = succeeded or 0
        return {
            "source": "manual",
            "attempts": attempted,
            "successes": succeeded,
            "success_rate_pct": _pct(succeeded, attempted),
            "metrics_hint": auto_hint,
            "note": None,
        }

    # No manual input: fall back to the metrics hint, clearly flagged as provisional.
    return {
        "source": "auto_from_metrics",
        "attempts": slot_lookups,
        "successes": bookings_logged,
        "success_rate_pct": _pct(bookings_logged, slot_lookups) if slot_lookups else None,
        "metrics_hint": auto_hint,
        "note": (
            "Auto-derived from the metrics log: booking confirmations are not separately "
            "instrumented, so 'successes' counts logged book_slot tool calls and is a lower "
            "bound. Re-run with --bookings-attempted N --bookings-succeeded M after real "
            "test calls for the headline number."
        ),
    }


# ---------------------------------------------------------------------------
# EVAL 5 — transcription accuracy proxy (no ground truth available)
# ---------------------------------------------------------------------------
_TRANSCRIPT_RE = re.compile(r'"user_transcript":\s*"((?:[^"\\]|\\.)*)"')
_INAUDIBLE_RE = re.compile(r"\[(inaudible|unclear|unintelligible|noise|silence)\]", re.I)
# A trailing function word strongly suggests STT cut the utterance off mid-phrase.
_DANGLING_RE = re.compile(
    r"\b(the|a|an|to|of|for|and|or|but|with|about|that|is|was|my|your|his)$", re.I
)


def _is_malformed(text: str) -> tuple[bool, Optional[str]]:
    t = text.strip()
    if not t:
        return True, "empty"
    if _INAUDIBLE_RE.search(t):
        return True, "inaudible_marker"
    # Single-token utterance that isn't a normal short reply.
    words = t.split()
    if len(words) == 1 and t.lower().rstrip(".!?") not in {
        "okay", "ok", "yes", "no", "yeah", "yep", "nope", "sure", "thanks", "hello", "hi", "bye"
    }:
        return True, "single_token"
    # Sentence fragment ending on a dangling function word (no terminal punctuation).
    if t[-1] not in ".!?" and _DANGLING_RE.search(t):
        return True, "dangling_fragment"
    return False, None


def eval_transcription_proxy(logs_path: Path) -> dict[str, Any]:
    if not logs_path.exists():
        return {
            "available": False,
            "note": f"No transcript log at {logs_path}. metrics.jsonl stores no transcript "
                    "text, so the proxy has nothing to read.",
        }

    raw = logs_path.read_text(errors="ignore")
    transcripts = [m.group(1) for m in _TRANSCRIPT_RE.finditer(raw)]
    flagged = []
    for t in transcripts:
        bad, reason = _is_malformed(t)
        if bad:
            flagged.append({"text": t[:80], "reason": reason})

    n = len(transcripts)
    return {
        "available": True,
        "is_proxy": True,
        "source_log": str(logs_path.name),
        "n_transcripts": n,
        "n_malformed": len(flagged),
        "proxy_wer_pct": _pct(len(flagged), n),
        "flagged": flagged,
        "sample_size_warning": (
            f"Only {n} transcripts available — treat the proxy WER as directional, not precise."
            if n < 20 else None
        ),
        "note": (
            "Proxy, NOT true WER. True WER needs paired ground-truth transcripts for the same "
            "audio, which were never recorded (the assignment has no labeled audio set). This "
            "counts structurally malformed Sarvam STT outputs (empty, inaudible markers, "
            "single-token, or mid-phrase truncations) over total transcripts in logs.txt."
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Part C — unified eval suite.")
    parser.add_argument("--k", type=int, default=settings.RAG_CHAT_TOP_K, help="retrieval top-k (chat path)")
    parser.add_argument("--model", default=settings.LLM_MODEL, help="answer model")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="judge model")
    parser.add_argument("--no-judge", action="store_true", help="skip eval 2 (no judge spend)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--bookings-attempted", type=int, default=None, help="eval 4: booking attempts")
    parser.add_argument("--bookings-succeeded", type=int, default=None, help="eval 4: confirmed bookings")
    parser.add_argument("--logs", default=str(_DEFAULT_LOGS), help="path to call log for transcript proxy")
    parser.add_argument("--out", default=str(_RESULTS_DIR), help="directory for the combined JSON report")
    args = parser.parse_args()

    items = _load_golden(args.limit)
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    judged = not args.no_judge

    print(f"Part C eval suite | {len(items)} questions | chat k={args.k} | "
          f"model={args.model} | judge={'off' if args.no_judge else args.judge_model}")

    # --- Evals 1 + 2 share one retrieve+answer pass over the chat path ---
    print("  [1/5,2/5] retrieval + chat answers…")
    results: list[ItemResult] = []
    for i, item in enumerate(items, 1):
        print(f"      ({i}/{len(items)}) {item['id']}…        ", end="\r", flush=True)
        results.append(_answer_chat(client, args.model, item, args.k))

    if judged:
        print("\n  judging answers…                        ")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda r: _judge(client, args.judge_model, r), results))

    retrieval_eval = eval_retrieval(results, args.k)
    hallucination_eval = eval_hallucination(results) if judged else {"skipped": True}

    print("  [3/5] voice latency from metrics log…")
    voice_eval = eval_voice_latency()

    print("  [4/5] task completion…")
    booking_eval = eval_task_completion(args.bookings_attempted, args.bookings_succeeded)

    print("  [5/5] transcription proxy…")
    transcription_eval = eval_transcription_proxy(Path(args.logs))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "meta": {
            "timestamp": ts,
            "n_questions": len(items),
            "chat_top_k": args.k,
            "answer_model": args.model,
            "judge_model": None if args.no_judge else args.judge_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "relevance_floor": settings.RAG_RELEVANCE_FLOOR,
            "voice_top_k": settings.RAG_VOICE_TOP_K,
        },
        "eval_1_retrieval": retrieval_eval,
        "eval_2_hallucination": hallucination_eval,
        "eval_3_voice_latency": voice_eval,
        "eval_4_task_completion": booking_eval,
        "eval_5_transcription_proxy": transcription_eval,
        "items": [asdict(r) for r in results],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))

    _print_console_summary(report)
    print(f"\nCombined report written to {out_path}")


def _print_console_summary(report: dict[str, Any]) -> None:
    r1 = report["eval_1_retrieval"]
    r2 = report["eval_2_hallucination"]
    r3 = report["eval_3_voice_latency"]
    r4 = report["eval_4_task_completion"]
    r5 = report["eval_5_transcription_proxy"]

    print("\n" + "=" * 72)
    print("  PART C EVAL SUMMARY")
    print("=" * 72)
    print("  [1] Retrieval")
    print(f"      precision@{r1['k']}: {r1['precision_at_k']}   recall@{r1['k']}: {r1['recall_at_k']}"
          f"   mean relevance: {r1['mean_relevance_score']}")
    print(f"      out-of-corpus refusal: {r1['out_of_corpus']['refusal_rate']}%"
          f"  (false positives: {r1['out_of_corpus']['false_positives'] or 'none'})")
    if not r2.get("skipped"):
        print("  [2] Chat groundedness")
        print(f"      hallucination rate: {r2['hallucination_rate_pct']}%   grounded: {r2['grounded_pct']}%"
              f"   correctness: {r2['answer_correctness_pct']}%   refusal acc: {r2['refusal_accuracy_pct']}%")
    print("  [3] Voice latency")
    e2e = r3["e2e_latency_secs"]
    ttfb = r3["tts_ttfb_secs"]
    print(f"      e2e p50/p95: {e2e['p50']}s / {e2e['p95']}s (under-2s {r3['pct_e2e_under_2s']}%)   "
          f"first-audio TTFB p50/p95: {ttfb['p50']}s / {ttfb['p95']}s (under-2s {r3['pct_first_audio_under_2s']}%)")
    print(f"      turns: {r3['responded_voice_turns']} over ~{r3['approx_call_sessions']} calls")
    if r3["warning"]:
        print(f"      ! {r3['warning']}")
    print("  [4] Task completion")
    print(f"      booking success: {r4['successes']}/{r4['attempts']} "
          f"({r4['success_rate_pct']}%)  [source: {r4['source']}]")
    print("  [5] Transcription proxy")
    if r5.get("available"):
        print(f"      proxy WER: {r5['proxy_wer_pct']}%  ({r5['n_malformed']}/{r5['n_transcripts']} transcripts)")
    else:
        print(f"      unavailable: {r5.get('note')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
