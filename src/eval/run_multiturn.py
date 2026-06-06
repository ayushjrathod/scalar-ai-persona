"""
Multi-turn / follow-up evaluation for the RAG-grounded persona.

The single-shot eval (run_eval.py) sends every question standalone. But the real
failure in the call logs was a *follow-up*: "He has a lot of experience" then
"Where did he work?" — a pronoun that only resolves from prior turns. The live
pipeline retrieves on the RAW last user message (agent/pipeline.py:66), so a
follow-up like "where did he work?" or "what about Advista?" embeds with no referent.

This runner replays each golden conversation turn-by-turn, accumulating history exactly
as the voice agent would (static system prompt + persisted user/assistant turns +
per-turn retrieved context), and grades the turns marked `grade: true`. The report
splits results into FIRST-TURN vs FOLLOW-UP so the follow-up gap is visible.

--rewrite condenses the conversation into a standalone query (resolving pronouns) before
retrieval — the candidate fix. Run once without it (production behavior) and once with it
to measure the delta.

Usage (run from src/):
    python -m eval.run_multiturn                  # raw last-message retrieval (production)
    python -m eval.run_multiturn --rewrite        # standalone-query rewrite before retrieval
    python -m eval.run_multiturn --no-judge        # retrieval-only, no judge spend
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import time

from openai import OpenAI

from agent.persona import build_system_prompt
from eval.run_eval import _as_bool, _judge, _p50_p95, _pct
from rag.retrieve import format_for_voice, retrieve
from utils.config import settings

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden_multiturn.jsonl"
_DEFAULT_OUT = _SRC.parent / "temp" / "eval"

# Mirrors agent/pipeline.py:_GREETING so the model sees the same conversation opener the
# caller hears. Duplicated (not imported) to avoid pulling in the heavy livekit deps.
_GREETING = (
    "Hi, this is Ayush's AI representative. "
    "I'm here to answer questions about his background and experience. "
    "What would you like to know?"
)

_REWRITE_SYSTEM = (
    "You rewrite a caller's latest message into a single standalone search query for a "
    "knowledge base of one engineer's resume and projects. Resolve pronouns and implicit "
    "references ('he', 'it', 'that', 'there', 'the X one', 'what about Y') using the "
    "conversation so the query stands on its own. Output ONLY the rewritten query text."
)


@dataclass
class TurnResult:
    convo_id: str
    category: str
    turn_index: int           # 0-based index within the conversation's user turns
    is_followup: bool
    answerable: bool
    question: str             # the actual user utterance (what the model answers)
    reference: str
    query_used: str = ""      # what retrieval actually searched on (raw, or rewritten)
    answer: str = ""
    context: str = ""
    retrieved_sources: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    top_score: Optional[float] = None
    low_confidence: bool = False
    source_hit: Optional[bool] = None
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    correct: Optional[bool] = None
    grounded: Optional[bool] = None
    appropriate_refusal: Optional[bool] = None
    judge_reasoning: str = ""

    @property
    def passed(self) -> Optional[bool]:
        if self.correct is None and self.appropriate_refusal is None:
            return None
        if self.answerable:
            return bool(self.correct) and bool(self.grounded)
        return bool(self.appropriate_refusal) and bool(self.grounded)


def _rewrite_query(client: OpenAI, history: list[dict], user_msg: str) -> str:
    """Condense recent history + the latest message into a standalone retrieval query."""
    convo = "\n".join(
        f"{'Caller' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in history[-6:]
        if m["role"] in {"user", "assistant"}
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest caller message: {user_msg}"},
            ],
            temperature=0,
            max_tokens=60,
        )
        return (completion.choices[0].message.content or "").strip() or user_msg
    except Exception:  # noqa: BLE001
        return user_msg


def _run_conversation(client: OpenAI, model: str, convo: dict, k: int, rewrite: bool) -> list[TurnResult]:
    history: list[dict] = [{"role": "assistant", "content": _GREETING}]
    graded: list[TurnResult] = []
    user_turn_idx = 0

    for turn in convo["turns"]:
        user_msg = turn["user"]
        is_followup = user_turn_idx > 0

        # Retrieval query: production retrieves on the raw last message; --rewrite resolves
        # references first. First turns never need a rewrite.
        query = _rewrite_query(client, history, user_msg) if (rewrite and is_followup) else user_msg

        t0 = time.perf_counter()
        retrieval = retrieve(query, k=k)
        retrieval_s = time.perf_counter() - t0
        context = format_for_voice(retrieval)

        system = build_system_prompt(context) if context else build_system_prompt()
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}]

        t1 = time.perf_counter()
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=160,
        )
        generation_s = time.perf_counter() - t1
        answer = (completion.choices[0].message.content or "").strip()

        # Persist the real turn into history; the retrieved context is ephemeral (matches
        # the pipeline, which adds context to a throwaway turn_ctx, not Agent.chat_ctx).
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer})
        user_turn_idx += 1

        if not turn.get("grade"):
            continue

        sources = [
            r.metadata.get("file_path") or r.metadata.get("project", "unknown")
            for r in retrieval.results
        ]
        expected = turn.get("expected_sources", [])
        tr = TurnResult(
            convo_id=convo["id"],
            category=convo["category"],
            turn_index=user_turn_idx - 1,
            is_followup=is_followup,
            answerable=turn["answerable"],
            question=user_msg,
            reference=turn["reference"],
            query_used=query,
            answer=answer,
            context=context,
            retrieved_sources=sources,
            expected_sources=expected,
            top_score=retrieval.results[0].score if retrieval.results else None,
            low_confidence=retrieval.low_confidence,
            source_hit=(any(s in sources for s in expected) if (turn["answerable"] and expected) else None),
            retrieval_s=retrieval_s,
            generation_s=generation_s,
        )
        graded.append(tr)

    return graded


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def _group_stats(turns: list[TurnResult], judged: bool) -> dict[str, Any]:
    ans = [t for t in turns if t.answerable]
    with_src = [t for t in ans if t.expected_sources]
    out: dict[str, Any] = {
        "n": len(turns),
        "recall_at_k_pct": _pct(sum(1 for t in with_src if t.source_hit), len(with_src)),
        "low_confidence_pct": _pct(sum(1 for t in turns if t.low_confidence), len(turns)),
    }
    if judged:
        graded = [t for t in turns if t.passed is not None]
        out |= {
            "pass_pct": _pct(sum(1 for t in graded if t.passed), len(graded)),
            "correctness_pct": _pct(sum(1 for t in ans if t.correct), len(ans)),
            "groundedness_pct": _pct(sum(1 for t in graded if t.grounded), len(graded)),
        }
    return out


def _summarize(turns: list[TurnResult], judged: bool) -> dict[str, Any]:
    first = [t for t in turns if not t.is_followup]
    follow = [t for t in turns if t.is_followup]
    return {
        "n_graded": len(turns),
        "overall": _group_stats(turns, judged),
        "first_turn": _group_stats(first, judged),
        "follow_up": _group_stats(follow, judged),
        "retrieval_latency_secs": _p50_p95([t.retrieval_s for t in turns]),
        "e2e_latency_secs": _p50_p95([t.retrieval_s + t.generation_s for t in turns]),
    }


def _print_report(turns: list[TurnResult], summary: dict[str, Any], judged: bool, mode: str) -> None:
    print("\n" + "=" * 96)
    print(f"  MULTI-TURN PERSONA EVAL  —  {summary['n_graded']} graded turns  "
          f"(retrieval mode: {mode}, LLM={settings.LLM_MODEL}, floor={settings.RAG_RELEVANCE_FLOOR})")
    print("=" * 96)

    cols = f"{'convo':<24}{'turn':<6}{'kind':<11}"
    cols += f"{'pass':<6}{'corr':<6}{'grnd':<6}{'src':<5}{'lowC':<6}" if judged else f"{'src':<5}{'lowC':<6}{'top':<7}"
    print(cols)
    print("-" * 96)
    for t in turns:
        def mark(v: Optional[bool]) -> str:
            return "·" if v is None else ("✓" if v else "✗")
        kind = "follow-up" if t.is_followup else "first"
        row = f"{t.convo_id:<24}{t.turn_index:<6}{kind:<11}"
        if judged:
            row += (f"{mark(t.passed):<6}{mark(t.correct):<6}{mark(t.grounded):<6}"
                    f"{mark(t.source_hit):<5}{('Y' if t.low_confidence else '-'):<6}")
        else:
            top = f"{t.top_score:.3f}" if t.top_score is not None else "—"
            row += f"{mark(t.source_hit):<5}{('Y' if t.low_confidence else '-'):<6}{top:<7}"
        print(row)

    print("\n" + "-" * 96)
    print(f"  SUMMARY BY TURN POSITION  (retrieval mode: {mode})")
    print("-" * 96)
    hdr = f"{'group':<14}{'n':<5}{'recall@k':<11}{'low-conf':<11}"
    if judged:
        hdr += f"{'pass':<8}{'correct':<10}{'grounded':<10}"
    print(hdr)
    for label, key in (("overall", "overall"), ("first-turn", "first_turn"), ("follow-up", "follow_up")):
        g = summary[key]
        row = f"{label:<14}{g['n']:<5}{str(g['recall_at_k_pct']) + '%':<11}{str(g['low_confidence_pct']) + '%':<11}"
        if judged:
            row += (f"{str(g.get('pass_pct')) + '%':<8}{str(g.get('correctness_pct')) + '%':<10}"
                    f"{str(g.get('groundedness_pct')) + '%':<10}")
        print(row)

    rt = summary["retrieval_latency_secs"]
    e = summary["e2e_latency_secs"]
    print(f"\n  Retrieval latency : p50 {rt['p50']}s  p95 {rt['p95']}s")
    print(f"  End-to-end        : p50 {e['p50']}s  p95 {e['p95']}s")

    if judged:
        fails = [t for t in turns if t.passed is False]
        if fails:
            print("\n  FOLLOW-UP / TURN FAILURES")
            print("-" * 96)
            for t in fails:
                print(f"  [{t.convo_id} t{t.turn_index} {'follow-up' if t.is_followup else 'first'}] {t.question}")
                if t.query_used != t.question:
                    print(f"     searched: {t.query_used!r}")
                print(f"     answer  : {t.answer[:130]}")
                print(f"     judge   : {t.judge_reasoning}")
    print("=" * 96 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-turn / follow-up persona eval.")
    parser.add_argument("--k", type=int, default=settings.RAG_VOICE_TOP_K)
    parser.add_argument("--model", default=settings.LLM_MODEL)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--rewrite", action="store_true",
                        help="resolve references into a standalone query before retrieval (the candidate fix)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N conversations")
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    convos = [json.loads(line) for line in _GOLDEN_PATH.read_text().splitlines() if line.strip()]
    if args.limit:
        convos = convos[: args.limit]
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    judged = not args.no_judge
    mode = "query-rewrite (standalone)" if args.rewrite else "raw last-message (production)"

    print(f"Evaluating {len(convos)} conversations  |  k={args.k}  mode={mode}  "
          f"judge={'off' if args.no_judge else args.judge_model}")

    turns: list[TurnResult] = []
    for i, convo in enumerate(convos, 1):
        print(f"  [{i}/{len(convos)}] {convo['id']}…", end="\r", flush=True)
        turns.extend(_run_conversation(client, args.model, convo, args.k, args.rewrite))

    if judged:
        print("\n  Judging…                          ")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda t: _judge(client, args.judge_model, t), turns))

    summary = _summarize(turns, judged)
    _print_report(turns, summary, judged, mode)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "rewrite" if args.rewrite else "raw"
    out_path = out_dir / f"multiturn_{tag}_{ts}.json"
    out_path.write_text(json.dumps({
        "meta": {
            "timestamp": ts, "k": args.k, "answer_model": args.model,
            "judge_model": None if args.no_judge else args.judge_model,
            "retrieval_mode": mode, "relevance_floor": settings.RAG_RELEVANCE_FLOOR,
        },
        "summary": summary,
        "turns": [asdict(t) for t in turns],
    }, indent=2))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
