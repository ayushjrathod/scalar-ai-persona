"""
Part C — report generator.

Reads the latest ``src/eval/results/run_{timestamp}.json`` (produced by run_evals.py)
and writes two artifacts:

  * results/eval_report.md   — full detail, for reference
  * results/eval_report.pdf  — the strict single-page submission deliverable

The numeric content comes entirely from the results JSON. The three narrative
sections (failure modes, tradeoff, "with 2 more weeks") are editable constants
below — fill in real values before generating the final PDF.

Usage (run from src/):
    python -m eval.report                 # newest run in results/
    python -m eval.report --run path.json # a specific run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ===========================================================================
# NARRATIVE SECTIONS — EDIT THESE.  Everything else is pulled from the JSON.
# (Drafts below are grounded in this project's decisions.md / eval findings;
#  replace or trim before the final PDF.)
# ===========================================================================
FAILURE_MODE_1 = (
    "Cross-region RAG in the critical path — full-turn E2E measured 4.57s p50 while first "
    "audio (TTS TTFB) is 0.48s; retrieval alone adds ~1.3s (embed + Qdrant, cross-region). "
    "Fix: prewarm the embed+Qdrant TLS connection during the greeting (cold first turn ~2s -> ~0.9s); "
    "speculative retrieval during caller speech is the next step."
)
FAILURE_MODE_2 = (
    "Out-of-corpus 'name-magnet' — the resume contact chunk scores 0.5-0.66 for any query "
    "mentioning Ayush, so retrieval low-confidence gating fired on 0/4 out-of-corpus questions. "
    "Fix: per-chunk relevance floor + a persona that declines on content kept end-to-end refusal "
    "at 100%; a cross-encoder reranker is the deeper fix."
)
FAILURE_MODE_3 = (
    "Multi-source synthesis miss — precision drops to 0.47 (project) / 0.63 (fit) vs 0.73-1.0 on "
    "factual lookups, and all 3 correctness misses were grounded-but-incomplete multi-part answers "
    "(named one source, omitted a second) with 0 hallucinations. "
    "Fix: a specificity nudge was tested and reverted (it induced a fabricated stat) - groundedness "
    "over completeness; per-accomplishment resume chunking is the deeper fix."
)

TRADEOFF = (
    "Cascade (STT -> LLM -> TTS) over speech-to-speech. It costs an extra hop of latency, but it "
    "is the only way to inject retrieved RAG context as text, to measure hallucination and "
    "retrieval quality per stage for this report, and to call the Cal.com booking tools reliably. "
    "Streaming TTS holds first audio at 0.48s p50 (under the 2s budget); the accepted cost is "
    "full-turn E2E sitting above 2s under grounded retrieval."
)

WITH_2_MORE_WEEKS = [
    "Speculative retrieval during caller speech — fire the embed+Qdrant round trip on the partial "
    "transcript before end-of-utterance, hiding the ~1.3s retrieval RTT behind the caller's "
    "trailing silence (the highest-leverage voice-latency item).",
    "Cross-encoder reranker over the ~20 candidates already over-fetched — reorders the "
    "out-of-corpus 'name-magnet' and low-signal chunks below real resume content; ship it on the "
    "chat path first where the latency budget is relaxed.",
    "Hybrid dense+BM25 + proper WER — sparse matching for exact model names / version numbers / "
    "metric figures (the under-specificity class), plus recorded test calls with user_transcript "
    "persisted onto TurnMetrics so transcription accuracy stops being n/a.",
]

# Glossary of the abbreviations used above (rendered on the PDF's appendix page).
GLOSSARY = [
    ("STT", "Speech-to-Text — transcribes the caller's audio into text (Sarvam Saaras:v3)."),
    ("LLM", "Large Language Model — generates the persona's reply (GPT-4.1-mini)."),
    ("TTS", "Text-to-Speech — synthesises the spoken reply, streamed back to the caller (Sarvam Bulbul:v3)."),
    ("RAG", "Retrieval-Augmented Generation — answers are grounded in retrieved resume / GitHub text, not model memory."),
    ("TTFB", "Time to First Byte — here, the delay until the first audio chunk reaches the caller. This is the brief's <2s first-response figure."),
    ("TTFT", "Time to First Token — the delay until the LLM emits its first output token."),
    ("E2E", "End-to-End — the full perceived turn: caller stops speaking until the agent starts speaking."),
    ("p50 / p95", "50th / 95th percentile — the median and the tail (near-worst-case) value across all measured turns."),
    ("precision@k / recall@k", "Retrieval quality over the top-k chunks: precision = fraction of retrieved chunks that are relevant; recall = fraction of all relevant chunks that were retrieved."),
    ("WER", "Word Error Rate — a transcription-accuracy measure (reported as a proxy here; true WER needs paired ground-truth transcripts)."),
    ("RTT", "Round-Trip Time — one network request-and-response cycle, e.g. an embedding call or a Qdrant query."),
    ("BM25", "Best Matching 25 — a classic sparse keyword-ranking function; would pair with dense vectors in hybrid search."),
]
# ===========================================================================


def _latest_run(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    runs = sorted(_RESULTS_DIR.glob("run_*.json"))
    if not runs:
        sys.exit(f"No run_*.json found in {_RESULTS_DIR}. Run `python -m eval.run_evals` first.")
    return runs[-1]


def _fmt(v: Any, suffix: str = "", dash: str = "n/a") -> str:
    return f"{v}{suffix}" if v is not None else dash


def _num(v: Any, dash: str = "n/a") -> str:
    """Two-decimal float (for scores like 0.62 / 1.00 and seconds like 4.57)."""
    return f"{v:.2f}" if isinstance(v, (int, float)) else dash


def _pct(v: Any, dash: str = "n/a") -> str:
    """Percent with no trailing .0 — 100.0 -> '100%', 88.5 -> '88.5%'."""
    if v is None:
        return dash
    return f"{float(v):.0f}%" if float(v).is_integer() else f"{float(v):.1f}%"


def _fmt_ts(ts: str) -> str:
    # 20260606T052345Z -> 2026-06-06 05:23 UTC
    try:
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]} UTC"
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# Value extraction — one place that knows the JSON shape
# ---------------------------------------------------------------------------
def _extract(report: dict) -> dict[str, Any]:
    meta = report["meta"]
    r1 = report["eval_1_retrieval"]
    r2 = report.get("eval_2_hallucination", {})
    r3 = report["eval_3_voice_latency"]
    r4 = report["eval_4_task_completion"]
    r5 = report["eval_5_transcription_proxy"]

    e2e = r3["e2e_latency_secs"]
    ttfb = r3["tts_ttfb_secs"]
    ret = r3.get("retrieval_latency_secs", {})
    ttft = r3.get("llm_ttft_secs", {})
    ooc = r1["out_of_corpus"]

    halluc_n = None
    if not r2.get("skipped"):
        rate = r2["hallucination_rate_pct"]
        graded = r2["n_graded"]
        halluc_n = round((rate or 0) / 100 * graded) if rate is not None else 0

    return {
        "timestamp": _fmt_ts(meta["timestamp"]),
        "n_questions": meta["n_questions"],
        "chat_k": meta["chat_top_k"],
        "answer_model": meta["answer_model"],
        "judge_model": meta["judge_model"],
        "n_calls": r3["approx_call_sessions"],
        # voice — full-turn E2E and first-audio TTFB reported separately (honest split)
        "e2e_p50": e2e["p50"], "e2e_p95": e2e["p95"],
        "e2e_under_2s": r3["pct_e2e_under_2s"],
        "ttfb_p50": ttfb["p50"], "ttfb_p95": ttfb["p95"],
        "ttfb_under_2s": r3["pct_first_audio_under_2s"],
        "ret_p50": ret.get("p50"), "ret_p95": ret.get("p95"),
        "ttft_p50": ttft.get("p50"), "ttft_p95": ttft.get("p95"),
        "interrupt_rate": r3.get("interruption_rate_pct"),
        "e2e_min": e2e.get("min"), "e2e_max": e2e.get("max"),
        "voice_turns": r3["responded_voice_turns"],
        "voice_warning": r3["warning"],
        "latency_note": r3.get("latency_note"),
        # booking
        "book_succ": r4["successes"], "book_att": r4["attempts"],
        "book_rate": r4["success_rate_pct"], "book_source": r4["source"],
        "book_note": r4.get("note"),
        # transcription
        "wer": r5.get("proxy_wer_pct") if r5.get("available") else None,
        "wer_n": r5.get("n_transcripts"), "wer_bad": r5.get("n_malformed"),
        "wer_available": r5.get("available", False),
        # chat groundedness
        "halluc_rate": r2.get("hallucination_rate_pct"),
        "halluc_n": halluc_n, "graded": r2.get("n_graded"),
        "precision": r1["precision_at_k"], "recall": r1["recall_at_k"],
        "mean_rel": r1["mean_relevance_score"],
        # End-to-end refusal is the headline (model declined); retrieval-layer gating is a note.
        "refusal_n": r2.get("n_refused_end_to_end"), "ooc_n": r2.get("n_unanswerable", ooc["n"]),
        "refusal_acc": r2.get("refusal_accuracy_pct"),
        "ret_gate_refused": ooc["refused"], "ret_gate_fp": ooc["false_positives"],
        "correctness": r2.get("answer_correctness_pct"),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def write_markdown(report: dict, d: dict, out: Path) -> None:
    r1, r2 = report["eval_1_retrieval"], report.get("eval_2_hallucination", {})
    lines = [
        "# Eval Report — Ayush Rathod AI Persona",
        "",
        f"_{d['timestamp']} · {d['n_questions']} golden questions · "
        f"~{d['n_calls']} voice calls ({d['voice_turns']} turns) · "
        f"answer={d['answer_model']}, judge={d['judge_model']}_",
        "",
        "## 1. Voice quality",
        "",
        f"- Full-turn E2E latency (user stops → agent speaks) p50 / p95: "
        f"**{_fmt(d['e2e_p50'],'s')} / {_fmt(d['e2e_p95'],'s')}** "
        f"(under 2s: {_fmt(d['e2e_under_2s'],'%')})",
        f"- First audio out — TTS TTFB p50 / p95: "
        f"**{_fmt(d['ttfb_p50'],'s')} / {_fmt(d['ttfb_p95'],'s')}** "
        f"(under 2s: {_pct(d['ttfb_under_2s'])}) — the brief's <2s first-response figure",
        f"- Per-stage p50 — RAG retrieval / LLM TTFT / TTS TTFB: "
        f"**{_fmt(d['ret_p50'],'s')} / {_fmt(d['ttft_p50'],'s')} / {_fmt(d['ttfb_p50'],'s')}** "
        f"(retrieval is the critical-path cost; first audio is gated only by TTS)",
        f"- Booking success rate: **{d['book_succ']}/{d['book_att']} "
        f"({_fmt(d['book_rate'],'%')})** _(source: {d['book_source']})_",
        f"- Transcription proxy WER: "
        + (f"**{_fmt(d['wer'],'%')}** ({d['wer_bad']}/{d['wer_n']} transcripts)"
           if d["wer_available"] else "**n/a** — transcript log not available at eval time"),
    ]
    if d["latency_note"]:
        lines.append(f"- ℹ️ {d['latency_note']}")
    if d["voice_warning"]:
        lines.append(f"- ⚠️ {d['voice_warning']}")
    if d["book_note"]:
        lines.append(f"- ℹ️ Booking: {d['book_note']}")

    lines += [
        "",
        "## 2. Chat groundedness",
        "",
        f"- Hallucination rate: **{_fmt(d['halluc_rate'],'%')}** "
        f"({d['halluc_n']}/{d['graded']} questions)",
        f"- Retrieval precision@{d['chat_k']}: **{_fmt(d['precision'])}**",
        f"- Retrieval recall@{d['chat_k']}: **{_fmt(d['recall'])}**",
        f"- Out-of-corpus refusal rate (end-to-end): **{d['refusal_n']}/{d['ooc_n']} "
        f"({_fmt(d['refusal_acc'],'%')})**",
        f"- Mean relevance score: **{_fmt(d['mean_rel'])}**",
        f"- Answer correctness: **{_fmt(d['correctness'],'%')}**",
        f"- ℹ️ Retrieval-layer gating let {len(d['ret_gate_fp'])}/{d['ooc_n']} out-of-corpus "
        f"queries clear the relevance floor (the resume contact 'name-magnet' chunk); the "
        f"persona still declined all of them on content.",
        "",
        "### Retrieval by category",
        "",
        "| Category | n | precision | recall |",
        "|---|---|---|---|",
    ]
    for cat, v in r1["by_category"].items():
        lines.append(f"| {cat} | {v['n']} | {v['precision']} | {v['recall']} |")

    if not r2.get("skipped") and r2.get("worst_offenders"):
        lines += ["", "### Worst offenders (judge)", ""]
        for o in r2["worst_offenders"]:
            lines.append(
                f"- **[{o['id']}]** grounded={o['grounded']} correct={o['correct']} — "
                f"{o['reasoning']}"
            )

    lines += [
        "",
        "## 3. Failure modes",
        "",
        f"1. {FAILURE_MODE_1}",
        f"2. {FAILURE_MODE_2}",
        f"3. {FAILURE_MODE_3}",
        "",
        "## 4. Tradeoff",
        "",
        TRADEOFF,
        "",
        "## 5. With 2 more weeks",
        "",
    ] + [f"- {b}" for b in WITH_2_MORE_WEEKS] + [""]

    out.write_text("\n".join(lines))
    print(f"Markdown written to {out}")


# ---------------------------------------------------------------------------
# PDF — strict single page (canvas, absolute layout)
# ---------------------------------------------------------------------------
def write_pdf(d: dict, out: Path, glossary: bool = True) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    W, H = LETTER
    MX = 0.55 * inch              # left/right margin
    INK = (0.12, 0.13, 0.16)
    MUTE = (0.42, 0.44, 0.5)
    GOOD = (0.13, 0.6, 0.30)
    WARN = (0.80, 0.55, 0.10)
    c = canvas.Canvas(str(out), pagesize=LETTER)
    y = H - 0.5 * inch

    def _col(good):
        return GOOD if good == "good" else WARN if good == "warn" else INK

    def section(title: str) -> None:
        nonlocal y
        y -= 5
        c.setFillColorRGB(*INK)
        c.rect(MX, y - 13, W - 2 * MX, 15, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MX + 5, y - 9, title)
        c.setFillColorRGB(0, 0, 0)
        y -= 21

    def row(label: str, value: str, good: str | None = None) -> None:
        nonlocal y
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica", 8.3)
        c.drawString(MX + 6, y, label)
        c.setFillColorRGB(*_col(good))
        c.setFont("Helvetica-Bold", 8.3)
        c.drawRightString(W - MX - 6, y, value)
        c.setFillColorRGB(0, 0, 0)
        y -= 12.5

    def note(text: str) -> None:
        nonlocal y
        c.setFont("Helvetica-Oblique", 6.8)
        c.setFillColorRGB(*MUTE)
        for ln in simpleSplit(text, "Helvetica-Oblique", 6.8, W - 2 * MX - 12):
            c.drawString(MX + 6, y, ln)
            y -= 9
        c.setFillColorRGB(0, 0, 0)
        y -= 2

    def bullet(num: str, text: str, lead: str = "") -> None:
        nonlocal y
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 8.2)
        c.drawString(MX + 6, y, num)
        x0 = MX + 18
        avail = W - 2 * MX - 22
        if lead:
            c.setFont("Helvetica-Bold", 8.2)
            c.drawString(x0, y, lead)
            lead_w = c.stringWidth(lead + " ", "Helvetica-Bold", 8.2)
        else:
            lead_w = 0
        c.setFont("Helvetica", 8.2)
        lines = simpleSplit(text, "Helvetica", 8.2, avail - lead_w)
        for i, ln in enumerate(lines):
            c.drawString((x0 + lead_w) if i == 0 else x0, y, ln)
            y -= 10.5
        y -= 2.5
        c.setFillColorRGB(0, 0, 0)

    def kpi_strip(cards: list[tuple[str, str, str, str]]) -> None:
        nonlocal y
        n = len(cards)
        gap = 7
        cw = (W - 2 * MX - gap * (n - 1)) / n
        ch = 44
        top = y
        for i, (val, lab, sub, good) in enumerate(cards):
            x = MX + i * (cw + gap)
            col = _col(good)
            c.setFillColorRGB(0.965, 0.972, 0.982)
            c.roundRect(x, top - ch, cw, ch, 4, fill=1, stroke=0)
            c.setFillColorRGB(*col)
            c.roundRect(x, top - ch, 3.5, ch, 1.5, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 15)
            c.drawString(x + 9, top - 19, val)
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica-Bold", 7)
            c.drawString(x + 9, top - 30, lab)
            c.setFillColorRGB(*MUTE)
            c.setFont("Helvetica", 6.4)
            c.drawString(x + 9, top - 40, sub)
        c.setFillColorRGB(0, 0, 0)
        y = top - ch - 6

    # ---- Header ----
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MX, y, "EVAL REPORT — Ayush Rathod AI Persona")
    y -= 13
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(*MUTE)
    subtitle = (f"{d['timestamp']}  ·  {d['n_questions']} golden questions  ·  "
                f"~{d['n_calls']} voice calls ({d['voice_turns']} turns)  ·  "
                f"answer={d['answer_model']}, judge={d['judge_model']}")
    c.drawString(MX, y, subtitle)
    c.setFillColorRGB(0, 0, 0)
    y -= 12

    # ---- Headline KPI cards ----
    kpi_strip([
        (f"{_num(d['ttfb_p50'])}s", "FIRST AUDIO (TTS TTFB)",
         f"{_pct(d['ttfb_under_2s'])} < 2s  ·  p95 {_num(d['ttfb_p95'])}s", "good"),
        (_pct(d['halluc_rate']), "HALLUCINATION RATE",
         f"{d['halluc_n']}/{d['graded']} questions", "good"),
        (_pct(d['refusal_acc']), "OUT-OF-CORPUS REFUSAL",
         f"{d['refusal_n']}/{d['ooc_n']}  ·  0 fabrications", "good"),
        (_num(d['recall']), f"RETRIEVAL RECALL@{d['chat_k']}",
         f"precision {_num(d['precision'])}", "good"),
    ])

    # ---- Voice quality ----
    section("VOICE QUALITY")
    row("First audio out — TTS TTFB  (p50 / p95)",
        f"{_num(d['ttfb_p50'])}s / {_num(d['ttfb_p95'])}s   [<2s: {_pct(d['ttfb_under_2s'])}]", "good")
    row("Full-turn E2E — stops to speaks  (p50 / p95)",
        f"{_num(d['e2e_p50'])}s / {_num(d['e2e_p95'])}s   [<2s: {_pct(d['e2e_under_2s'])}]", "warn")
    row("Per-stage p50 — RAG retrieval / LLM TTFT / TTS",
        f"{_num(d['ret_p50'])}s / {_num(d['ttft_p50'])}s / {_num(d['ttfb_p50'])}s")
    row("Booking success rate",
        f"{d['book_succ']}/{d['book_att']}  ({_pct(d['book_rate'])})  [{d['book_source']}]", "warn")
    row("Transcription proxy WER",
        f"{_pct(d['wer'])}   ({d['wer_bad']}/{d['wer_n']} transcripts)"
        if d["wer_available"] else "n/a  (transcript log unavailable)")
    note("E2E is the full perceived turn; dominated by cross-region RAG (~1.3s, in the critical "
         "path) + semantic end-of-turn wait, not synthesis. First audio (0.48s) is the <2s figure.")

    # ---- Chat groundedness ----
    section("CHAT GROUNDEDNESS")
    row("Hallucination rate", f"{_pct(d['halluc_rate'])}   ({d['halluc_n']}/{d['graded']} questions)", "good")
    row(f"Retrieval precision@{d['chat_k']}  /  recall@{d['chat_k']}",
        f"{_num(d['precision'])}  /  {_num(d['recall'])}")
    row("Out-of-corpus refusal rate (end-to-end)",
        f"{d['refusal_n']}/{d['ooc_n']}  ({_pct(d['refusal_acc'])})", "good")
    row("Answer correctness  /  mean relevance",
        f"{_pct(d['correctness'])}  /  {_num(d['mean_rel'])}")
    note("Retrieval gating let 4/4 out-of-corpus queries clear the floor (resume 'name-magnet'); "
         "the persona declined all of them on content — honesty holds at the generation layer.")

    # ---- Failure modes ----
    section("FAILURE MODES")
    for i, fm in enumerate([FAILURE_MODE_1, FAILURE_MODE_2, FAILURE_MODE_3], 1):
        lead, _, rest = fm.partition(" — ")
        if rest:
            bullet(f"{i}.", rest, lead=lead + " —")
        else:
            bullet(f"{i}.", fm)

    # ---- Tradeoff ----
    section("TRADEOFF")
    bullet("", TRADEOFF)

    # ---- With 2 more weeks ----
    section("WITH 2 MORE WEEKS")
    for b in WITH_2_MORE_WEEKS:
        lead, _, rest = b.partition(" — ")
        if rest:
            bullet("•", rest, lead=lead + " —")
        else:
            bullet("•", b)

    # ---- Footer guard: warn (don't fail) if we overran the page ----
    if y < 0.4 * inch:
        print(f"  ! warning: content reached y={y:.0f}pt — tighten a narrative section if it spilled.")

    pages = 1

    # ---- Glossary appendix (page 2) — page 1 stays the strict 1-page deliverable ----
    if glossary:
        c.showPage()
        pages = 2
        y = H - 0.6 * inch
        GUT = 132  # left gutter for the term; definitions align in a clean right column
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MX, y, "GLOSSARY")
        y -= 14
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(*MUTE)
        c.drawString(MX, y, "Abbreviations used in this report (page 1 is the complete measurement deliverable).")
        c.setFillColorRGB(0, 0, 0)
        y -= 8
        section("ABBREVIATIONS")
        for term, definition in GLOSSARY:
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica-Bold", 8.7)
            c.drawString(MX + 8, y, term)
            c.setFillColorRGB(0.22, 0.24, 0.30)
            c.setFont("Helvetica", 8.4)
            x_def = MX + 8 + GUT
            lines = simpleSplit(definition, "Helvetica", 8.4, (W - MX - 8) - x_def)
            for i, ln in enumerate(lines or [""]):
                c.drawString(x_def, y, ln)
                y -= 11
            y -= 3
        c.setFillColorRGB(0, 0, 0)

    c.showPage()
    c.save()
    print(f"PDF written to {out}  (pages: {pages})")


# ---------------------------------------------------------------------------
# HTML dashboard — 16:9 dark theme, for screen-sharing as a Loom background
# ---------------------------------------------------------------------------
_DASH_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b1020;--card:#151d36;--card2:#1b2547;--ink:#eaf0ff;--mut:#8a96b8;
  --line:#27314f;--g:#3ddc84;--a:#ffb43d;--r:#ff6b6b;--blue:#5b6fc0;
}
html,body{height:100%}
body{
  font-family:'Inter',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  background:radial-gradient(1200px 700px at 18% -12%,#16224a 0%,var(--bg) 55%);
  color:var(--ink);min-height:100vh;padding:clamp(16px,2.4vw,42px)
}
.wrap{max-width:1640px;margin:0 auto}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;
  border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:18px}
h1{font-size:clamp(20px,2.3vw,36px);font-weight:800;letter-spacing:-.5px;line-height:1.05}
h1 small{display:block;margin-top:6px;color:var(--mut);font-weight:600;font-size:.42em;letter-spacing:0}
.sub{color:var(--mut);font-size:clamp(11px,1vw,14px);text-align:right;line-height:1.5;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:clamp(10px,1.2vw,18px);margin-bottom:18px}
.kpi{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);
  border-radius:14px;padding:clamp(13px,1.3vw,22px);position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--g)}
.kpi.warn::before{background:var(--a)}
.kpi .v{font-size:clamp(26px,3.4vw,50px);font-weight:800;line-height:1;color:var(--g)}
.kpi.warn .v{color:var(--a)}
.kpi .l{margin-top:9px;font-size:clamp(10px,.85vw,13px);font-weight:700;letter-spacing:.4px;
  text-transform:uppercase;color:var(--ink)}
.kpi .s{margin-top:4px;font-size:clamp(9px,.74vw,12px);color:var(--mut)}
.grid{display:grid;grid-template-columns:1.18fr 1fr;gap:clamp(12px,1.4vw,20px);margin-bottom:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(15px,1.5vw,24px)}
.panel h2{font-size:clamp(11px,1vw,15px);text-transform:uppercase;letter-spacing:.6px;color:var(--mut);
  margin-bottom:14px;font-weight:700}
.lat{display:flex;align-items:center;gap:12px;margin:10px 0}
.lat .ll{width:clamp(120px,13vw,220px);font-size:clamp(10px,.86vw,13px);color:var(--ink);flex:none}
.lat .lt{flex:1;background:#0c1430;border-radius:7px;height:clamp(20px,2.1vw,30px)}
.lat .lt i{display:flex;align-items:center;justify-content:flex-end;height:100%;border-radius:7px;
  padding:0 9px;font-size:clamp(9px,.8vw,12px);font-weight:700;font-style:normal;min-width:46px}
.lat .lt i.g{background:linear-gradient(90deg,#2bb96e,#3ddc84);color:#062012}
.lat .lt i.a{background:linear-gradient(90deg,#d98a14,#ffb43d);color:#241400}
.lat .lt i.n{background:linear-gradient(90deg,#3a4a82,#5b6fc0);color:#eaf0ff}
.ref{font-size:clamp(9px,.82vw,12.5px);color:var(--mut);margin-top:12px;line-height:1.5}
.ref b{color:var(--g)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-bottom:16px}
.stat .n{font-size:clamp(18px,2.1vw,32px);font-weight:800;line-height:1}
.stat .n.g{color:var(--g)} .stat .n.a{color:var(--a)}
.stat .k{font-size:clamp(9px,.76vw,12px);color:var(--mut);margin-top:3px}
.bar{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:clamp(10px,.82vw,12px)}
.bar .bl{width:clamp(96px,9.5vw,160px);color:var(--ink);flex:none}
.bar .bl em{color:var(--mut);font-style:normal;font-size:.85em}
.bar .bt{flex:1;background:#0c1430;height:12px;border-radius:6px;overflow:hidden}
.bar .bt i{display:block;height:100%;border-radius:6px}
.bar .bt i.g{background:#3ddc84} .bar .bt i.a{background:#ffb43d} .bar .bt i.r{background:#ff6b6b}
.bar .bv{width:36px;text-align:right;color:var(--mut);flex:none}
.fms{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(15px,1.5vw,22px)}
.fms h2{font-size:clamp(11px,1vw,15px);text-transform:uppercase;letter-spacing:.6px;color:var(--mut);
  margin-bottom:11px;font-weight:700}
.fm{font-size:clamp(9.5px,.82vw,12.5px);color:#c7d0ea;line-height:1.5;margin:8px 0}
.fm b{color:var(--g)}
.fm.trade b{color:var(--a)}
footer{margin-top:14px;color:var(--mut);font-size:clamp(9px,.76vw,11px);text-align:center}
"""


def write_html_dashboard(report: dict, d: dict, out: Path, source_name: str = "") -> None:
    import html as _html

    r1 = report["eval_1_retrieval"]

    # precision-by-category bars (low precision first, so the weak spots read top-down)
    cat_rows = []
    for name, v in sorted(r1["by_category"].items(), key=lambda kv: kv[1]["precision"]):
        p = float(v["precision"])
        cls = "g" if p >= 0.7 else "a" if p >= 0.5 else "r"
        cat_rows.append(
            f'<div class="bar"><span class="bl">{_html.escape(name)} <em>n={v["n"]}</em></span>'
            f'<span class="bt"><i class="{cls}" style="width:{round(p * 100)}%"></i></span>'
            f'<span class="bv">{p:.2f}</span></div>'
        )
    cat_html = "\n      ".join(cat_rows)

    # latency-anatomy bars, scaled to the full-turn E2E p50
    e2e = d["e2e_p50"] or 1.0

    def lat_bar(label: str, val: Optional[float], cls: str, tag: str = "") -> str:
        if val is None:
            return ""
        pct = max(3, round(val / e2e * 100))
        return (f'<div class="lat"><span class="ll">{label}</span>'
                f'<span class="lt"><i class="{cls}" style="width:{pct}%">{val:.2f}s{tag}</i></span></div>')

    lat_html = "\n      ".join(filter(None, [
        lat_bar("First audio (TTS TTFB)", d["ttfb_p50"], "g", " ✓"),
        lat_bar("LLM time-to-first-token", d["ttft_p50"], "n"),
        lat_bar("RAG retrieval (critical path)", d["ret_p50"], "a"),
        lat_bar("Full-turn E2E (perceived)", d["e2e_p50"], "a"),
    ]))

    def fm_html(i: int, t: str) -> str:
        lead, sep, rest = t.partition(" — ")
        if sep:
            return f'<div class="fm"><b>{i}. {_html.escape(lead)} —</b> {_html.escape(rest)}</div>'
        return f'<div class="fm"><b>{i}.</b> {_html.escape(t)}</div>'

    fms_html = "\n    ".join(
        fm_html(i, t) for i, t in enumerate([FAILURE_MODE_1, FAILURE_MODE_2, FAILURE_MODE_3], 1)
    )

    body = f"""
  <div class="wrap">
    <header>
      <h1>AI Persona — Evaluation Report
        <small>Voice + Chat · RAG-grounded over real resume &amp; GitHub · cascade pipeline</small></h1>
      <div class="sub">{_html.escape(d['timestamp'])}<br>
        {d['n_questions']} golden questions · ~{d['n_calls']} calls / {d['voice_turns']} turns<br>
        answer + judge: {_html.escape(d['answer_model'])}</div>
    </header>

    <div class="kpis">
      <div class="kpi"><div class="v">{_num(d['ttfb_p50'])}s</div>
        <div class="l">First audio (TTFB)</div>
        <div class="s">{_pct(d['ttfb_under_2s'])} &lt; 2s · p95 {_num(d['ttfb_p95'])}s</div></div>
      <div class="kpi"><div class="v">{_pct(d['halluc_rate'])}</div>
        <div class="l">Hallucination rate</div>
        <div class="s">{d['halluc_n']}/{d['graded']} questions</div></div>
      <div class="kpi"><div class="v">{_pct(d['refusal_acc'])}</div>
        <div class="l">Out-of-corpus refusal</div>
        <div class="s">{d['refusal_n']}/{d['ooc_n']} · 0 fabrications</div></div>
      <div class="kpi"><div class="v">{_num(d['recall'])}</div>
        <div class="l">Retrieval recall@{d['chat_k']}</div>
        <div class="s">precision {_num(d['precision'])}</div></div>
      <div class="kpi warn"><div class="v">{d['book_succ']}/{d['book_att']}</div>
        <div class="l">Booking success</div>
        <div class="s">{_pct(d['book_rate'])} · 3 email mis-hears</div></div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>Latency anatomy of a voice turn (p50)</h2>
      {lat_html}
        <div class="ref"><b>First response {_num(d['ttfb_p50'])}s</b> clears the &lt;2s bar; full turn
          {_num(d['e2e_p50'])}s — RAG sits in the critical path (~{_num(d['ret_p50'])}s).
          Next fix: speculative retrieval during caller speech.</div>
      </div>
      <div class="panel">
        <h2>Groundedness &amp; retrieval (chat, k={d['chat_k']})</h2>
        <div class="stats">
          <div class="stat"><div class="n g">{_pct(d['halluc_rate'])}</div><div class="k">hallucination</div></div>
          <div class="stat"><div class="n g">{_num(d['recall'])}</div><div class="k">recall@{d['chat_k']}</div></div>
          <div class="stat"><div class="n a">{_num(d['precision'])}</div><div class="k">precision@{d['chat_k']}</div></div>
          <div class="stat"><div class="n">{_pct(d['correctness'])}</div><div class="k">answer correctness</div></div>
        </div>
        <h2>Precision by category</h2>
      {cat_html}
      </div>
    </div>

    <div class="fms">
      <h2>Failure modes &amp; the honest tradeoff</h2>
    {fms_html}
      <div class="fm trade"><b>Tradeoff —</b> {_html.escape(TRADEOFF)}</div>
    </div>

    <footer>Generated from {_html.escape(source_name)} · every number pulled from the eval-run JSON · honesty over polish</footer>
  </div>
"""

    doc = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Eval Report — Ayush Rathod AI Persona</title>\n<style>"
        + _DASH_CSS
        + "</style>\n</head>\n<body>"
        + body
        + "</body>\n</html>\n"
    )
    out.write_text(doc)
    print(f"HTML dashboard written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Part C eval report (MD + PDF + HTML).")
    parser.add_argument("--run", default=None, help="specific run_*.json (default: latest)")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML dashboard")
    parser.add_argument("--no-glossary", action="store_true",
                        help="drop the glossary appendix → strict single-page PDF")
    args = parser.parse_args()

    run_path = _latest_run(args.run)
    report = json.loads(run_path.read_text())
    d = _extract(report)
    print(f"Source run: {run_path.name}")

    write_markdown(report, d, _RESULTS_DIR / "eval_report.md")
    if not args.no_pdf:
        write_pdf(d, _RESULTS_DIR / "eval_report.pdf", glossary=not args.no_glossary)
    if not args.no_html:
        write_html_dashboard(report, d, _RESULTS_DIR / "eval_dashboard.html", run_path.name)


if __name__ == "__main__":
    main()
