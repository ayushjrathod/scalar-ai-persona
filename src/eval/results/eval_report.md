# Eval Report — Ayush Rathod AI Persona

_2026-06-06 10:55 UTC · 30 golden questions · ~9 voice calls (32 turns) · answer=gpt-4.1-mini, judge=gpt-4.1-mini_

## 1. Voice quality

- Full-turn E2E latency (user stops → agent speaks) p50 / p95: **4.572s / 8.143s** (under 2s: 0.0%)
- First audio out — TTS TTFB p50 / p95: **0.478s / 1.033s** (under 2s: 100%) — the brief's <2s first-response figure
- Per-stage p50 — RAG retrieval / LLM TTFT / TTS TTFB: **1.297s / 0.884s / 0.478s** (retrieval is the critical-path cost; first audio is gated only by TTS)
- Booking success rate: **7/10 (70.0%)** _(source: manual)_
- Transcription proxy WER: **n/a** — transcript log not available at eval time
- ℹ️ E2E is the full perceived turn (user stops -> agent speaks); it is dominated by cross-region RAG retrieval (~1.3s, in the critical path) and the semantic end-of-turn wait, not by synthesis. First audio out (TTS TTFB) is the <2s first-response figure.
- ℹ️ Booking: 7/10 confirmed bookings. The 3 failures were email mis-transcription - Sarvam STT mis-heard the caller's spelled-out email, so the Cal.com confirmation would reach a wrong address. Ties to the transcription-accuracy finding; an email read-back confirmation step is the mitigation.

## 2. Chat groundedness

- Hallucination rate: **0.0%** (0/30 questions)
- Retrieval precision@6: **0.616**
- Retrieval recall@6: **1.0**
- Out-of-corpus refusal rate (end-to-end): **4/4 (100.0%)**
- Mean relevance score: **0.612**
- Answer correctness: **88.5%**
- ℹ️ Retrieval-layer gating let 4/4 out-of-corpus queries clear the relevance floor (the resume contact 'name-magnet' chunk); the persona still declined all of them on content.

### Retrieval by category

| Category | n | precision | recall |
|---|---|---|---|
| achievements | 1 | 1.0 | 1.0 |
| contact | 1 | 0.5 | 1.0 |
| education | 1 | 1.0 | 1.0 |
| fit | 4 | 0.625 | 1.0 |
| project | 12 | 0.472 | 1.0 |
| skills | 2 | 0.834 | 1.0 |
| work | 5 | 0.733 | 1.0 |

### Worst offenders (judge)

- **[fit-evals]** grounded=True correct=False — The answer mentions only the ValueX evaluation suite but omits the key fact that Ayush also built an LLM-as-judge evaluation pipeline at DevDynamics, which is central to the gold reference.
- **[fit-production]** grounded=True correct=False — The answer omits key details about the production systems Ayush shipped, such as the MCP server on AWS ECS, the two-tier AST cache on AWS Lambda, and FastAPI services with Celery/Redis and Docker depl
- **[proj-valuex-decisions]** grounded=True correct=False — The answer mentions the SSE design choice and some system goals but omits key design decisions like the two-phase architecture, the raised safety guard threshold, the single structured-output LLM call

## 3. Failure modes

1. Cross-region RAG in the critical path — full-turn E2E measured 4.57s p50 while first audio (TTS TTFB) is 0.48s; retrieval alone adds ~1.3s (embed + Qdrant, cross-region). Fix: prewarm the embed+Qdrant TLS connection during the greeting (cold first turn ~2s -> ~0.9s); speculative retrieval during caller speech is the next step.
2. Out-of-corpus 'name-magnet' — the resume contact chunk scores 0.5-0.66 for any query mentioning Ayush, so retrieval low-confidence gating fired on 0/4 out-of-corpus questions. Fix: per-chunk relevance floor + a persona that declines on content kept end-to-end refusal at 100%; a cross-encoder reranker is the deeper fix.
3. Multi-source synthesis miss — precision drops to 0.47 (project) / 0.63 (fit) vs 0.73-1.0 on factual lookups, and all 3 correctness misses were grounded-but-incomplete multi-part answers (named one source, omitted a second) with 0 hallucinations. Fix: a specificity nudge was tested and reverted (it induced a fabricated stat) - groundedness over completeness; per-accomplishment resume chunking is the deeper fix.

## 4. Tradeoff

Cascade (STT -> LLM -> TTS) over speech-to-speech. It costs an extra hop of latency, but it is the only way to inject retrieved RAG context as text, to measure hallucination and retrieval quality per stage for this report, and to call the Cal.com booking tools reliably. Streaming TTS holds first audio at 0.48s p50 (under the 2s budget); the accepted cost is full-turn E2E sitting above 2s under grounded retrieval.

## 5. With 2 more weeks

- Speculative retrieval during caller speech — fire the embed+Qdrant round trip on the partial transcript before end-of-utterance, hiding the ~1.3s retrieval RTT behind the caller's trailing silence (the highest-leverage voice-latency item).
- Cross-encoder reranker over the ~20 candidates already over-fetched — reorders the out-of-corpus 'name-magnet' and low-signal chunks below real resume content; ship it on the chat path first where the latency budget is relaxed.
- Hybrid dense+BM25 + proper WER — sparse matching for exact model names / version numbers / metric figures (the under-specificity class), plus recorded test calls with user_transcript persisted onto TurnMetrics so transcription accuracy stops being n/a.
