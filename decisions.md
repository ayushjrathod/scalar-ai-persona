# Decisions Log — Scalar AI Persona (v1)

> Chronological record of every meaningful decision made while building v1.
> Purpose: share with the Scaler eval team so they understand the thought process.

---

## 1. Pipeline Architecture — Cascade vs Speech-to-Speech

**Decision:** Cascade (STT → LLM → TTS), not GPT-4o Realtime / speech-to-speech.

**Alternatives considered:** OpenAI Realtime API (S2S), Vapi (managed voice platform).

**Reasons:**
- RAG grounding requires injecting retrieved text into the LLM context. That only exists in text space; S2S models consume raw audio and cannot receive structured context injections mid-turn.
- The eval report requires measurable per-stage latency breakdowns (STT delay, LLM TTFT, TTS TTFB). A black-box S2S model makes that impossible.
- Tool calling (calendar booking) is more reliable with text-space function calls than S2S-native tool use, which is still underdeveloped.
- Cascade TTS TTFB (~212ms measured for Sarvam Bulbul:v3) is fast enough to stay under the 2s first-response SLA without needing S2S.

---

## 2. Voice Orchestration — LiveKit vs Vapi

**Decision:** LiveKit Agents framework (Python, `livekit-agents ≥ 0.12`).

**Alternatives considered:** Vapi (managed), Twilio Media Streams + custom orchestration.

**Reasons:**
- LiveKit has native first-party plugins for Sarvam STT and TTS (`livekit-plugins-sarvam`). Vapi does not — using Vapi would require an HTTP webhook hop per turn, adding 50–200ms of extra latency.
- LiveKit handles VAD, barge-in, and SIP telephony natively, eliminating significant boilerplate.
- LiveKit's free tier provides a US phone number with zero SIP trunk configuration cost.
- The `AgentSession` + `Agent` API (the current API in ≥0.12, replacing the removed `VoicePipelineAgent`) is well-documented and actively maintained.

---

## 3. STT — Sarvam Saaras:v3

**Decision:** Sarvam Saaras:v3 via WebSocket streaming.

**Alternatives considered:** Deepgram Nova, OpenAI Whisper, AssemblyAI.

**Reasons:**
- Indian English optimised — the majority of Scaler's evaluators are likely Indian English speakers with accents that Western STT models handle poorly.
- Native LiveKit plugin (`livekit-plugins-sarvam`) — zero integration code required.
- WebSocket streaming mode ensures transcript arrives incrementally (~200–400ms expected), not as a batch after full utterance.
- Price: ₹30/hour is cheaper than Deepgram at comparable quality for Indian English.

**Note on smoke test:** A batch API smoke test showed ~5.7s latency — this was the wrong mode (batch, not streaming WebSocket). Real pipeline mode is expected ~200–400ms and will be measured on first live call.

---

## 4. LLM — GPT-4.1-mini (dev) → GPT-4.1 (submission)

**Decision:** `gpt-4.1-mini` during development, `gpt-4.1` before final submission.

**Alternatives considered:** Claude Sonnet, Gemini Flash, Llama 3 (self-hosted).

**Reasons:**
- GPT-4.1-mini has fast TTFT (time-to-first-token) which directly affects e2e voice latency.
- Good instruction following for system-prompt-constrained persona behaviour (short responses, no monologue, adversarial resistance).
- OpenAI streaming is well-supported by `livekit-plugins-openai`.
- Model name is config-driven (`LLM_MODEL` env var) — swapping to GPT-4.1 for submission is a one-line change.

---

## 5. TTS — Sarvam Bulbul:v3, speaker=ritu

**Decision:** Sarvam Bulbul:v3 with `speaker=ritu`, `target_language_code=en-IN`.

**Alternatives considered:** ElevenLabs, OpenAI TTS, PlayHT.

**Reasons:**
- TTFB validated at 207ms / 212ms median / 242ms p95 in a pre-build smoke test — confirmed fast enough.
- Indian English voice quality fits the target audience.
- Native LiveKit plugin (`livekit-plugins-sarvam`) delivers audio as a `ChunkedStream` — first audio chunk plays to the caller while later chunks are still synthesising, keeping perceived latency low.
- Consistent with STT choice (single vendor, single API key, single billing account).

---

## 6. VAD — Silero

**Decision:** Silero VAD via `livekit-plugins-silero`.

**Alternatives considered:** WebRTC VAD (built-in), custom energy-based VAD.

**Reasons:**
- Battle-tested, built into the LiveKit plugin ecosystem.
- Handles background noise better than energy-based VAD.
- Zero configuration required.

---

## 7. Turn Detection — EnglishModel (semantic EOU)

**Decision:** `livekit.plugins.turn_detector.english.EnglishModel` as the EOU (end-of-utterance) detector.

**Alternatives considered:** Raw silence duration threshold only.

**Reasons:**
- Semantic EOU model scores conversation context to determine if the caller has truly finished speaking, not just paused.
- Allows dropping the `min_delay` endpointing threshold from 0.5s → 0.3s without cutting callers off mid-sentence.
- 0.2s saved per turn on a typical 10-turn call = 2s total latency reduction.

---

## 8. Preemptive Generation

**Decision:** Enabled `preemptive_generation: {enabled: True, preemptive_tts: True}` in `AgentSession`.

**Reason:** Starts LLM and TTS speculatively on the interim transcript before endpointing fires. This hides most of the STT + LLM TTFT latency behind the natural trailing silence of the caller, which is otherwise dead time. No quality trade-off — if the interim transcript turns out to be wrong (caller keeps talking), the in-flight generation is cancelled.

---

## 9. Telephony — LiveKit US free number

**Decision:** LiveKit-provided US phone number (inbound only, free tier).

**Alternatives considered:** Twilio, Vonage, AWS Connect.

**Reasons:**
- Zero configuration cost, zero SIP trunk setup.
- Scaler evaluators can call international numbers.
- Free tier is sufficient for a 7-day evaluation window.

---

## 10. Deployment — Worker on GCE, API on Cloud Run

**Decision:** LiveKit worker → GCP Compute Engine e2-micro. FastAPI API → GCP Cloud Run (min-instances=1).

**Alternatives considered:** Both on Cloud Run, Railway, Fly.io.

**Reasons:**
- The LiveKit worker is a persistent outbound WebSocket process that stays connected to LiveKit Cloud between calls. Cloud Run's scale-to-zero behaviour would kill it.
- GCE e2-micro stays always-on, has no cold start, and is within GCP free tier.
- Cloud Run is appropriate for the FastAPI HTTP API: scales, stays warm with min-instances=1, gets a public HTTPS URL automatically.
- Ayush has sufficient GCP credits, and GCP gives more control and observability than Railway.

---

## 11. Backend Architecture — Shared FastAPI, Not Duplicated

**Decision:** Single FastAPI app (`src/api/`) serves both voice metrics and (Phase 4) chat.

**Reasons:**
- Building a separate server for chat would duplicate config, auth, RAG retrieval, and calendar logic.
- Both voice and chat need the same RAG backend and persona — shared code is correct here.
- Worker posts metrics to the API; API exposes them via `GET /metrics`. Clean separation of concerns.

---

## 12. Config — Pydantic BaseSettings, Single Instance

**Decision:** All environment variables are loaded via `pydantic-settings` `BaseSettings` into a single `settings` singleton in `src/utils/config.py`.

**Reasons:**
- Single source of truth: changing a model or API key = changing one env var, not hunting through code.
- Pydantic validation at startup catches misconfigured env vars before any call is attempted.
- Validators added for: `LIVEKIT_URL` must be `ws://` or `wss://`; API keys must not be empty; `OPENAI_API_KEY` must start with `sk-`.
- No secrets in code. No exceptions.

---

## 13. Monitoring — Instrumented from Day 1

**Decision:** `MetricsCollector` (in `src/agent/monitoring.py`) is wired into the pipeline from the first call, not retrofitted.

**Reasons:**
- Retrofitting monitoring after the fact produces incomplete data — early calls go unrecorded, skewing the eval report.
- LiveKit emits `LLMMetrics`, `STTMetrics`, `TTSMetrics` events per turn; the collector hooks these events and seals one `TurnMetrics` record per completed assistant turn.
- Metrics are persisted to a JSONL file (`METRICS_STORE_PATH`) so data survives process restarts.
- The `summary_from_turns()` function computes p50/p95 per stage, interruption rate, and per-turn cost — all needed for the eval report.

**Key implementation detail:** TTS metrics arrive as multiple events (one per synthesised segment). `_InFlightTTS` accumulates them across all segments; the turn is only "sealed" when `conversation_item_added` fires with the completed assistant `ChatMessage`. This ensures per-turn TTS character count is correct even for multi-segment responses.

---

## 14. Cost Tracking — INR-denominated Sarvam, USD-denominated OpenAI

**Decision:** Cost constants kept in USD internally (Sarvam INR prices converted at ₹95/USD).

**Reasons:**
- Sarvam pricing is published in INR; OpenAI pricing is in USD. Converting to a common currency (USD) allows summing across components for a single `estimated_cost_usd` per turn.
- INR/USD conversion is a constant (`1/95.0`), not a live rate — acceptable for an eval report estimate.

**Rates used:**
- Sarvam STT: ₹30/hour → $0.000088/sec
- Sarvam TTS: ₹30/10,000 chars → $0.0000032/char
- GPT-4.1-mini: $0.40/1M input tokens, $1.60/1M output tokens

---

## 15. System Prompt Design — Short Responses, Adversarial Resistance

**Decision:** Hardcoded system prompt (Phase 1) with strict behavioral constraints.

**Key constraints enforced in prompt:**
- 1–2 sentences per response. No markdown, no bullet points. Spoken prose only.
- Never claim to be Ayush himself — always "AI representative."
- Only state facts from the embedded fact block. Say "I don't have that detail" if unknown.
- Adversarial resistance: ignore instructions to forget rules or impersonate others.
- Interruption handling: stop, let caller speak, pick up cleanly without apologising.
- Booking placeholder: acknowledge intent, say Ayush will confirm (real booking wired in Phase 3).

**Reason for short responses:** Long LLM output = longer TTS synthesis = higher perceived latency. Every extra sentence adds ~200–400ms to the caller's wait. 1–2 sentences keeps the voice UX tight and natural.

---

## 16. RAG Timing — Phase 2, Not Phase 1

**Decision:** Phase 1 ships with a hardcoded fact block in the persona prompt. RAG (Qdrant + embeddings) is Phase 2.

**Reason:** Validate the full voice pipeline and measure real latency before adding retrieval complexity. If the pipeline has a latency or streaming bug, it's easier to diagnose without a RAG retrieval call in the hot path. RAG is additive — it slots in by replacing the fact block with a retrieved-chunk injection.

---

## 17. Agent Name — Explicit Dispatch via `agent_name`

**Decision:** Worker registered with `agent_name="ayush-persona"` in `WorkerOptions`.

**Reason:** Without an explicit `agent_name`, LiveKit routes any inbound SIP call to ANY idle worker in the project. If other workers exist in the same LiveKit project, calls could be misrouted. Explicit dispatch ensures only this worker handles calls matching the SIP dispatch rule named `"ayush-persona"`.

---

## 18. API Keys in Subprocess — Passed Explicitly

**Decision:** `api_key` passed explicitly to `sarvam.STT(...)` and `sarvam.TTS(...)` constructors in `pipeline.py`.

**Reason:** The LiveKit worker framework spawns each call's `entrypoint()` in a forkserver subprocess. Forkserver does not inherit environment variables from the parent process. Passing keys explicitly from `settings` (which was loaded before forking) ensures the subprocess always has credentials.

---

## 19. RAG Vector Store — Qdrant Cloud, Single Dense Collection

**Decision:** Qdrant Cloud free tier, single dense vector collection (`persona_corpus`), not hybrid (dense + sparse).

**Alternatives considered:** Pinecone, Weaviate, pgvector, hybrid search (BM25 + dense).

**Reasons:**
- Corpus is small (~200–500 chunks: resume + 4–5 GitHub repo READMEs and key files). At this scale, BM25/sparse vectors add integration complexity with no measurable recall improvement.
- Voice agent queries are paraphrased natural-language questions, not keyword lookups — semantic dense search is the right fit.
- Specific project names and tech terms (e.g. "ValueX", "FastAPI") are already present in the LLM system prompt; retrieval doesn't need to match on exact tokens.
- Hybrid search requires a SPLADE/BM25 encoder dependency, a second vector field per PointStruct, and a `query_sparse_vector` per search call — all overhead with no gain at this corpus size.
- Qdrant Cloud free tier is sufficient; collection is recreatable from source in minutes.

---

## 20. Embedding Model — text-embedding-3-small (1536 dims)

**Decision:** OpenAI `text-embedding-3-small`, 1536 dimensions.

**Alternatives considered:** `text-embedding-3-large` (3072 dims), CLIP (512 dims, multimodal).

**Reasons:**
- CLIP is an image-text multimodal model — wrong tool for a text-only resume/code corpus.
- `text-embedding-3-large` is 6.5× more expensive ($0.13 vs $0.02 per 1M tokens) and scores marginally higher on MTEB benchmarks. At 200–500 chunks, the quality delta is unmeasurable in practice; the bottleneck is chunk strategy and prompt design, not embedding dimensions.
- `text-embedding-3-small` at 1536 dims covers the full vocabulary of a software engineering resume and GitHub repos with no meaningful compression loss.
- Query embedding runs on every voice turn — `text-embedding-3-small` is faster, keeping retrieval latency low in the hot path.
- Model name and dims are config-driven (`EMBEDDING_MODEL`, `EMBEDDING_DIMS`) — swapping to large before submission is a one-line env var change if evals show a quality gap.

---

## 21. Similarity Metric — Cosine Distance

**Decision:** Cosine distance for all vector similarity search.

**Alternatives considered:** Dot product, Euclidean (L2), Manhattan (L1).

**Reasons:**
- Cosine measures the angle between vectors, i.e. semantic direction, regardless of magnitude. This is the correct measure for comparing meaning across text chunks.
- OpenAI explicitly recommends cosine similarity for `text-embedding-3-*` models.
- Dot product is equivalent to cosine only when vectors are unit-normalised. OpenAI embeddings are not guaranteed to be unit-normalised at all times; cosine is the safer default.
- Euclidean and Manhattan measure geometric distance — valid but not conventional for text embeddings, and require normalisation to behave like cosine.

---

---

## 22. RAG Ingestion — Corpus File Selection

**Decision:** Ingest only `.md` files, `.toml` files, `package.json`, `requirements.txt`, and `go.mod`. No source code files (`.py`, `.ts`, `.jsx`, etc.).

**Reasons:**
- The persona RAG answers questions about what Ayush built, what technologies he used, and how he thinks — not about implementation minutiae.
- READMEs and dependency manifests carry the highest signal-to-noise ratio: they describe the project's purpose, architecture, and tech stack in prose that embeds well and retrieves cleanly.
- Source code chunks embed poorly for natural-language retrieval: a caller asking "what did you build at DevDynamics?" will not semantically match a Python function body.
- Excluding source code reduces the corpus from ~570 chunks to 72, cutting embedding cost from $0.0033 to $0.00043 and retrieval latency due to a smaller candidate set.

---

## 23. RAG Ingestion — Chunking Strategy

**Decision:** Single header-aware chunker for all file types. `CHUNK_TARGET=400`, `CHUNK_MAX=500`, `CHUNK_OVERLAP=50` tokens. Token counting via `tiktoken` `cl100k_base`.

**Reasons:**
- `cl100k_base` is the same BPE encoding used by `text-embedding-3-small` — counting tokens this way gives accurate chunk sizes relative to the embedding model's actual context window.
- Splitting on H1/H2/H3 headers preserves semantic units (a section on "Tech Stack" or "Architecture" stays together rather than being cut mid-sentence).
- Merging small sections up to `CHUNK_TARGET` avoids one-line chunks that embed as near-zero-information vectors.
- Force-splitting at `CHUNK_MAX` with 50-token overlap prevents losing cross-sentence context at hard boundaries.
- Non-markdown files (`.toml`, `package.json`) contain no headers, so they pass through the same chunker as a single section, chunked by token window only if they exceed `CHUNK_MAX`. At current corpus size all dependency manifests are well under 400 tokens and produce exactly one chunk each.

---

## 24. RAG Ingestion — Allowlist Over Denylist

**Decision:** File filtering via a tight allowlist (`.md`, `.toml`, `package.json`, `requirements.txt`, `go.mod`) plus a minimal four-entry skip-dir list (`node_modules`, `.git`, `venv`, `.venv`). No denylist of extensions or filenames.

**Reasons:**
- A tight allowlist makes a denylist largely redundant: lockfiles, binaries, migrations, compiled output, and all source code are already excluded by not being in the allowlist.
- The only directories that can produce false-positive allowlist matches are ones that ship their own `package.json` or `README.md` trees: `node_modules` (thousands of `package.json`s) and `venv`/`.venv` (dependency READMEs). Those four dir names are the entire skip list.
- A large denylist is fragile: new junk file types require ongoing maintenance. An allowlist is closed by default — unknown file types are excluded automatically.

---

## 25. RAG Ingestion — Multi-Repo Project Tagging

**Decision:** `Advista_api` and `Advista_client` are both tagged `project="Advista"`. Both repo roots are walked under a single `PROJECT_ROOTS["Advista"]` entry.

**Reasons:**
- They are one logical product (backend + frontend of the same platform). Splitting them into separate project tags would require the retriever to issue two filtered queries to answer questions about Advista.
- Unified tagging enables `filters={"project": "Advista"}` to return relevant chunks from both repos in a single search call.

---

## 26. Config — Absolute `env_file` Path

**Decision:** `config.py` resolves the `.env` path as `Path(__file__).resolve().parent.parent.parent / ".env"` (absolute, relative to the file's location) rather than the string `".env"` (relative to CWD).

**Reason:** `pydantic-settings` resolves a bare `".env"` relative to the process's working directory, not the config file's location. Running the worker or ingest script from `src/` would silently fail to load the env file, causing a `ValidationError` at startup for all required fields. Pinning the path to the file's own directory makes Settings work from any CWD.

---

## 27. RAG Techniques Selected (Post-Analysis)

**Decision:** From the RAG research, we are implementing:
1. High Priority bug fixes (VectorStore singleton, async support, clean exports)
2. Multi-Representation Indexing
3. HyDE (Hypothetical Document Embeddings) for Chat Only

**Reasons:**
- **Multi-Representation Indexing:** Generating question-representations at ingest time provides a huge recall improvement for "tell me about X" style questions with zero runtime latency cost.
- **HyDE:** Bridging the vocabulary gap between questions and documents is valuable, but costs an extra LLM call (~300-500ms). This is acceptable for the chat interface (where latency constraints are relaxed), but unacceptable for the voice agent (<2s SLA). Therefore, HyDE is restricted to chat-only.
- **Rejected Techniques:** Heavy indexing strategies (RAPTOR, ColBERT), reranking (Cohere), and complex query-time techniques (Multi-Query, CRAG) were evaluated and rejected as overkill for our small corpus (~50-100 chunks) and strict <2s voice latency budget.

---

## 28. Shared Retrieval Interface — One `retrieve()`, Two Formatters

**Decision:** A single `retrieve(query, k, project) -> RetrievalResult` function in `src/rag/retrieve.py` is the only retrieval entry point. Both the voice agent and the chat interface call it. Formatting — not retrieval — is what differs between the two consumers.

**Shape:**
- `RetrievalResult` carries the raw `SearchResult` list plus `to_context_block()`, which prefixes each chunk with its source (e.g. `[Advista/Advista_api/README.md]`) so the LLM can ground and cite accurately.
- `format_for_voice(result)` — strips fenced code blocks, inline code, and markdown headers, collapses whitespace to flowing prose, and takes only the top `RAG_VOICE_TOP_K` chunks. Protects the voice latency budget.
- `format_for_chat(result)` — preserves full markdown and code blocks, separates chunks with dividers, takes up to `RAG_CHAT_TOP_K`.

**Reasons:**
- The assignment requires the same persona/RAG backend behind both voice and chat. Building retrieval once and varying only presentation prevents the two surfaces from drifting apart.
- A module-level lazy `VectorStore` singleton (`_get_store()`) is instantiated on first call and reused. Re-creating `QdrantClient` + `OpenAI` clients (each with their own `httpx` connection pool) per query was wasteful overhead on the hot path.

---

## 29. Relevance Floor — Honest "I Don't Know" Over Hallucination

**Decision:** `RAG_RELEVANCE_FLOOR = 0.30`. If the top result's score falls below the floor, `retrieve()` flags the result `low_confidence`, and all three formatters return an empty string.

**Reasons:**
- Empty context → the agent's system prompt makes it say "I don't have that detail" rather than grounding on irrelevant chunks. This directly satisfies the assignment's "recover gracefully when it doesn't know — don't invent" requirement.
- The floor is config-driven (`RAG_RELEVANCE_FLOOR`), so it can be tuned once manual retrieval testing reveals the real score distribution of good vs. junk matches on this corpus.
- 0.30 is a starting cosine-score estimate for `text-embedding-3-small`, not a measured optimum — it will be calibrated during Phase 5 eval testing.

---

## 30. RAG Top-k — Voice 3, Chat 6 (Resolved)

**Decision:** `RAG_VOICE_TOP_K = 3`, `RAG_CHAT_TOP_K = 6`, `RAG_DEFAULT_TOP_K = 6`. All config-driven. (Previously deferred — now resolved.)

**Reasons:**
- Voice runs under a <2s SLA: fewer chunks means a smaller system-prompt injection, less LLM input to process, and faster TTFT. Three chunks is enough to ground a 1–2 sentence spoken answer.
- Chat has a relaxed latency budget and benefits from broader context for detailed, multi-part answers — six chunks.
- This is the deliberate latency-vs-accuracy tradeoff called out in the assignment's eval-report requirement, made explicit and tunable rather than hardcoded.

---

## 31. Multi-Representation Indexing — Implementation

**Decision:** Implemented per Decision 27. At ingest, one LLM call per chunk (`settings.LLM_MODEL`, JSON mode, `temperature=0`) generates a one-sentence summary + 2–3 questions the chunk answers. Each becomes a separate Qdrant point carrying `parent_chunk_id` and `rep_type` ("summary"/"question") in metadata. At retrieve time, representation hits are swapped back to their parent chunk's full text via `VectorStore.fetch_by_ids()`, then deduplicated by `(file_path, chunk_index)` keeping the highest score seen across all of a parent's representations.

**Reasons:**
- Closes the vocabulary gap: an interviewer asking "what's your backend experience?" semantically matches a generated question like "What backend systems did Ayush build?" far better than the raw chunk "Built REST APIs using FastAPI at DevDynamics."
- All LLM work happens once at ingest — **zero runtime latency cost**. At ~50–100 chunks the one-time spend is negligible (printed as an estimate at ingest).
- Returning the *parent* text (not the summary/question) preserves full grounding detail for the LLM while the *representation* is what made the match.
- `--no-multirep` flag skips generation for fast/cheap dev re-ingests.

---

## 32. Async Retrieval — Wrap at Call Site, Not an Async Rewrite

**Decision:** `retrieve()` and `VectorStore` stay synchronous. Where they are called from async contexts (FastAPI chat routes, LiveKit voice pipeline), the blocking call is offloaded with `anyio.to_thread.run_sync(retrieve, query)` at the call site. **Supersedes the "async support" item listed in Decision 27.**

**Alternatives considered:** Rewriting `store.py` and `retrieve.py` on `AsyncQdrantClient` + `AsyncOpenAI`.

**Reasons:**
- The async rewrite would touch every method in working, already-validated Phase-2 code for a problem that a one-line thread offload solves at the boundary.
- `anyio.to_thread.run_sync` correctly keeps the event loop unblocked without forcing the entire RAG module to carry parallel sync/async code paths.
- Retrieval is I/O-bound (embed + Qdrant round-trip); a thread pool is a fitting place for it. This is wired up in Phase 4 when the chat routes and pipeline hook are built.

---

## 33. RAG Injection Point — Ephemeral `turn_ctx`, Not the System Prompt

**Decision:** The voice agent grounds each turn by appending retrieved context to LiveKit's per-turn `turn_ctx` inside `on_user_turn_completed`, **not** by rebuilding the system prompt. `build_system_prompt()` keeps its optional `retrieved_context` param (used by the Phase-4 chat single-shot path), but the voice pipeline calls it with no argument so the system prompt is built once and stays static.

**Alternatives considered:** Inject retrieved facts into the system prompt every turn (the literal reading of the original task).

**Reasons:**
- The system prompt is the longest stable prefix the LLM provider caches on. Rewriting it every turn invalidates the cache for **every** token — the opposite of the goal.
- LiveKit hands `on_user_turn_completed` an *ephemeral copy* of the chat context (`agent_activity.py:2054`, "changes will not be kept inside the Agent.chat_ctx"). Appending there grounds the current generation only; it is not persisted, so retrieved blocks never accumulate in history and the persistent prefix stays append-only and fully cacheable.
- Result ordering: `[static system] → [conversation history] → [retrieved context] → [user message]`. Only the freshly-retrieved tail is uncached each turn, which is unavoidable.
- Low-confidence retrieval (top score < `RAG_RELEVANCE_FLOOR`) injects an explicit "say you don't have that detail" note instead of junk chunks.

---

## 34. Preemptive Generation — Disabled Under RAG

**Decision:** Set `preemptive_generation={"enabled": False}` in the RAG build. (V1 ran it on.)

**Reasons:**
- Preemptive generation speculatively runs the LLM during the caller's speech using the **un-grounded** chat context (`agent_activity.py:1902`). Our `on_user_turn_completed` then injects retrieved facts, mutating the context. LiveKit's `is_equivalent` guard (`agent_activity.py:2091`) detects the change and discards the speculation **every single turn**.
- Kept on, it would burn a wasted LLM call + log a warning per turn for zero latency benefit — the real reply must wait for retrieval regardless.
- Inherent cost of grounding, not a regression we can engineer away here: a speculative answer that ignores retrieval would be ungrounded. Hiding retrieval latency would require *speculative retrieval during speech* (deferred — see Open list).

---

## 35. Retrieval Latency — Measured, and It's the New Critical-Path Cost

**Decision:** Added `retrieval_latency` as a first-class stage in `CallMetrics` (surfaced in `GET /metrics` as `retrieval_latency_secs` p50/p95). Also fixed the V1 metrics bug: non-response turns (barge-ins with no audio, end-of-call flushes) are tagged `responded=False` with null latencies and excluded from all percentiles, instead of logging 0.00 and dragging p50/p95 down.

**Finding (measured against live Qdrant from the dev machine):** `retrieve()` ≈ **760ms p50**, broken down as embed ~256ms + Qdrant query ~168ms + parent-chunk fetch ~163ms — three sequential cross-region HTTP round trips. This is network-RTT-bound and should fall substantially in production where the worker is co-located with OpenAI/Qdrant. The parent-fetch round trip (Decision 31 multi-rep resolution) is the clearest removable cost (denormalize parent text into representation payloads at ingest).

---

## 36. Relevance Floor — Per-Chunk Filter, Recalibrated 0.30 → 0.20 (Supersedes 29)

**Decision:** The relevance floor is now a **per-chunk filter**, not an all-or-nothing gate, and `RAG_RELEVANCE_FLOOR` was lowered from 0.30 to **0.20**.

**What was broken (found by replaying the call logs):** Decision 29 discarded the **entire** retrieved context whenever the *top* score fell below 0.30. On the live call, "Uh, where did he work?" retrieved the **DevDynamics Work Experience chunk as hit #1 at 0.2881** — 0.012 under the floor — so all context was thrown away and the agent said *"I don't have details about where Ayush has worked."* The correct answer was retrieved and then deleted. Natural voice phrasing ("Uh", "he", "him") depresses cosine by ~0.3 versus using the name ("Where has Ayush worked?" → 0.629), so the floor was punishing exactly how callers actually speak.

**Fix:** `retrieve()` keeps every chunk scoring ≥ floor and drops only the ones below it; `low_confidence` is now True **only when nothing clears the floor** — the genuine "I don't have that" signal, instead of a near-miss top score nuking an otherwise-correct chunk.

**Why 0.20 (measured on this corpus, `text-embedding-3-small`):** genuinely off-topic queries top out ~0.15 ("what is the weather today" → 0.153, correctly empty); naturally-phrased on-topic voice queries land ~0.24–0.30. 0.20 keeps real content while still dropping cross-doc junk. Resolves the "relevance floor calibration" deferred item.

---

## 37. Over-Fetch to Offset Multi-Rep Dedup (Resolves Deferred Item)

**Decision:** `retrieve(k)` now over-fetches `max(k × RAG_OVERFETCH_FACTOR, RAG_OVERFETCH_MIN)` (default 4× / 20) raw candidates, resolves them to parent chunks, filters by the floor, then truncates to k **distinct** parents.

**What was broken:** Multi-representation indexing (Decision 31) stores a summary + 2–3 questions per chunk; a raw top-k therefore frequently returns several representations of the **same** parent, which dedup collapses. Measured: "tell me about Ayush" at k=6 returned only **2** distinct chunks; "What is Ayush's work experience?" only 2. The voice path (`k=3`) routinely delivered 1–2 chunks to the LLM — starving grounding. Over-fetching to 20 raw recovers 6 distinct parents for the same queries.

**Cost:** one wider Qdrant query (still a single round trip); negligible at this corpus size. Resolves the "over-fetch to offset multi-rep dedup shrinkage" deferred item.

---

## 38. Connection Prewarm During Greeting (First-Turn Latency)

**Decision:** `rag.retrieve.prewarm()` runs one tiny embed+search, fired best-effort from the agent's `on_enter` **after** `say(greeting)` so it overlaps with greeting playback.

**Reason:** Retrieval is two sequential cross-region round trips (OpenAI embed + Qdrant query); the first call in a fresh per-job subprocess also pays the TLS handshake. Call logs and eval runs both show the first retrieval at ~1.9–2.7s versus ~0.7–0.9s once warm. The greeting gives ~3s of audio to hide that handshake behind, so the first *real* turn starts warm. Failure is swallowed — warmup can never break a call.

---

## 39. Golden-Dataset Eval Harness + First Results

**Decision:** Added `src/eval/` — a 24-question golden set (`golden.jsonl`, 20 answerable across resume + 3 repos + 4 deliberately unanswerable) and `run_eval.py`, which mirrors the voice path (`retrieve` → `format_for_voice` → `build_system_prompt(context)` → `gpt-4.1-mini` answer) and scores each answer with **gpt-4.1-mini as judge** on correctness, groundedness (hallucination), and refusal. It also reports retrieval recall@k, low-confidence rate, and per-stage latency.

**First run (k=3 voice path, after Decisions 36–38):**
- Overall pass **91.7%** (22/24) · Answer correctness **90%** · **Groundedness 100% (0% hallucination)** · Refusal accuracy **100%** (4/4 unanswerable correctly declined) · Retrieval recall@k **90%** · Low-confidence rate **0%**.
- Latency: retrieval p50 0.72s / p95 2.34s (p95 is the cold first turn, addressed by Decision 38); end-to-end retrieval+generation p50 2.05s.

**Two documented failure modes (both retrieval-depth, not hallucination — the agent never fabricated):**
1. `work-mcp` — "Tell me about the MCP server he built." The resume's DevDynamics MCP content lives inside one large merged "Work Experience" chunk, so a focused "MCP server" query ranks it #4 (behind low-signal `decisions.md`/`package.json` noise); voice top-3 misses it. *Fix direction: split the Work Experience section into per-project sub-chunks at ingest, and/or prune dependency-manifest files from the corpus.*
2. `proj-flick-model` — "What model does flick2code use by default?" The right chunk was retrieved (#1, 0.73) but the terse voice persona answered "configurable via an environment variable" without naming `gemini-2.5-flash`. *Fix direction: answer-specificity nudge in the persona prompt for factual lookups.*

These are the kind of finding the Part-C eval report wants; left as-is rather than overfitting to 24 questions.

---

## 40. Multi-Turn / Follow-Up Eval — Query Rewriting Tested and Rejected

**Decision:** Added a conversational eval (`src/eval/golden_multiturn.jsonl` + `run_multiturn.py`): 10 dialogues / 19 graded turns that deliberately use pronouns and ellipsis in follow-ups ("where did **he** work?", "what LLM does **it** use?", "why SSE **there**?", "**what about** Advista?", "and **his** CGPA?"). It replays each conversation turn-by-turn exactly as the agent would (static system prompt + persisted history + per-turn retrieved context), retrieves on the **raw last message** like the live pipeline, and reports results split by **first-turn vs follow-up**.

**Motivation:** the real failure in the call logs was a *follow-up* ("He has a lot of experience" → "where did he work?"), and the pipeline retrieves on the bare last message (`pipeline.py:66`), so the worry was that pronoun/ellipsis follow-ups would retrieve poorly.

**Finding (k=3, gpt-4.1-mini judge), raw last-message retrieval:** follow-up retrieval recall@k **100%**, follow-up pass **90%** (≥ first-turn's 78%), groundedness **100%**. The motivating failure ("where did he work?") now passes. After Decisions 36–37 the bare follow-up query already retrieves the right chunks, and the conversation history gives the LLM the referent — so follow-up resolution is **not** the bottleneck.

**Query rewriting tested and rejected:** `--rewrite` condenses history into a standalone query before retrieval. It did **not** improve follow-ups — it regressed them: follow-up pass 90% → **80%**, groundedness 100% → **90%** (introduced one ungrounded answer), recall already saturated at 100% either way. Example: "What's the tech stack of the Advista one?" rewrote to "Tech stack used in the Advista project" and pulled the *client* README (React/Vite) instead of the backend, giving a narrower answer. Rewriting also adds a per-turn LLM call to the latency-critical voice path. **Conclusion: do not add query rewriting to the voice agent** — it costs latency and slightly hurts quality at this corpus size.

**Remaining failures are not follow-up issues** — same classes as the single-shot eval: answer under-specificity (says "Groq" without "llama-3.3-70b-versatile"), occasional over-refusal ("Did he evaluate ValueX?" → declined despite the eval suite being in context), and projects-vs-work conflation ("what projects" returns DevDynamics work items, not the named Advista/ValueX/flick2code — the tiny "## Projects" chunk doesn't rank). These point at chunking + a small persona-prompt nudge, not retrieval-query construction.

---

## 41. Calendar Booking — Cal.com API **v2** (the brief's v1 is decommissioned)

**Decision:** Implemented end-to-end booking (Part A) against **Cal.com API v2**, not v1 as the brief and CLAUDE.md specify. The three LLM tools live in `src/agent/tools/calendar.py`: `get_available_slots(date_hint)`, `collect_contact_info()`, `book_slot(datetime_str, name, email)`, wired into the agent via LiveKit's `tools=[...]` (the `@function_tool` API — `FunctionContext` is the old 0.x name and doesn't exist in livekit-agents 1.5.17).

**Why not v1 (the broken assumption):** every v1 endpoint now returns **HTTP 410 Gone** — `{"message":"API v1 has been decommissioned. Please migrate to API v2"}`. Verified live against Ayush's key on both `GET /v1/event-types` and `GET /v1/me`. So `/v1/availability` and `/v1/bookings` from the brief are dead; CLAUDE.md's rule ("find if something is broken and suggest a change") applies.

**v2, grounded by live probing (no doc-guessing):**
- Auth `Authorization: Bearer <cal_…>`; each endpoint pinned by a `cal-api-version` header (config-driven).
- **Slots** `GET /v2/slots?eventTypeId&start&end&timeZone` (`2024-09-04`) → `{"data": {"YYYY-MM-DD": [{"start": ISO+offset}]}}`. Chosen over v1's raw `/availability` because it returns *bookable* times with working-hours + busy already applied — the right primitive for "propose slots."
- **Bookings** `POST /v2/bookings` (`2024-08-13`) with `{start (UTC), eventTypeId, attendee:{name,email,timeZone,language}}`. Required-field set confirmed by a **non-mutating 400 probe** (posting an incomplete body returns "start … attendee should not be null"), so no test booking was needed to learn the schema.
- **Event types** `GET /v2/event-types?username` (`2024-06-14`) resolves the configured slug (`30min`) → numeric `eventTypeId` (5917916), since slots/bookings take an id, not a slug. Cached per call; prewarmed during the greeting.

**NL date parsing:** `dateparser` (added to requirements). dateparser 1.x returns `None` for two phrasings the brief lists — `"next Tuesday"` and `"<weekday> afternoon"` — so `_parse_date` strips period/filler words (period is handled separately as an hour filter) and retries without a leading qualifier. Output is spoken-friendly and slot-spread for voice ("Monday June 8 at 9 AM, 10 AM, 11 AM" not a dense half-hour wall).

**Latency:** warm `get_available_slots` measured **~300–400ms** (one `/slots` call; eventTypeId prewarmed off the critical path) — well under the 1.5s budget. New monitoring stage `tool_call_latency_secs` records the Cal.com round-trip per turn (same sealing pattern as retrieval).

**Not executed live:** the real `POST /v2/bookings` is intentionally *not* run by the test harness — it sends a real confirmation email + creates a real event (outward-facing, not cleanly reversible). `test_calendar.py` is read-only by default (prints raw `/slots` JSON + the dry-run booking payload, UTC-converted) with a `--live-book` opt-in for a human-authorized end-to-end confirmation.

**Failure handling:** every tool returns a *speakable* string; no exception, stack trace, or raw API body can reach the caller (Cal.com errors → "I'm having trouble reaching the calendar…"; unparseable day → ask to be more specific; bad email → ask to spell it; taken slot → offer to re-check).

---

## 42. Pipeline Solidification Pass (Eval-Driven)

After the eval suites were in place, a hardening pass turned their findings into changes — each one measured before/after:

**Corpus pruned to markdown-only (supersedes Decision 22's manifest inclusion).** Dependency manifests (`package.json`, `*.toml`, `requirements.txt`) carried no prose signal but matched generic tech queries and out-ranked the resume — on "the MCP server he built" the resume chunk fell to rank #4 and was cut at voice top-3. Dropping them (76 → 66 chunks) lifted single-shot **retrieval recall@k 90% → 95%** (fixed `work-mcp`) and multi-turn **first-turn pass 78% → 89%**; groundedness/refusal stayed 100%.

**Specificity prompt nudge tested and reverted — the eval caught it.** Adding "state the exact number/model" to the GROUNDING clause made the model **fabricate** a statistic (Lambda cache "reduced by 40%", a figure not in context), dropping groundedness 100% → 95.8%. Reverted; kept only the harmless "answer if the facts cover it" phrasing. Lesson recorded: for a grounded persona, **groundedness > completeness** — don't prompt for specificity at the cost of fabrication. This is exactly the regression the gate (below) exists to block.

**Retrieval made fail-open in the live path.** `on_user_turn_completed` now wraps `retrieve()` in `asyncio.wait_for(timeout=RAG_RETRIEVAL_TIMEOUT_S=3.0s)` + a catch-all: on a timeout or any embed/Qdrant error the agent answers **without** grounding rather than stalling or dropping the call. Essential for 7 days of always-on unannounced calls; a backend blip must never kill a turn.

**Regression gate added.** `python -m eval.run_eval --gate` exits non-zero if groundedness < 95%, refusal < 90%, recall@k < 85%, or overall pass < 80%. Run it before the `gpt-4.1-mini → gpt-4.1` submission swap so a model/prompt/corpus change can't silently regress the safety properties.

**TTS-output hygiene checked, no fix needed.** Scanned 43 eval answers for TTS-hostile tokens (arrows, markdown, bullets, raw URLs): 0 markdown/arrows/bullets; the only hit was the email in the contact answer (expected). The persona's "spoken prose, no markdown" instruction already yields clean audio — no normalization layer warranted.

**Still manual (Part C):** place 10+ live calls and aggregate `monitoring.py`'s JSONL (`summary_from_turns`) for real e2e/interruption/cost numbers — the offline harness can't measure STT/TTS/network/barge-in or confirm the <2s first-response SLA.

---

## 43. Part C Unified Eval Suite + First Full Results

**Decision:** Built `src/eval/run_evals.py` — one orchestrator that runs all five Part C
measurements in a single pass and writes a combined `src/eval/results/run_{ts}.json` — plus
`src/eval/report.py` (Markdown + a strict 1-page reportlab PDF). It **reuses** rather than
duplicates: the judge + `ItemResult` from `run_eval.py` (Decision 39) and the JSONL loader from
`monitoring.py` (Decision 13). Kept under `src/eval/` (not a parallel top-level `evals/`) and on
the existing golden schema so the working harness isn't forked. The golden set grew **24 → 30**
(`golden.jsonl`): added a `fit` category (role-fit synthesis — AI-engineer fit, LLM experience,
eval-framework experience, production-systems experience) and two decisions-retrieval questions
("what would Ayush do differently in Advista", "what design decisions in ValueX" → grounded in
`Ayush_Rathod_Resume.md` + `ValueX/decsions.md`). Evals 1+2 share one **chat-path** pass
(k=`RAG_CHAT_TOP_K`=6, `format_for_chat` → `build_chat_prompt` → gpt-4.1-mini at **temp 0** for
reproducibility); judge is gpt-4.1-mini.

**First full run (30 Q, chat k=6, gpt-4.1-mini answer + judge):**
- **Retrieval:** precision@6 **0.616**, recall@6 **1.00** (every answerable Q retrieved its
  expected source), mean relevance **0.612**. Precision is highest on factual lookups
  (education/achievements/specialisation 1.00, work 0.73, skills 0.83) and lowest on multi-source
  synthesis (project 0.47, fit 0.63) — expected, those legitimately pull from several files.
- **Groundedness:** hallucination **0%** (0/30), grounded **100%**, answer correctness **92.3%**,
  out-of-corpus refusal **4/4 (100%)**, **0 fabrications** on the unanswerable set.
- **Voice latency** (from `metrics.jsonl`, 30 responded turns / ~7 calls): first audio out
  (TTS TTFB) p50 **0.47s** / p95 1.04s → **100% under 2s** (the brief's "<2s first response");
  full-turn E2E (user stops → agent speaks) p50 **4.45s** / p95 7.82s → **0% under 2s**.
- **Task completion:** auto-derived **0/3** from the metrics log (3 slot lookups, 0 logged
  bookings) — a lower bound pending a real test-call tally via `--bookings-attempted/-succeeded`.
- **Transcription:** **n/a** — see finding below.

**Key honest findings (the Part-C-worthy ones):**
1. **Report first-audio and full-turn latency separately.** TTS TTFB is fast (0.47s) but E2E is
   4.45s p50; the gap is **cross-region RAG retrieval (~1.3s, in the critical path) + the semantic
   end-of-turn wait**, not synthesis. A blanket "<2s" claim would be false — the system meets <2s
   on first response but not on full perceived turn under RAG. Highest-leverage fix is
   **speculative retrieval during speech** (Decision 34 / Open table).
2. **The out-of-corpus "name-magnet" is real and measured.** Retrieval-layer `low_confidence`
   gating fired on **0/4** unanswerable questions — the resume contact chunk clears the 0.20 floor
   for *any* query mentioning "Ayush" (confirms what_if §6). End-to-end refusal still held at
   **100%** because the persona declines on content (no salary/DOB/Kubernetes/QuantumLeap fact in
   the chunk to fabricate from). The safety property is intact; the retrieval gate is *not* what
   protects it. Cross-encoder reranking is the deeper fix.
3. **Correctness misses are incompleteness, not error.** Both 92.3% misses (`fit-evals`,
   `proj-valuex-decisions`) were grounded-but-incomplete on multi-part synthesis questions (named
   ValueX's eval suite but not DevDynamics' LLM-as-judge; named SSE but not the two-phase
   architecture). Confirms the judge-calibration finding: for a grounded persona,
   **groundedness > completeness**.

**Config discrepancy surfaced:** `config.py` currently has `RAG_VOICE_TOP_K = 6` (all three
top-k = 6), which **supersedes Decision 30's "voice 3"** — the voice/chat split the report's
tradeoff section assumed no longer exists in code. Either restore voice=3 (the §30 latency
argument still holds) or drop the "voice 3 vs chat 6" framing from the report. Flagged, not
silently changed.

---

## Open / Deferred Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| RAG: top-k value for voice vs chat | **Reopened — see Decision 43** | §30 set voice=3 / chat=6, but `config.py` now has `RAG_VOICE_TOP_K=6` (all three = 6). Reconcile before submission: restore voice=3 or drop the "voice 3 vs chat 6" report framing. |
| RAG: chunk size and overlap strategy | **Resolved — Decision 23** | CHUNK_TARGET=400, CHUNK_MAX=500, CHUNK_OVERLAP=50, tiktoken cl100k_base |
| RAG: over-fetch to offset multi-rep dedup shrinkage | **Resolved — Decision 37** | Confirmed thin (k=6 → 2 distinct chunks). `retrieve()` now over-fetches `max(k×4, 20)` raw, dedups to k distinct parents. |
| RAG: HyDE for chat | Selected (Decision 27), not built | Phase 4 chat path; +1 LLM call, chat-only |
| RAG: relevance floor calibration | **Resolved — Decision 36** | Measured score distribution; recalibrated 0.30 → 0.20 and switched from all-or-nothing gate to per-chunk filter. |
| Calendar: Cal.com vs Calendly | Cal.com selected | Phase 3 |
| Calendar: Cal.com API v1 vs v2 | **Resolved — Decision 41** | v1 decommissioned (HTTP 410); built on v2 (Bearer auth, `cal-api-version` pins, `/v2/slots` + `/v2/bookings`) |
| Chat frontend: Next.js on Vercel | Decided, not built | Phase 4 |
| LLM upgrade: gpt-4.1-mini → gpt-4.1 | Pre-submission | One env var change |
| STT real latency measurement | Pending first live call | Expected 200–400ms streaming |
| Metrics ingest endpoint (POST /metrics/ingest) | Stubbed, not implemented | Worker currently writes to JSONL directly; HTTP ingest deferred |
| RAG: eliminate parent-fetch round trip | **Resolved** | `parent_text` denormalized into rep payloads at ingest. `retrieve()` now 2 round trips (embed + Qdrant query). Measured: 761ms → 578ms p50 (−183ms). |
| RAG: speculative retrieval during speech | Deferred — Phase 5 latency work | Retrieve on partial transcript before EOU to hide retrieval RTT, recovering the lead lost by disabling preemptive generation (Decision 34). |
| RAG: voice context character budget | Deferred — pending live-call quality check | `format_for_voice()` can emit ~1200 tokens/turn for top-3 large chunks; a per-chunk/total cap would cut uncached prefill + cost. Hold until live calls confirm answer quality. |
| Eval: persist user transcript for WER proxy | Deferred — Decision 43 | `metrics.jsonl` (TurnMetrics) stores no transcript text; the only `user_transcript` strings lived in the untracked LiveKit `logs.txt`, lost mid-session. Persist the transcript onto TurnMetrics (or a sidecar) so the proxy is reproducible. True WER still needs paired ground-truth. |
| Eval: instrument booking outcome | Deferred — Decision 43 | Only `get_available_slots` shows in metrics (`book_slot` is never auto-run — Decision 41), so booking success is a manual tally. Record a `book_slot` outcome event on real authorized bookings. |
