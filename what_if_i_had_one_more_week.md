# What If I Had One More Week? (RAG Enhancements)

This document outlines the advanced RAG techniques that were researched and evaluated, but deliberately skipped or deferred due to the strict 36-hour deadline, the small corpus size, and the sub-2-second voice latency budget.

If I had one more week to work on this persona agent, here is what I would explore:

## 1. Advanced Query Construction
- **Multi-Query / RAG-Fusion:** Currently, we embed the user's raw query (or a HyDE hypothetical answer). With more time, we could use an LLM to generate 3-5 variants of the user's question, retrieve chunks for all of them, and use Reciprocal Rank Fusion (RRF) to merge the results. *Why skipped:* Each variant requires an embedding call and search, which would blow our voice latency budget. For a ~50-chunk corpus, single-query recall is already sufficient.
- **Step-Back Prompting & Decomposition:** Breaking down complex queries into sub-questions or abstracting them. *Why skipped:* Users typically ask simple, direct questions about Ayush's experience (e.g., "what projects have you built?"), not multi-hop reasoning questions.

## 2. Advanced Retrieval Strategies
- **Semantic Routing:** We could use embedding similarity to classify the query into different intents (e.g., RAG, Calendar Booking, Chitchat) instead of hitting the vector store for every query. This is a very fast (latency-free) optimization that we evaluated but deferred.
- **CRAG (Corrective RAG):** Adding a self-correction loop where an LLM grades the relevance of retrieved chunks before generation. If the chunks are irrelevant, we could trigger a web search fallback. *Why skipped:* Web search fallback doesn't make sense for a personal persona—if a fact isn't in Ayush's resume or repos, the agent should just say "I don't know" rather than hallucinating from the web.
- **Self-RAG:** Training or prompting the LLM to output reflection tokens to decide when to retrieve, when to generate directly, and how to evaluate its own outputs. *Why skipped:* Requires fine-tuning and significant prompt engineering, way beyond the 36-hour scope.

## 3. Reranking and Fine-Grained Matching
- **Cross-Encoder Reranking (e.g., Cohere Rerank):** Using a dedicated reranker to re-score the top-K chunks after the initial vector search. *Why skipped:* Adds a new API dependency and ~200-400ms latency. With a small corpus, standard embedding similarity is usually sufficient without a second-stage precision filter.
- **ColBERT:** A token-level embedding approach for more fine-grained retrieval (late interaction). *Why skipped:* Requires separate model infrastructure (e.g., RAGatouille) and the tech stack is locked to OpenAI embeddings.

## 4. Heavy Indexing
- **RAPTOR (Hierarchical Tree Indexing):** Building a recursive tree of summaries over the document chunks, allowing retrieval at different levels of abstraction. *Why skipped:* Building a hierarchical tree over a tiny 50-chunk corpus is overkill and wouldn't yield meaningful improvements.

## 5. Other Engineering Improvements
- **Caching:** Adding an LRU cache on the retrieval endpoint to save latency and embedding API costs for repeated questions.
- **Hybrid Search (Dense + Sparse):** Adding BM25 sparse vectors to Qdrant for exact term matching (e.g., tech stack keywords or specific project names). *Why skipped:* Dense vectors (text-embedding-3-small) handle semantic matching well enough for our current needs, and hybrid search requires a SPLADE/BM25 encoder dependency.

---

## 6. Evidence-Backed Priorities (surfaced by the eval suite)

The items above were researched in the abstract. These are the ones the golden-dataset and multi-turn evals actually flagged as the current weak spots — each tied to a *measured* failure, which is what makes them the first things I'd pick up.

- **Proposition / per-accomplishment chunking of the resume.** The biggest retrieval miss ("tell me about the MCP server he built") traced to the whole DevDynamics section sitting in one ~400-token chunk, so a focused query ranked it #4 and voice top-3 dropped it. Pruning dependency-manifest noise (done — recall@k 90→95%) bumped it into top-3, but the real fix is splitting that section into one chunk per accomplishment so each gets its own embedding + generated questions. *Why deferred:* the merge-based chunker and `CHUNK_TARGET` would need reworking plus a re-ingest — too risky to change late.

- **Cross-encoder reranking over the candidates we already over-fetch.** §3 called reranking overkill; the evals give counter-evidence — low-signal files (`ValueX/decsions.md`, score 0.36) out-rank real resume content (0.33), and the contact/header chunk is a "name magnet" scoring 0.5–0.66 for *anything* mentioning "Ayush". Since retrieval already over-fetches ~20 candidates before dedup, a reranker slots in cleanly to reorder by true relevance. *Why deferred:* +200–400ms — ship it on the chat path first, where the latency budget is relaxed.

- **Hybrid dense+BM25 (the §5 item, now with evidence).** The under-specificity failures — exact model names (`gemini-2.5-flash`), version numbers, metric figures — are precisely where sparse term matching beats blurry dense similarity. Motivated by data now, not just theory.

- **Judge calibration + a larger, more stable golden set.** The gpt-4.1-mini judge penalized concise, voice-appropriate answers as "incomplete" (gave email+phone but not LinkedIn). One week: calibrate against ~30 human labels, add multi-judge agreement, generate answers at temperature 0 for reproducibility, grow to 50–100 questions, and add precision@k (per-chunk relevance labels) plus k-/floor-sweep curves.

- **Adversarial & consistency eval — the one dimension at zero coverage.** The persona prompt claims prompt-injection resistance and identity guarding, but nothing tests them. Add injection, prompt-extraction, "are you Ayush?", and same-question-three-ways consistency checks.

- **Speculative retrieval during caller speech.** Recovers the latency lost by disabling preemptive generation: start retrieval on the partial transcript before end-of-utterance fires, hiding the embed + Qdrant round trip behind the caller's trailing silence. Highest-leverage voice-latency item remaining.

> Note: **query rewriting was tested and rejected, not skipped.** At this corpus size it *regressed* follow-up accuracy (90% → 80% pass) and added a per-turn LLM call to the voice path — conversation history already gives the model the referent. It only becomes worth revisiting if the corpus grows large enough that bare follow-up queries stop retrieving well.

setup unit intgreation testing and CI
