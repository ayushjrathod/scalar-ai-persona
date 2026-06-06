"""
Shared retrieval interface for both voice agent and chat interface.

Usage (run from src/):
    python -m rag.retrieve "what did Ayush build at DevDynamics?"
    python -m rag.retrieve "Advista architecture" --project Advista --k 4
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag.models import SearchResult
from utils.config import settings


@dataclass
class RetrievalResult:
    query: str
    results: list[SearchResult]
    low_confidence: bool = False

    def to_context_block(self) -> str:
        """Format results as a prompt-injectable block with [source] prefixes."""
        if self.low_confidence or not self.results:
            return ""
        return "\n\n".join(f"[{_source_label(r.metadata)}]\n{r.text}" for r in self.results)


_store = None


def _get_store():
    global _store
    if _store is None:
        from rag.store import VectorStore
        _store = VectorStore()
    return _store


def prewarm() -> None:
    """Establish the OpenAI + Qdrant TLS connections up front with one tiny search."""
    try:
        _get_store().search("warmup", k=1)
    except Exception as exc: 
        import logging
        logging.getLogger(__name__).debug("RAG prewarm skipped: %s", exc)


def retrieve(
    query: str,
    k: int = settings.RAG_DEFAULT_TOP_K,
    project: str | None = None,
) -> RetrievalResult:
    """Embed query and return up to k distinct parent chunks scoring >= RAG_RELEVANCE_FLOOR."""
    filters = {"project": project} if project else None
    fetch_k = max(k * settings.RAG_OVERFETCH_FACTOR, settings.RAG_OVERFETCH_MIN)
    raw = _get_store().search(query, k=fetch_k, filters=filters)
    resolved = _resolve_parent_chunks(raw)
    kept = [r for r in resolved if r.score >= settings.RAG_RELEVANCE_FLOOR][:k]
    return RetrievalResult(query=query, results=kept, low_confidence=not kept)


_REP_ONLY_KEYS: frozenset[str] = frozenset({"parent_chunk_id", "rep_type", "parent_text"})


def _resolve_parent_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Swap representation hits (summary/question) for their original parent chunk text."""
    direct: list[SearchResult] = []
    rep_best: dict[str, float] = {}   
    rep_meta: dict[str, dict] = {}  

    for r in results:
        pid = r.metadata.get("parent_chunk_id")
        if pid:
            if pid not in rep_best or r.score > rep_best[pid]:
                rep_best[pid] = r.score
                rep_meta[pid] = r.metadata
        else:
            direct.append(r)

    if not rep_best:
        return results

    resolved: list[SearchResult] = []
    legacy_ids: list[str] = []

    for pid, score in rep_best.items():
        meta = rep_meta[pid]
        parent_text = meta.get("parent_text")
        if parent_text:
            # Reconstruct the parent result from the denormalized payload — zero extra I/O.
            parent_meta = {k: v for k, v in meta.items() if k not in _REP_ONLY_KEYS}
            resolved.append(SearchResult(text=parent_text, score=score, metadata=parent_meta))
        else:
            legacy_ids.append(pid)

    if legacy_ids:
        # Backward-compat fetch for points ingested before parent_text was denormalized.
        parent_map = _get_store().fetch_by_ids(legacy_ids)
        resolved.extend(
            replace(parent_map[pid], score=rep_best[pid])
            for pid in legacy_ids
            if pid in parent_map
        )

    # Merge direct + resolved; deduplicate by (file_path, chunk_index), keep highest score
    best: dict[tuple, SearchResult] = {}
    for r in direct + resolved:
        key = (r.metadata.get("file_path"), r.metadata.get("chunk_index"))
        if key not in best or r.score > best[key].score:
            best[key] = r

    return sorted(best.values(), key=lambda r: r.score, reverse=True)


# Formatters
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def format_for_voice(result: RetrievalResult) -> str:
    """Top RAG_VOICE_TOP_K chunks, code and markdown stripped, whitespace collapsed."""
    if result.low_confidence or not result.results:
        return ""
    lines = []
    for r in result.results[: settings.RAG_VOICE_TOP_K]:
        text = _strip_for_voice(r.text)
        if text:
            lines.append(f"[{_source_label(r.metadata)}] {text}")
    return "\n".join(lines)


def format_for_chat(result: RetrievalResult) -> str:
    """Top RAG_CHAT_TOP_K chunks with full markdown and code blocks preserved."""
    if result.low_confidence or not result.results:
        return ""
    parts = [
        f"**[{_source_label(r.metadata)}]**\n{r.text.strip()}"
        for r in result.results[: settings.RAG_CHAT_TOP_K]
    ]
    return "\n\n---\n\n".join(parts)


def _source_label(metadata: dict) -> str:
    return metadata.get("file_path") or metadata.get("project", "unknown")


def _strip_for_voice(text: str) -> str:
    text = _CODE_BLOCK_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    text = _HEADER_RE.sub("", text)
    return " ".join(text.split())


# CLI used for testing
def main() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval quality interactively.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=settings.RAG_DEFAULT_TOP_K)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    print(f"\nQuery: {args.query!r}  k={args.k}  project={args.project!r}  floor={settings.RAG_RELEVANCE_FLOOR}\n")

    result = retrieve(args.query, k=args.k, project=args.project)

    print(f"{'=' * 60}\nRAW RESULTS  ({len(result.results)} hits, low_confidence={result.low_confidence})\n{'=' * 60}")
    for i, r in enumerate(result.results, 1):
        preview = r.text[:120].replace("\n", " ")
        print(f"  [{i}] score={r.score:.4f}  {_source_label(r.metadata)}")
        print(f"       {preview!r}")

    print(f"\n{'=' * 60}\nto_context_block()\n{'=' * 60}")
    print(result.to_context_block() or "<empty — low confidence or no results>")

    print(f"\n{'=' * 60}\nformat_for_voice()  (top {settings.RAG_VOICE_TOP_K})\n{'=' * 60}")
    print(format_for_voice(result) or "<empty — low confidence or no results>")

    print(f"\n{'=' * 60}\nformat_for_chat()  (top {settings.RAG_CHAT_TOP_K})\n{'=' * 60}")
    print(format_for_chat(result) or "<empty — low confidence or no results>")
    print()


if __name__ == "__main__":
    main()
