"""
RAG ingestion pipeline.

Walks repos/, chunks every allowed file, upserts into Qdrant via VectorStore.

Usage (run from src/):
    python -m rag.ingest --dry-run          # chunk + count, no embedding spend
    python -m rag.ingest --reset            # wipe collection first, then ingest
    python -m rag.ingest                    # upsert without wiping
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag.models import Chunk 

# Corpus layout
REPOS_DIR = _SRC.parent / "repos"

PROJECT_ROOTS: dict[str, list[Path]] = {
    "flick2code": [REPOS_DIR / "flick2code"],
    "ValueX":     [REPOS_DIR / "ValueX"],
    "Advista":    [
        REPOS_DIR / "Advista" / "Advista_api",
        REPOS_DIR / "Advista" / "Advista_client",
    ],
}
RESUME_PATH = REPOS_DIR / "Ayush_Rathod_Resume.md"

# Chunking parameters
CHUNK_TARGET = 400   # merge small sections up to this
CHUNK_MAX    = 500   # hard cap before force-splitting
CHUNK_OVERLAP = 50   # token overlap on force-splits

_ENC = tiktoken.get_encoding("cl100k_base")

def _tokens(text: str) -> int:
    return len(_ENC.encode(text))

# File filter — markdown-only 
SKIP_DIRS: frozenset[str] = frozenset({"node_modules", ".git", "venv", ".venv"})

def _include(path: Path) -> bool:
    return path.suffix.lower() == ".md"


def _walk(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fname in sorted(filenames):
            fp = Path(dirpath) / fname
            if _include(fp):
                found.append(fp)
    return found


# Chunking
_HEADER_RE = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)


def _split_on_headers(text: str) -> list[str]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []
    sections: list[str] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(preamble)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        s = text[m.start() : end].strip()
        if s:
            sections.append(s)
    return sections


def _force_split(text: str) -> list[str]:
    tokens = _ENC.encode(text)
    out: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_MAX, len(tokens))
        out.append(_ENC.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP
    return out


def _chunk(text: str) -> list[str]:
    sections = _split_on_headers(text)
    raw: list[str] = []
    buf = ""
    for section in sections:
        candidate = f"{buf}\n\n{section}".strip() if buf else section
        if _tokens(candidate) <= CHUNK_TARGET:
            buf = candidate
        else:
            if buf:
                raw.extend(_force_split(buf) if _tokens(buf) > CHUNK_MAX else [buf])
            buf = section
    if buf:
        raw.extend(_force_split(buf) if _tokens(buf) > CHUNK_MAX else [buf])
    return raw


# Multi-representation generation (summary + questions per chunk)
_REP_SYSTEM_PROMPT = (
    "Return JSON with two keys:\n"
    '- "summary": one sentence capturing the key facts in this text\n'
    '- "questions": list of 2-3 natural-language questions this text answers\n'
)

_LLM_INPUT_COST_PER_1M  = 0.40   # gpt-4.1-mini input, USD
_LLM_OUTPUT_COST_PER_1M = 1.60   # gpt-4.1-mini output, USD
_APPROX_REP_OUTPUT_TOKENS = 80


def generate_representations(chunks: list[Chunk]) -> list[Chunk]:
    """For each chunk generate a summary and 2-3 questions as extra embedding targets.

    Representation chunks carry parent_chunk_id + rep_type in metadata so retrieve.py
    can swap them back to the original text at query time.
    """
    from openai import OpenAI
    from utils.config import settings 

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = settings.LLM_MODEL
    rep_chunks: list[Chunk] = []

    for i, chunk in enumerate(chunks, 1):
        print(f"  Generating representations {i}/{len(chunks)}…", end="\r", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REP_SYSTEM_PROMPT},
                    {"role": "user",   "content": chunk.text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=256,
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [WARN] rep generation failed for chunk {chunk.id}: {exc}", file=sys.stderr)
            continue

        # Denormalize parent text so retrieve.py can reconstruct the parent SearchResult
        # from the rep chunk's own payload — no second fetch_by_ids round trip needed.
        base_meta = {**chunk.metadata, "parent_chunk_id": chunk.id, "parent_text": chunk.text}

        summary = (data.get("summary") or "").strip()
        if summary:
            rep_chunks.append(Chunk(text=summary, metadata={**base_meta, "rep_type": "summary"}))

        for q in (data.get("questions") or [])[:3]:
            if isinstance(q, str) and q.strip():
                rep_chunks.append(Chunk(text=q.strip(), metadata={**base_meta, "rep_type": "question"}))

    print(f"  Generated {len(rep_chunks)} representations for {len(chunks)} chunks.    ")
    return rep_chunks


# Per-file and per-project helpers
def _chunks_for_file(path: Path, project: str, source_type: str) -> list[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [SKIP] {path}: {exc}", file=sys.stderr)
        return []
    if not text.strip():
        return []
    rel = str(path.relative_to(REPOS_DIR))
    return [
        Chunk(
            text=chunk,
            metadata={
                "source_type": source_type,
                "project": project,
                "file_path": rel,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(_chunk(text))
        if chunk.strip()
    ]


def ingest_resume() -> list[Chunk]:
    if not RESUME_PATH.exists():
        print(f"[WARN] Resume not found: {RESUME_PATH}", file=sys.stderr)
        return []
    return _chunks_for_file(RESUME_PATH, "resume", "resume")


def ingest_project(project: str, roots: list[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] Root not found: {root}", file=sys.stderr)
            continue
        for fp in _walk(root):
            chunks.extend(_chunks_for_file(fp, project, "repo"))
    return chunks


# main
_EMBED_COST_PER_1M = 0.02  # text-embedding-3-small, USD


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into Qdrant")
    parser.add_argument("--reset",      action="store_true", help="Wipe collection first")
    parser.add_argument("--dry-run",    action="store_true", help="Count only, no writes")
    parser.add_argument("--no-multirep", action="store_true", help="Skip representation generation")
    args = parser.parse_args()

    all_chunks: dict[str, list[Chunk]] = defaultdict(list)
    all_chunks["resume"].extend(ingest_resume())
    for project, roots in PROJECT_ROOTS.items():
        all_chunks[project].extend(ingest_project(project, roots))

    total_chunks = sum(len(v) for v in all_chunks.values())
    total_tokens = sum(_tokens(c.text) for cs in all_chunks.values() for c in cs)

    print("\n=== Chunk distribution ===")
    for proj in ["resume", *sorted(k for k in all_chunks if k != "resume")]:
        cs = all_chunks[proj]
        t = sum(_tokens(c.text) for c in cs)
        avg = t // len(cs) if cs else 0
        print(f"  {proj:<15}  {len(cs):>4} chunks  {t:>8,} tokens  (avg {avg} tok/chunk)")
    print(f"  {'TOTAL':<15}  {total_chunks:>4} chunks  {total_tokens:>8,} tokens")

    embed_cost = total_tokens * _EMBED_COST_PER_1M / 1_000_000
    print(f"\n  Embedding cost (est): ${embed_cost:.5f}")
    if not args.no_multirep:
        # Rep generation sends each chunk once as input, emits ~80 output tokens.
        llm_cost = (
            total_tokens * _LLM_INPUT_COST_PER_1M
            + total_chunks * _APPROX_REP_OUTPUT_TOKENS * _LLM_OUTPUT_COST_PER_1M
        ) / 1_000_000
        print(f"  LLM rep cost  (est): ${llm_cost:.5f}  ({total_chunks} calls)")

    if args.dry_run:
        print("\n[dry-run] Nothing written.")
        return

    from rag.store import VectorStore  # noqa: PLC0415
    store = VectorStore()
    if args.reset:
        print("\nResetting collection…")
        store.reset()
    else:
        store.ensure_collection()

    flat = [c for cs in all_chunks.values() for c in cs]

    if not args.no_multirep:
        print(f"\nGenerating multi-representations for {len(flat)} chunks…")
        rep_chunks = generate_representations(flat)
        flat = flat + rep_chunks

    print(f"\nUpserting {len(flat)} points…")
    store.upsert(flat)
    print(f"Done. Collection now has {store.collection_info()['points_count']} points.")


if __name__ == "__main__":
    main()
