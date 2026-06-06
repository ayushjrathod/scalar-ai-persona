from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

REQUIRED_METADATA = {"source_type", "project", "file_path", "chunk_index"}
VALID_SOURCE_TYPES = {"resume", "repo"}


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        missing = REQUIRED_METADATA - self.metadata.keys()
        if missing:
            raise ValueError(f"Chunk metadata missing required keys: {missing}")
        if self.metadata["source_type"] not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {VALID_SOURCE_TYPES}, "
                f"got {self.metadata['source_type']!r}"
            )


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]
