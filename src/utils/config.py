from pathlib import Path
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # LiveKit connection
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str

    # External service keys
    SARVAM_API_KEY: str
    OPENAI_API_KEY: str

    # Model selection
    LLM_MODEL: str = "gpt-4.1-mini"
    STT_MODEL: str = "saaras:v3"
    TTS_MODEL: str = "bulbul:v3"
    TTS_SPEAKER: str = "ritu"
    TTS_LANGUAGE: str = "en-IN"

    # Worker / dispatch
    AGENT_NAME: str = "ayush-persona"  # must match the SIP dispatch rule in LiveKit Cloud

    # Logging
    LOG_LEVEL: str = "INFO"

    # Shared metrics store (worker writes after call, API reads most recent)
    METRICS_STORE_PATH: str = str(_PROJECT_ROOT / "metrics" / "metrics.jsonl")

    # RAG
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "persona"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMS: int = 1536
    RAG_EMBED_BATCH_SIZE: int = 96
    # Retrieval tuning
    RAG_DEFAULT_TOP_K: int = 6
    RAG_VOICE_TOP_K: int = 6
    RAG_CHAT_TOP_K: int = 6
    # Cosine relevance floor per chunk. Off-topic ~0.15, on-topic voice queries ~0.24-0.30.
    RAG_RELEVANCE_FLOOR: float = 0.20
    # Over-fetch to compensate for multi-rep dedup collapsing raw top-k to fewer distinct chunks.
    RAG_OVERFETCH_FACTOR: int = 4
    RAG_OVERFETCH_MIN: int = 20
    # Timeout for the voice retrieval path..
    RAG_RETRIEVAL_TIMEOUT_S: float = 3.0

    # Calender booking
    CALCOM_API_KEY: Optional[str] = None
    CALCOM_USERNAME: Optional[str] = None
    CALCOM_EVENT_SLUG: Optional[str] = None
    CALCOM_API_BASE: str = "https://api.cal.com/v2"
    CALCOM_EVENT_TYPES_API_VERSION: str = "2024-06-14"
    CALCOM_SLOTS_API_VERSION: str = "2024-09-04"
    CALCOM_BOOKINGS_API_VERSION: str = "2024-08-13"
    CALCOM_TIMEZONE: str = "Asia/Kolkata"
    CALCOM_BOOKING_LANGUAGE: str = "en"
    CALCOM_MAX_SLOTS_OFFERED: int = 4
    CALCOM_SLOT_SEARCH_DAYS: int = 7

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}

    @field_validator("LIVEKIT_URL")
    @classmethod
    def validate_livekit_url(cls, v: str) -> str:
        if not v.startswith(("ws://", "wss://")):
            raise ValueError("LIVEKIT_URL must start with 'ws://' or 'wss://'")
        return v

    @field_validator("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "SARVAM_API_KEY")
    @classmethod
    def validate_not_empty(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'")
        return v

    @field_validator("QDRANT_URL")
    @classmethod
    def validate_qdrant_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must start with 'http://' or 'https://'")
        return v

    @model_validator(mode="after")
    def validate_qdrant_credentials(self) -> "Settings":
        url_set = self.QDRANT_URL is not None
        key_set = self.QDRANT_API_KEY is not None
        if url_set != key_set:
            missing = "QDRANT_API_KEY" if url_set else "QDRANT_URL"
            raise ValueError(
                f"{missing} must be set when the other Qdrant credential is provided"
            )
        return self


settings = Settings()
