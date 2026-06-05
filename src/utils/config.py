from pydantic import field_validator
from pydantic_settings import BaseSettings


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
    METRICS_STORE_PATH: str = "/tmp/metrics.jsonl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

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


settings = Settings()
