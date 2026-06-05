import json
import logging
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession

logger = logging.getLogger(__name__)

# kept the price tracking in usd and not inr
_INR_TO_USD: float = 1 / 95.0

# Sarvam STT (Saaras:v3): ₹30 per hour of audio
SARVAM_STT_COST_PER_SEC: float = (30.0 / 3600.0) * _INR_TO_USD  # $/sec

# Sarvam TTS (Bulbul:v3): ₹30 per 10,000 characters
SARVAM_TTS_COST_PER_CHAR: float = (30.0 / 10_000.0) * _INR_TO_USD  # $/char

# GPT-4.1-mini: $0.4/1M input tokens, $1.60/1M output tokens (USD)
GPT_INPUT_USD_PER_1M: float = 0.4  
GPT_OUTPUT_USD_PER_1M: float = 1.6
GPT_COST_PER_INPUT_TOKEN: float = GPT_INPUT_USD_PER_1M / 1_000_000
GPT_COST_PER_OUTPUT_TOKEN: float = GPT_OUTPUT_USD_PER_1M / 1_000_000

RING_BUFFER_SIZE: int = 100
METRICS_API_LIMIT: int = 100


@dataclass
class TurnMetrics:
    """Latency and cost record for one completed voice turn."""

    timestamp: float

    # Latency — sourced from ChatMessage.metrics (MetricsReport, pre-computed by LiveKit)
    e2e_latency: float         # user stopped speaking → agent audio started
    eou_delay: float           # VAD end-of-speech → EOU decision
    transcription_delay: float # EOU decision → transcript ready
    llm_ttft: float            # LLM time to first token
    tts_ttfb: float            # TTS time to first byte (first segment only)

    # Token counts — from LLMMetrics (paired by recency)
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_cached_tokens: int

    # TTS — aggregated across all segments for this turn
    tts_characters: int
    interrupted: bool  # True if TTS was cancelled mid-playback

    # STT audio — from STTMetrics (paired by recency)
    stt_audio_secs: float

    # Cost estimate
    estimated_cost_usd: float


@dataclass
class _InFlightTTS:
    """Accumulates TTS segments until the turn boundary fires."""
    ttfb: float           # first segment's ttfb
    total_chars: int = 0
    interrupted: bool = False


class MetricsCollector:
    def __init__(self) -> None:
        self._buffer: deque[TurnMetrics] = deque(maxlen=RING_BUFFER_SIZE)

        # Queues are LIFO (pop from right) so we get the most recent match.
        self._pending_llm: deque[Any] = deque(maxlen=10)   # LLMMetrics
        self._pending_stt: deque[Any] = deque(maxlen=10)   # STTMetrics
        self._in_flight_tts: Optional[_InFlightTTS] = None # accumulates until turn seals

        self._total_cost_usd: float = 0.0

    # Event handlers

    def handle_plugin_metrics(self, metric: Any) -> None:
        from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

        if isinstance(metric, STTMetrics):
            self._pending_stt.append(metric)

        elif isinstance(metric, LLMMetrics):
            self._pending_llm.append(metric)

        elif isinstance(metric, TTSMetrics):
            # First TTS segment of a new turn initialises the in-flight tracker.
            if self._in_flight_tts is None:
                self._in_flight_tts = _InFlightTTS(ttfb=metric.ttfb)
            self._in_flight_tts.total_chars += metric.characters_count
            if metric.cancelled:
                self._in_flight_tts.interrupted = True

    def handle_conversation_item(self, event: Any) -> None:
        """Handler for "conversation_item_added" events — seals one turn."""
        from livekit.agents.llm.chat_context import ChatMessage

        item = event.item
        if not isinstance(item, ChatMessage) or item.role != "assistant":
            return

        report = item.metrics  # MetricsReport TypedDict
        self._seal(report)

    def handle_close(self, _event: Any) -> None:
        """Handler for "close" events — flushes remaining in-flight data."""
        # If the call ended mid-turn (e.g. interrupted), seal with whatever we have.
        if self._in_flight_tts is not None or self._pending_llm:
            self._seal({})

    # Internal 

    def _seal(self, report: dict) -> None:
        """Commit a TurnMetrics record from the current pending state."""
        llm = self._pending_llm.pop() if self._pending_llm else None
        stt = self._pending_stt.pop() if self._pending_stt else None
        tts = self._in_flight_tts
        self._in_flight_tts = None

        e2e = float(report.get("e2e_latency") or 0.0)
        eou = float(report.get("end_of_turn_delay") or 0.0)
        tr_delay = float(report.get("transcription_delay") or 0.0)
        llm_ttft = float(report.get("llm_node_ttft") or (llm.ttft if llm else 0.0))
        tts_ttfb = float(report.get("tts_node_ttfb") or (tts.ttfb if tts else 0.0))

        prompt_tok = llm.prompt_tokens if llm else 0
        completion_tok = llm.completion_tokens if llm else 0
        cached_tok = llm.prompt_cached_tokens if llm else 0

        tts_chars = tts.total_chars if tts else 0
        interrupted = tts.interrupted if tts else False
        stt_audio = stt.audio_duration if stt else 0.0

        cost = self._estimate_cost(stt_audio, tts_chars, prompt_tok, completion_tok)
        self._total_cost_usd += cost

        turn = TurnMetrics(
            timestamp=time.time(),
            e2e_latency=e2e,
            eou_delay=eou,
            transcription_delay=tr_delay,
            llm_ttft=llm_ttft,
            tts_ttfb=tts_ttfb,
            llm_prompt_tokens=prompt_tok,
            llm_completion_tokens=completion_tok,
            llm_cached_tokens=cached_tok,
            tts_characters=tts_chars,
            interrupted=interrupted,
            stt_audio_secs=stt_audio,
            estimated_cost_usd=round(cost, 6),
        )
        self._buffer.append(turn)
        logger.info(json.dumps({"event": "turn_metrics", **asdict(turn)}))
        persist_turn(turn)

    @staticmethod
    def _estimate_cost(
        stt_audio_secs: float,
        tts_chars: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        return (
            stt_audio_secs * SARVAM_STT_COST_PER_SEC
            + tts_chars * SARVAM_TTS_COST_PER_CHAR
            + prompt_tokens * GPT_COST_PER_INPUT_TOKEN
            + completion_tokens * GPT_COST_PER_OUTPUT_TOKEN
        )

    # Public API

    def get_summary(self) -> dict[str, Any]:
        """Return p50/p95 stats per stage + cost totals. Fed to GET /metrics."""
        return summary_from_turns(list(self._buffer))

# Helpers

def persist_turn(turn: TurnMetrics, path: str | None = None) -> None:
    if path is None:
        from utils.config import settings
        path = settings.METRICS_STORE_PATH
    try:
        with open(path, "a") as f:
            f.write(json.dumps(asdict(turn)) + "\n")
    except OSError as exc:
        logger.warning("Failed to persist turn metrics to %s: %s", path, exc)


def load_turns(path: str | None = None, limit: int = METRICS_API_LIMIT) -> list[TurnMetrics]:
    if path is None:
        from utils.config import settings
        path = settings.METRICS_STORE_PATH
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    turns: list[TurnMetrics] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(TurnMetrics(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Skipping malformed metrics line: %s | error: %s", line[:120], exc)
    return turns


def summary_from_turns(turns: list[TurnMetrics]) -> dict[str, Any]:
    total_cost = sum(t.estimated_cost_usd for t in turns)
    base: dict[str, Any] = {
        "turn_count": len(turns),
        "total_cost_usd": round(total_cost, 4),
    }
    if not turns:
        return base

    def stats(vals: list[float]) -> dict[str, float]:
        return {
            "p50": round(statistics.median(vals), 3),
            "p95": round(statistics.quantiles(vals, n=100)[94], 3),
        }

    return base | {
        "e2e_latency_secs": stats([t.e2e_latency for t in turns]),
        "llm_ttft_secs": stats([t.llm_ttft for t in turns]),
        "tts_ttfb_secs": stats([t.tts_ttfb for t in turns]),
        "transcription_delay_secs": stats([t.transcription_delay for t in turns]),
        "eou_delay_secs": stats([t.eou_delay for t in turns]),
        "interruption_rate": round(
            sum(1 for t in turns if t.interrupted) / len(turns), 3
        ),
        "avg_tokens": {
            "prompt": round(statistics.mean(t.llm_prompt_tokens for t in turns), 1),
            "completion": round(statistics.mean(t.llm_completion_tokens for t in turns), 1),
            "cached": round(statistics.mean(t.llm_cached_tokens for t in turns), 1),
        },
        "recent_turns": [asdict(t) for t in turns[-10:]],
    }


def attach_metrics(
    session: "AgentSession",
    collector: MetricsCollector,
    *,
    stt: Any,
    llm: Any,
    tts: Any,
) -> None:
    stt.on("metrics_collected", collector.handle_plugin_metrics)
    llm.on("metrics_collected", collector.handle_plugin_metrics)
    tts.on("metrics_collected", collector.handle_plugin_metrics)

    session.on("conversation_item_added", collector.handle_conversation_item)
    session.on("close", collector.handle_close)
