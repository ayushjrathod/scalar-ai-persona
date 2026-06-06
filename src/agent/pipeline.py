import asyncio
import logging
import time

import anyio.to_thread
from livekit.agents import Agent, AgentSession, InterruptionOptions, TurnHandlingOptions
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai as openai_plugin
from livekit.plugins import sarvam
from livekit.plugins import silero
from livekit.plugins.turn_detector.english import EnglishModel, _EUORunnerEn  # _EUORunnerEn registers its inference runner at import time

from agent.monitoring import MetricsCollector, attach_metrics
from agent.persona import build_system_prompt
from agent.tools.calendar import CalendarTools
from rag.retrieve import format_for_voice, prewarm, retrieve
from utils.config import settings

logger = logging.getLogger(__name__)

# Injected when retrieval comes back empty / low-confidence. 
_NO_CONTEXT_NOTE = (
    "No grounded facts were retrieved for this question. If you don't already know the "
    "answer from your role, tell the caller you don't have that detail — do not guess."
)

_GREETING = (
    "Hi, this is Ayush's AI representative. "
    "I'm here to answer questions about his background and experience. I can also help schedule a meeting if you'd like. "
    "What would you like to know?"
)


class PersonaAgent(Agent):
    def __init__(
        self, instructions: str, collector: MetricsCollector, calendar: CalendarTools
    ) -> None:
        super().__init__(instructions=instructions, tools=calendar.function_tools())
        self._collector = collector
        self.calendar = calendar

    async def on_enter(self) -> None:
        # The caller hears this as soon as the SIP participant joins the room.
        self.session.say(_GREETING)
        # Warm the embed + Qdrant TLS connections + calendar behind the greeting
        asyncio.create_task(anyio.to_thread.run_sync(prewarm))
        asyncio.create_task(self.calendar.prewarm())

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        # Inject RAG context into the ephemeral turn_ctx (not the system prompt) so the
        # stable cached prefix is never invalidated.
        query = (new_message.text_content or "").strip()
        if not query:
            return

        # Retrieval is a blocking HTTP path (OpenAI embed + Qdrant search). Offload it to a worker thread
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                anyio.to_thread.run_sync(retrieve, query, settings.RAG_VOICE_TOP_K),
                timeout=settings.RAG_RETRIEVAL_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("retrieval failed (%s) — answering without grounding", exc)
            result = None
        elapsed = time.perf_counter() - started

        context = format_for_voice(result) if result else ""
        if context:
            turn_ctx.add_message(
                role="assistant",
                content=(
                    "Reference facts about Ayush, retrieved for the caller's current "
                    f"question. State only what is grounded here:\n{context}"
                ),
            )
        else:
            turn_ctx.add_message(role="assistant", content=_NO_CONTEXT_NOTE)

        # Retrieval now sits in the critical path before the LLM . A failed
        # retrieval counts as low-confidence so the metric reflects the ungrounded turn.
        self._collector.record_retrieval(
            latency_s=elapsed,
            low_confidence=(result.low_confidence if result else True) or not context,
        )
        logger.info(
            "retrieval | %.0fms hits=%d low_conf=%s q=%r",
            elapsed * 1000,
            len(result.results) if result else 0,
            result.low_confidence if result else None,
            query[:80],
        )


def build_pipeline() -> tuple[AgentSession, PersonaAgent, MetricsCollector]:
    vad = silero.VAD.load()

    # Sarvam Saaras:v3 via WebSocket streaming. api_key passed explicitly because
    # the job runs in a forkserver subprocess that doesn't inherit env vars.
    stt = sarvam.STT(
        model=settings.STT_MODEL,
        language=settings.TTS_LANGUAGE,
        api_key=settings.SARVAM_API_KEY,
    )

    llm = openai_plugin.LLM(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_completion_tokens=80,
    )

    # Bulbul:v3, TTFB ~212ms measured;
    tts = sarvam.TTS(
        model=settings.TTS_MODEL,
        speaker=settings.TTS_SPEAKER,
        target_language_code=settings.TTS_LANGUAGE,
        api_key=settings.SARVAM_API_KEY,
    )

    # EnglishModel scores conversation context to decide if the caller is truly
    # done speaking. _EUORunnerEn registered its inference runner at import time.
    # This lets us drop min_delay from 0.5s → 0.3s without cutting callers off.
    turn_detection = EnglishModel()

    # turn_detection: semantic EOU model — safe to use short endpointing delay.
    # interruption: adaptive mode (ML-based) preferred over VAD-only.
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detection,
            endpointing={"min_delay": 0.3},
            interruption=InterruptionOptions(enabled=True),
            preemptive_generation={"enabled": False},
        ),
    )

    collector = MetricsCollector()
    calendar = CalendarTools(collector=collector)
    agent = PersonaAgent(
        instructions=build_system_prompt(), collector=collector, calendar=calendar
    )
    attach_metrics(session, collector, stt=stt, llm=llm, tts=tts)
    session.on("close", lambda _ev: asyncio.create_task(calendar.aclose()))

    logger.info(
        "pipeline ready: stt=%s llm=%s tts=%s/%s",
        settings.STT_MODEL, settings.LLM_MODEL, settings.TTS_MODEL, settings.TTS_SPEAKER,
    )

    return session, agent, collector
