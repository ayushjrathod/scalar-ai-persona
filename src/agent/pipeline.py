import logging

from livekit.agents import Agent, AgentSession, InterruptionOptions, TurnHandlingOptions
from livekit.plugins import openai as openai_plugin
from livekit.plugins import sarvam
from livekit.plugins import silero
from livekit.plugins.turn_detector.english import EnglishModel, _EUORunnerEn  # _EUORunnerEn registers its inference runner at import time

from agent.persona import build_system_prompt

logger = logging.getLogger(__name__)

_GREETING = (
    "Hi, this is Ayush's AI representative. "
    "I'm here to answer questions about his background and experience. "
    "What would you like to know?"
)


class PersonaAgent(Agent):
    """Ayush's voice persona. on_enter speaks the opening greeting."""

    async def on_enter(self) -> None:
        # The caller hears this as soon as the SIP participant joins the room.
        self.session.say(_GREETING)


def build_pipeline() -> tuple[AgentSession, PersonaAgent]:
    from utils.config import settings

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
    )

    # Bulbul:v3, ritu — TTFB ~212ms measured; ChunkedStream keeps first-audio < 2s.
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
            preemptive_generation={"enabled": True, "preemptive_tts": True},
        ),
    )

    agent = PersonaAgent(instructions=build_system_prompt())

    logger.info(
        "pipeline built: STT=%s LLM=%s TTS=%s speaker=%s lang=%s turn_det=EnglishModel endpointing_min=0.3s preemptive_tts=True",
        settings.STT_MODEL,
        settings.LLM_MODEL,
        settings.TTS_MODEL,
        settings.TTS_SPEAKER,
        settings.TTS_LANGUAGE,
    )

    return session, agent
