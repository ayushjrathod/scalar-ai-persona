import logging

from livekit.agents import JobContext, WorkerOptions, cli

from agent.pipeline import build_pipeline

logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext) -> None:
    """Invoked once per inbound call by the LiveKit worker framework."""
    logger.info("call started | room=%s job=%s", ctx.room.name, ctx.job.id)

    # Connect the agent participant to the LiveKit room.
    await ctx.connect()

    # Block until the SIP caller (or any remote participant) has joined.
    caller = await ctx.wait_for_participant()
    logger.info("caller joined | identity=%s", caller.identity)

    # Build the full cascade pipeline. MetricsCollector is already wired
    session, agent, collector = build_pipeline()

    await session.start(agent, room=ctx.room, capture_run=True)

    # Call has ended
    summary = collector.get_summary()
    logger.info(
        "call ended | room=%s turns=%d cost_usd=%.4f e2e_p50=%.3fs",
        ctx.room.name,
        summary.get("turn_count", 0),
        summary.get("total_cost_usd", 0.0),
        summary.get("e2e_latency_secs", {}).get("p50", 0.0),
    )


def main() -> None:
    from utils.config import settings

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=settings.AGENT_NAME,
            ws_url=settings.LIVEKIT_URL,
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET,
        )
    )


if __name__ == "__main__":
    main()
