from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Literal

import openai
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.monitoring import TurnMetrics, persist_turn
from agent.persona import build_chat_prompt
from agent.tools.calendar import CalendarTools
from rag.retrieve import format_for_chat, retrieve
from utils.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Injected when retrieval returns nothing
_NO_CONTEXT_HINT = (
    "No relevant facts were found for this question. "
    "If you cannot answer from your representative knowledge, "
    "say: 'I don't have specific details on that.'"
)

# OpenAI function schemas for the three calendar tools.
_CALENDAR_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Look up open slots on Ayush's calendar. Call this first whenever the "
                "user wants to schedule, book, or set up a meeting or interview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_hint": {
                        "type": "string",
                        "description": (
                            "The day in natural language: 'tomorrow', 'next Tuesday', "
                            "'this week', 'Thursday afternoon'."
                        ),
                    }
                },
                "required": ["date_hint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collect_contact_info",
            "description": (
                "Return the question to ask the user for their name and email "
                "before booking. Call this once a slot has been agreed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_slot",
            "description": (
                "Confirm a booking after the user agreed to a specific slot and "
                "provided name and email. Never book without both."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime_str": {
                        "type": "string",
                        "description": "The agreed slot, e.g. 'Thursday June 12 at 2 PM'.",
                    },
                    "caller_name": {"type": "string", "description": "User's full name."},
                    "caller_email": {"type": "string", "description": "User's email address."},
                },
                "required": ["datetime_str", "caller_name", "caller_email"],
            },
        },
    },
]


# Request model

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8192)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4096)
    session_id: str = Field(default="anonymous", max_length=256)
    # Cap client-supplied history at 10 exchanges (20 messages) to prevent prompt bloat.
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


# Helpers
def _enrich_query(query: str, history: list[dict]) -> str:
    """Prepend 'Ayush Rathod' if absent; for short follow-up queries, prefix
    recent exchange text so retrieval has enough context."""
    enriched = query if "ayush" in query.lower() else f"Ayush Rathod {query}"
    if len(query.split()) <= 5 and history:
        tail = " ".join(h.get("content", "")[:150] for h in history[-2:]).strip()
        if tail:
            enriched = f"{tail} {enriched}"
    return enriched


def _extract_sources(results: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        label = r.metadata.get("file_path") or r.metadata.get("project") or "unknown"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


async def _execute_calendar_tool(
    ct: CalendarTools, name: str, args: dict
) -> tuple[str, dict | None]:
    """Call the CalendarTools method by name and return (result_text, optional_sse_action)."""
    if name == "get_available_slots":
        date_hint = args.get("date_hint", "")
        result = await ct.get_available_slots(date_hint)
        return result, {"action": "get_slots", "params": {"date_hint": date_hint, "result": result}}
    if name == "collect_contact_info":
        result = await ct.collect_contact_info()
        return result, None
    if name == "book_slot":
        result = await ct.book_slot(
            args.get("datetime_str", ""),
            args.get("caller_name", ""),
            args.get("caller_email", ""),
        )
        return result, {"action": "booking_confirmed", "params": {**args, "result": result}}
    return f"Unknown tool: {name}", None


# Core SSE generator

async def _sse_stream(
    message: str,
    session_id: str,
    history: list[dict],
) -> AsyncGenerator[str, None]:
    t_start = time.perf_counter()
    retrieval_latency: float | None = None
    llm_ttft: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    sources: list[str] = []

    try:
        # --- Retrieval ---
        enriched = _enrich_query(message, history)
        t_ret = time.perf_counter()
        try:
            result = await asyncio.to_thread(retrieve, enriched, settings.RAG_CHAT_TOP_K)
        except Exception as exc:
            logger.warning("chat retrieval failed (session=%s): %s", session_id, exc)
            result = None
        retrieval_latency = time.perf_counter() - t_ret

        if result and not result.low_confidence:
            context = format_for_chat(result)
            sources = _extract_sources(result.results)
            system_prompt = build_chat_prompt(context or None)
        else:
            system_prompt = build_chat_prompt(_NO_CONTEXT_HINT)

        logger.info(
            "chat retrieval | %.0fms hits=%d low_conf=%s session=%s q=%r",
            retrieval_latency * 1000,
            len(result.results) if result else 0,
            result.low_confidence if result else None,
            session_id,
            enriched[:80],
        )

        # --- Build message list ---
        # Cap history at 10 exchanges (20 messages); client may send more.
        capped = history[-20:]
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for h in capped:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        # --- Streaming LLM + tool-call loop ---
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        ct = CalendarTools()
        t_first_token: float | None = None

        for _depth in range(5): 
            stream = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=_CALENDAR_TOOLS,
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
            )

            finish_reason: str | None = None
            tool_calls_map: dict[int, dict] = {}
            assistant_content = ""

            async for chunk in stream:
                if chunk.usage:
                    prompt_tokens += chunk.usage.prompt_tokens or 0
                    completion_tokens += chunk.usage.completion_tokens or 0

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        llm_ttft = t_first_token - t_start
                    assistant_content += delta.content
                    yield f"data: {json.dumps({'delta': delta.content, 'done': False})}\n\n"

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": "", "name": "", "args": ""}
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            # name arrives once in the first chunk; arguments stream in pieces
                            if tc.function.name:
                                tool_calls_map[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["args"] += tc.function.arguments

            if finish_reason != "tool_calls" or not tool_calls_map:
                break

            # Append the assistant message that contains the tool call(s)
            ordered = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]
            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in ordered
                ],
            })

            # Execute each tool and feed results back into the message list
            for tc in ordered:
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                    tool_result, action_event = await _execute_calendar_tool(
                        ct, tc["name"], args
                    )
                except Exception as exc:
                    logger.exception(
                        "calendar tool %s failed (session=%s): %s",
                        tc["name"], session_id, exc,
                    )
                    tool_result = "Tool error — please try again."
                    action_event = None

                if action_event:
                    yield f"data: {json.dumps(action_event)}\n\n"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        yield f"data: {json.dumps({'delta': '', 'done': True, 'sources': sources})}\n\n"

    except Exception as exc:
        logger.exception("chat stream error (session=%s): %s", session_id, exc)
        yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    finally:
        total_time = time.perf_counter() - t_start
        try:
            turn = TurnMetrics(
                timestamp=time.time(),
                responded=True,
                retrieval_latency=retrieval_latency,
                llm_ttft=llm_ttft,
                e2e_latency=total_time,
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                source="chat",
                session_id=session_id,
            )
            persist_turn(turn)
        except Exception as exc:
            logger.warning("failed to persist chat metrics (session=%s): %s", session_id, exc)


# Routes
@router.post("/message")
async def chat_message(req: ChatRequest, request: Request) -> StreamingResponse:
    history = [h.model_dump() for h in req.history]

    async def generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _sse_stream(req.message, req.session_id, history):
                if await request.is_disconnected():
                    logger.info("chat client disconnected (session=%s)", req.session_id)
                    return
                yield chunk
        finally:
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def chat_health() -> dict:
    qdrant_configured = bool(settings.QDRANT_URL and settings.QDRANT_API_KEY)
    rag_reachable = False
    if qdrant_configured:
        try:
            async with asyncio.timeout(2.0):
                from rag.retrieve import _get_store
                await asyncio.to_thread(lambda: _get_store().search("health", k=1))
            rag_reachable = True
        except Exception:
            pass

    if not qdrant_configured:
        rag_status = "not configured"
    elif rag_reachable:
        rag_status = "connected"
    else:
        rag_status = "unreachable"

    cal_ok = bool(settings.CALCOM_API_KEY)
    return {
        "status": "ok",
        "rag": rag_status,
        "calendar": "connected" if cal_ok else "not configured",
    }
