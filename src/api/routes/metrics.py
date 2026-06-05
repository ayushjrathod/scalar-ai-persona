from fastapi import APIRouter
from agent.monitoring import load_turns, summary_from_turns

router = APIRouter()


@router.get("")
async def get_metrics():
    turns = load_turns()
    return summary_from_turns(turns)
