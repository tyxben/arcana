"""
Arcana: FastAPI Integration

Shows how to embed Arcana Runtime in a production web service.
Runtime is created once at startup, reused across requests.

Run:
    pip install fastapi uvicorn
    export DEEPSEEK_API_KEY=sk-xxx
    uvicorn examples.08_fastapi_integration:app --reload

Test:
    curl -X POST http://localhost:8000/agent \
         -H "Content-Type: application/json" \
         -d '{"goal": "What is Python?"}'
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import arcana

# Create Runtime ONCE at startup (connection reuse)
runtime = arcana.Runtime(
    providers={"deepseek": os.environ.get("DEEPSEEK_API_KEY", "")},
    budget=arcana.Budget(max_cost_usd=0.5),
    trace=True,
)

app = FastAPI(title="Arcana Agent API")


class AgentRequest(BaseModel):
    goal: str
    max_turns: int = 10


class AgentResponse(BaseModel):
    output: str
    success: bool
    steps: int
    tokens: int
    cost_usd: float


@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Run an agent task."""
    try:
        result = await runtime.run(request.goal, max_turns=request.max_turns)
        return AgentResponse(
            output=str(result.output),
            success=result.success,
            steps=result.steps,
            tokens=result.tokens_used,
            cost_usd=result.cost_usd,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "providers": runtime.providers,
        "tools": runtime.tools,
    }
