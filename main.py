from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core.agent import Agent
# from lang_core.agent import agent_lang
from lang_core_multi.agent import agent_lang
from lang_core_multi.logger import get_agent_logger
from lang_core_multi.mcp.mcp_client import initialize_mcp_infrastructure, shutdown_mcp_infrastructure

logger = get_agent_logger(__name__)
logger.info("=== Agent system starting up ===")

# ── FastAPI Lifespan Manager ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of application-wide background infrastructure.
    Guarantees the MCP server is initialized BEFORE the web server accepts traffic
    and safely winds down child processes when FastAPI shuts down.
    """
    logger.info("=== Starting up background infrastructure ===")
    try:
        # 1. Boot up the server process and register standard IO channels
        await initialize_mcp_infrastructure()
        logger.info("=== MCP background infrastructure ready ===")
        yield
    finally:
        # 2. Automatically triggers during app termination (Ctrl+C / SIGTERM)
        logger.info("=== Cleaning up background infrastructure ===")
        await shutdown_mcp_infrastructure()
        logger.info("=== Cleanup complete ===")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str

# @app.post("/chat", response_model=QueryResponse)
# def chat(request: QueryRequest):
#     agent = Agent()
#     result = agent.agent_logic(request.query)
#     return QueryResponse(answer=result)

@app.post("/chat_langg", response_model=QueryResponse)
async def chat_langg(request: QueryRequest):
    logger.info("Received query: %s", request.query)
    try:
        result = await agent_lang.ainvoke(
            {
                "query":request.query
            }
        )
        logger.info("Query completed successfully. Answer length: %d chars", len(result["answer"]))
        return QueryResponse(answer=result["answer"])
    except Exception:
        logger.error("Unhandled exception while processing query: %s", request.query, exc_info=True)
        raise
