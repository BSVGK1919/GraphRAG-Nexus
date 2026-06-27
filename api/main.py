"""
GraphRAG-Nexus — FastAPI Application
Main API entry point.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from config.settings import settings

logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise pipeline on startup."""
    global pipeline
    logger.info(f"🚀 {settings.app_name} v{settings.app_version} starting...")
    try:
        from graph.graph_builder import GraphRAGPipeline
        pipeline = GraphRAGPipeline()
        logger.info("✅ Pipeline initialised")
    except Exception as e:
        logger.error(f"❌ Pipeline init failed: {e}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Request/Response Models ───────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = ""


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: list
    confidence_band: str
    reflection_score: float
    reflection_loops: int
    claim_score: float
    graph_coverage: float
    llm_provider: str
    latency_ms: float
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "model": settings.anthropic_model,
        "embedding": settings.embedding_model,
        "pipeline_ready": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint."""
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised"
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    result = pipeline.run(
        question=request.question,
        session_id=request.session_id
    )

    result["answer"] = result.get("answer") or ""
    result["query_type"] = result.get("query_type") or "unknown"
    result["confidence_band"] = result.get("confidence_band") or "low"
    result["llm_provider"] = result.get("llm_provider") or "unknown"
    result["sources"] = result.get("sources") or []
    return QueryResponse(**result)


@app.get("/stats")
async def stats():
    """Return system statistics."""
    if not pipeline:
        return {"status": "pipeline not ready"}

    try:
        faiss_stats = pipeline.retrieval_orchestrator\
            .faiss_store.get_stats()
        bm25_stats = pipeline.retrieval_orchestrator\
            .bm25_store.get_stats()

        return {
            "faiss": faiss_stats,
            "bm25": bm25_stats,
            "pipeline": "ready"
        }
    except Exception as e:
        return {"error": str(e)}
