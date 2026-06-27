"""
GraphRAG-Nexus — Shared Agent State
Defines the state object passed between all agents
in the LangGraph pipeline.
"""

from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class GraphRAGState:
    """
    Shared state passed through all 8 agents.
    Each agent reads from and writes to this state.
    """

    # ── Input ─────────────────────────────────────
    question: str = ""
    session_id: str = ""

    # ── Query Planning (Agent 1) ──────────────────
    query_type: str = "skill"
    entities: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    complexity: str = "simple"
    requires_graph: bool = True
    requires_multi_hop: bool = False
    retrieval_strategies: list[str] = field(
        default_factory=lambda: ["vector", "graph", "bm25"]
    )

    # ── Retrieval (Agent 2) ───────────────────────
    faiss_results: list[dict] = field(default_factory=list)
    pinecone_results: list[dict] = field(default_factory=list)
    bm25_results: list[dict] = field(default_factory=list)
    retrieval_confidence: float = 0.0

    # ── Graph Reasoning (Agent 3) ─────────────────
    graph_context: str = ""
    graph_results: dict = field(default_factory=dict)
    graph_coverage: float = 0.0
    graph_entities_found: list[str] = field(
        default_factory=list
    )

    # ── Evidence Fusion (Agent 4) ─────────────────
    fused_results: list[dict] = field(default_factory=list)
    fusion_confidence: float = 0.0
    context: str = ""

    # ── Generation (Agent 5) ─────────────────────
    answer: str = ""
    llm_provider: str = "claude"
    llm_fallback_used: bool = False

    # ── Claim Verification (Agent 6) ─────────────
    claims: list[dict] = field(default_factory=list)
    verified_count: int = 0
    unverified_count: int = 0
    contradicted_count: int = 0
    claim_score: float = 0.0

    # ── Reflection (Agent 7) ──────────────────────
    reflection_score: float = 0.0
    reflection_feedback: str = ""
    reflection_passed: bool = False
    reflection_loops: int = 0

    # ── Citation Building (Agent 8) ───────────────
    citations: list[dict] = field(default_factory=list)
    confidence_band: str = "LOW"
    final_answer: str = ""

    # ── Meta ──────────────────────────────────────
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
