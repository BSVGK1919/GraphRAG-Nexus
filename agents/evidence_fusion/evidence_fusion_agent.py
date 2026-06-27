"""
GraphRAG-Nexus — Agent 4: Evidence Fusion Agent
Merges results from all retrieval paths into
a unified, ranked context for generation.
"""

import time
import logging
from graph.state import GraphRAGState
from vectorstore.evidence_fusion import EvidenceFusion
from config.settings import settings

logger = logging.getLogger(__name__)


class EvidenceFusionAgent:
    """
    Agent 4 — Evidence Fusion

    Responsibilities:
    - Merge FAISS + BM25 + Pinecone results
    - Apply RRF scoring
    - Build unified context string
    - Calculate fusion confidence
    - Add graph context to unified context
    """

    def __init__(self):
        self.fusion = EvidenceFusion()

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run evidence fusion agent."""
        start_time = time.time()
        logger.info("[EvidenceFusion] Fusing evidence...")

        try:
            # Collect all retrieval results
            result_lists = []

            if state.faiss_results:
                result_lists.append(state.faiss_results)
            if state.bm25_results:
                result_lists.append(state.bm25_results)
            if state.pinecone_results:
                result_lists.append(state.pinecone_results)

            if result_lists:
                # Fuse using RRF
                state.fused_results = self.fusion.fuse(
                    result_lists,
                    top_k=settings.retrieval_top_k
                )
                state.fusion_confidence = (
                    self.fusion.calculate_confidence(
                        state.fused_results
                    )
                )
            else:
                state.fused_results = []
                state.fusion_confidence = 0.0

            # Build unified context
            state.context = self._build_context(state)

        except Exception as e:
            logger.error(f"[EvidenceFusion] Failed: {e}")
            state.fused_results = []
            state.fusion_confidence = 0.0
            state.context = state.graph_context

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[EvidenceFusion] "
            f"fused={len(state.fused_results)} results "
            f"confidence={state.fusion_confidence:.2f} "
            f"({latency:.0f}ms)"
        )
        return state

    def _build_context(self, state: GraphRAGState) -> str:
        """Build unified context from all sources."""
        context_parts = []

        # Add vector/keyword context
        if state.fused_results:
            chunk_texts = []
            for i, result in enumerate(
                state.fused_results[:5], 1
            ):
                chunk = result["chunk"]
                text = (
                    chunk.text
                    if hasattr(chunk, "text")
                    else str(chunk)
                )
                source = (
                    chunk.source
                    if hasattr(chunk, "source")
                    else "unknown"
                )
                chunk_texts.append(
                    f"[Source {i} — {source}]\n{text}"
                )

            if chunk_texts:
                context_parts.append(
                    "\n\n".join(chunk_texts)
                )

        # Add graph context
        if state.graph_context:
            context_parts.append(
                f"[Knowledge Graph Context]\n"
                f"{state.graph_context}"
            )

        if not context_parts:
            return ""

        return "\n\n---\n\n".join(context_parts)
