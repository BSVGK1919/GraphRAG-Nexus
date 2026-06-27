"""
GraphRAG-Nexus — Agent 8: Citation Builder
Builds full source attribution and assigns
confidence band to the final answer.
"""

import time
import logging
from graph.state import GraphRAGState
from config.settings import settings

logger = logging.getLogger(__name__)


class CitationBuilderAgent:
    """
    Agent 8 — Citation Builder

    Responsibilities:
    - Build source citations from fused results
    - Add graph path citations
    - Assign confidence band
    - Build final answer with citations
    """

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run citation builder agent."""
        start_time = time.time()
        logger.info("[CitationBuilder] Building citations...")

        try:
            # Build citations from fused results
            citations = []

            for i, result in enumerate(
                state.fused_results[:5], 1
            ):
                chunk = result["chunk"]
                citation = {
                    "index": i,
                    "source": getattr(chunk, "source", "unknown"),
                    "source_type": getattr(
                        chunk, "source_type", "unknown"
                    ),
                    "retrieval_methods": result.get(
                        "retrieval_methods", []
                    ),
                    "score": result.get("score", 0.0),
                    "text_preview": (
                        getattr(chunk, "text", "")[:200]
                        if hasattr(chunk, "text") else ""
                    )
                }
                citations.append(citation)

            # Add graph citation if graph context exists
            if state.graph_context:
                citations.append({
                    "index": len(citations) + 1,
                    "source": "knowledge_graph",
                    "source_type": "graph",
                    "retrieval_methods": ["graph"],
                    "score": state.graph_coverage,
                    "text_preview": state.graph_context[:200]
                })

            state.citations = citations

            # Assign confidence band
            state.confidence_band = self._assign_confidence(
                state
            )

            # Build final answer
            state.final_answer = self._build_final_answer(state)

        except Exception as e:
            logger.error(f"[CitationBuilder] Failed: {e}")
            state.final_answer = state.answer
            state.confidence_band = "LOW"

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[CitationBuilder] "
            f"citations={len(state.citations)} "
            f"confidence={state.confidence_band} "
            f"({latency:.0f}ms)"
        )
        return state

    def _assign_confidence(
        self,
        state: GraphRAGState
    ) -> str:
        """Assign confidence band based on all scores."""
        scores = []

        if state.fusion_confidence > 0:
            scores.append(state.fusion_confidence)
        if state.graph_coverage > 0:
            scores.append(state.graph_coverage)
        if state.claim_score > 0:
            scores.append(state.claim_score)
        if state.reflection_score > 0:
            scores.append(state.reflection_score)

        if not scores:
            return "LOW"

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.85:
            return "VERIFIED"
        elif avg_score >= 0.70:
            return "HIGH"
        elif avg_score >= 0.55:
            return "MEDIUM"
        else:
            return "LOW"

    def _build_final_answer(
        self,
        state: GraphRAGState
    ) -> str:
        """Build final answer with source attribution."""
        if not state.answer:
            return ""

        answer = state.answer

        # Add sources if available
        if state.citations:
            source_list = []
            for citation in state.citations[:5]:
                source_type = citation["source_type"]
                source = citation["source"]
                if source_type != "graph":
                    source_list.append(
                        f"• {source} ({source_type})"
                    )
                else:
                    source_list.append(
                        "• Knowledge Graph"
                    )

            if source_list:
                sources_text = "\n".join(source_list)
                answer += (
                    f"\n\n📚 Sources "
                    f"({len(state.citations)}) | "
                    f"Type: {state.query_type} | "
                    f"Confidence: {state.confidence_band} | "
                    f"Reflection: "
                    f"{state.reflection_loops} loop(s) | "
                    f"Score: {state.reflection_score:.2f}"
                )

        return answer
