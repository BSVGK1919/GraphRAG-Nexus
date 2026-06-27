"""
GraphRAG-Nexus — LangGraph Pipeline Builder
Wires all 8 agents into a directed graph
with conditional edges and reflection loops.
"""

import logging
from langgraph.graph import StateGraph, END
from graph.state import GraphRAGState
from agents.query_planner.query_planner import QueryPlannerAgent
from agents.retrieval_orchestrator.retrieval_orchestrator import RetrievalOrchestratorAgent
from agents.graph_reasoner.graph_reasoner import GraphReasonerAgent
from agents.evidence_fusion.evidence_fusion_agent import EvidenceFusionAgent
from agents.generator.generator import GeneratorAgent
from agents.claim_verifier.claim_verifier import ClaimVerifierAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.citation_builder.citation_builder import CitationBuilderAgent
from config.settings import settings

logger = logging.getLogger(__name__)


def should_rewrite(state: GraphRAGState) -> str:
    """
    Conditional edge — decides if answer needs rewriting.
    Returns 'rewrite' or 'continue'.
    """
    if state.reflection_passed:
        return "continue"

    if state.reflection_loops >= settings.reflection_max_loops:
        logger.warning(
            "[Pipeline] Max reflection loops reached"
        )
        return "continue"

    if not state.answer:
        return "continue"

    if "does not contain enough information" in state.answer:
        return "continue"

    logger.info(
        f"[Pipeline] Rewriting — "
        f"score={state.reflection_score:.2f} "
        f"loop={state.reflection_loops}"
    )
    state.reflection_loops += 1
    return "rewrite"


def should_use_graph(state: GraphRAGState) -> str:
    """
    Conditional edge — decides if graph reasoning needed.
    Returns 'graph' or 'skip_graph'.
    """
    if state.requires_graph and state.entities:
        return "graph"
    return "skip_graph"


class GraphRAGPipeline:
    """
    Full GraphRAG pipeline with 8 agents.
    Built using LangGraph StateGraph.
    """

    def __init__(self):
        self.query_planner = QueryPlannerAgent()
        self.retrieval_orchestrator = RetrievalOrchestratorAgent()
        self.graph_reasoner = GraphReasonerAgent()
        self.evidence_fusion = EvidenceFusionAgent()
        self.generator = GeneratorAgent()
        self.claim_verifier = ClaimVerifierAgent()
        self.reflection = ReflectionAgent()
        self.citation_builder = CitationBuilderAgent()
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Build the LangGraph pipeline."""
        workflow = StateGraph(dict)

        # ── Add all agent nodes ───────────────────
        workflow.add_node(
            "query_planner",
            lambda s: vars(self.query_planner.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "retrieval",
            lambda s: vars(self.retrieval_orchestrator.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "graph_reasoner",
            lambda s: vars(self.graph_reasoner.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "evidence_fusion",
            lambda s: vars(self.evidence_fusion.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "generator",
            lambda s: vars(self.generator.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "claim_verifier",
            lambda s: vars(self.claim_verifier.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "reflection",
            lambda s: vars(self.reflection.run(
                GraphRAGState(**s)
            ))
        )
        workflow.add_node(
            "citation_builder",
            lambda s: vars(self.citation_builder.run(
                GraphRAGState(**s)
            ))
        )

        # ── Set entry point ───────────────────────
        workflow.set_entry_point("query_planner")

        # ── Add edges ─────────────────────────────
        workflow.add_edge("query_planner", "retrieval")

        # Conditional: use graph or skip
        workflow.add_conditional_edges(
            "retrieval",
            lambda s: should_use_graph(GraphRAGState(**s)),
            {
                "graph": "graph_reasoner",
                "skip_graph": "evidence_fusion"
            }
        )

        workflow.add_edge("graph_reasoner", "evidence_fusion")
        workflow.add_edge("evidence_fusion", "generator")
        workflow.add_edge("generator", "claim_verifier")
        workflow.add_edge("claim_verifier", "reflection")

        # Conditional: rewrite or continue
        workflow.add_conditional_edges(
            "reflection",
            lambda s: should_rewrite(GraphRAGState(**s)),
            {
                "rewrite": "generator",
                "continue": "citation_builder"
            }
        )

        workflow.add_edge("citation_builder", END)

        return workflow.compile()

    def run(self, question: str, session_id: str = "") -> dict:
        """
        Run the full pipeline for a question.

        Returns structured response dict.
        """
        import time
        start_time = time.time()

        logger.info(f"[Pipeline] Question: {question[:80]}")

        # Build initial state
        initial_state = vars(GraphRAGState(
            question=question,
            session_id=session_id
        ))

        try:
            # Run pipeline
            final_state = self.pipeline.invoke(initial_state)
            state = GraphRAGState(**final_state)

            latency = (time.time() - start_time) * 1000

            response = {
                "answer": state.final_answer or state.answer,
                "query_type": state.query_type,
                "sources": self._format_sources(state),
                "confidence_band": state.confidence_band,
                "reflection_score": state.reflection_score,
                "reflection_loops": state.reflection_loops,
                "claim_score": state.claim_score,
                "graph_coverage": state.graph_coverage,
                "llm_provider": state.llm_provider,
                "latency_ms": latency,
                "error": state.error
            }

            logger.info(
                f"[Pipeline] Done — "
                f"confidence={state.confidence_band} "
                f"score={state.reflection_score:.2f} "
                f"({latency:.0f}ms)"
            )
            return response

        except Exception as e:
            logger.error(f"[Pipeline] Failed: {e}")
            latency = (time.time() - start_time) * 1000
            return {
                "answer": (
                    f"Pipeline error: {str(e)}"
                ),
                "query_type": "skill",
                "sources": [],
                "confidence_band": "LOW",
                "reflection_score": 0.0,
                "reflection_loops": 0,
                "claim_score": 0.0,
                "graph_coverage": 0.0,
                "llm_provider": "none",
                "latency_ms": latency,
                "error": str(e)
            }

    def _format_sources(self, state: GraphRAGState) -> list:
        """Format citations as source list."""
        sources = []
        for citation in state.citations:
            sources.append({
                "text": citation.get("text_preview", ""),
                "source": citation.get("source", ""),
                "source_type": citation.get("source_type", ""),
                "score": citation.get("score", 0.0)
            })
        return sources
