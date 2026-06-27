"""
GraphRAG-Nexus — Agent 3: Graph Reasoner
Queries knowledge graph and performs
multi-hop reasoning over entities.
"""

import time
import logging
from graph.state import GraphRAGState
from knowledge_graph.graph_traversal import GraphTraversalEngine
from config.settings import settings

logger = logging.getLogger(__name__)


class GraphReasonerAgent:
    """
    Agent 3 — Graph Reasoner

    Responsibilities:
    - Query Neo4j knowledge graph
    - Perform multi-hop traversal
    - Extract graph context
    - Calculate graph coverage score
    """

    def __init__(self):
        self.traversal_engine = GraphTraversalEngine()

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run graph reasoning agent."""
        start_time = time.time()
        logger.info(
            f"[GraphReasoner] Querying graph for: "
            f"{state.question[:60]}"
        )

        if not state.requires_graph:
            logger.info("[GraphReasoner] Graph not required")
            return state

        try:
            # Query knowledge graph
            graph_result = self.traversal_engine.query(
                question=state.question,
                query_type=state.query_type,
                entities=state.entities
            )

            state.graph_context = graph_result["graph_context"]
            state.graph_results = graph_result
            state.graph_coverage = graph_result["graph_coverage"]
            state.graph_entities_found = graph_result[
                "entities_found"
            ]

            # Multi-hop traversal if needed
            if (state.requires_multi_hop
                    and len(state.entities) >= 2):
                hop_result = self.traversal_engine.multi_hop_traverse(
                    start_entity=state.entities[0],
                    end_entity=state.entities[1],
                    max_hops=settings.max_graph_hops
                )
                if hop_result["found"]:
                    hop_context = self._format_hop_result(
                        hop_result
                    )
                    state.graph_context += f"\n\n{hop_context}"

        except Exception as e:
            logger.error(f"[GraphReasoner] Failed: {e}")
            state.graph_context = ""
            state.graph_coverage = 0.0

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[GraphReasoner] coverage={state.graph_coverage} "
            f"context_len={len(state.graph_context)} "
            f"({latency:.0f}ms)"
        )
        return state

    def _format_hop_result(self, hop_result: dict) -> str:
        """Format multi-hop traversal result as text."""
        if not hop_result["paths"]:
            return ""

        path = hop_result["paths"][0]
        nodes = path.get("path_nodes", [])
        rels = path.get("relationships", [])

        if not nodes:
            return ""

        path_text = " → ".join([
            f"{nodes[i]} --[{rels[i]}]--> "
            if i < len(rels) else nodes[i]
            for i in range(len(nodes))
        ])

        return (
            f"Knowledge graph path from "
            f"{hop_result['start']} to {hop_result['end']}: "
            f"{path_text}"
        )
