"""
Unit tests for all 8 agents.
Uses mocking to avoid real API and DB calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from graph.state import GraphRAGState
from agents.query_planner.query_planner import QueryPlannerAgent
from agents.evidence_fusion.evidence_fusion_agent import EvidenceFusionAgent
from agents.citation_builder.citation_builder import CitationBuilderAgent
from agents.reflection.reflection_agent import ReflectionAgent
from vectorstore.faiss_store.faiss_store import DocumentChunk


# ── Query Planner Tests ───────────────────────────────────────

class TestQueryPlannerAgent:

    @patch("agents.query_planner.query_planner.anthropic")
    def setup_method(self, method, mock_anthropic):
        self.agent = QueryPlannerAgent()

    def test_classifies_simple_query(self):
        result = self.agent._classify_complexity(
            "What is Python?"
        )
        assert result == "simple"

    def test_classifies_complex_query(self):
        result = self.agent._classify_complexity(
            "Analyse the skill gap for a senior ML engineer role"
        )
        assert result == "complex"

    def test_classifies_long_query_as_complex(self):
        result = self.agent._classify_complexity(
            "I want to transition from software engineering "
            "to machine learning what steps should I take"
        )
        assert result == "complex"

    def test_plan_with_rules_skill_type(self):
        plan = self.agent._plan_with_rules(
            "What Python skills do I need?",
            ["Python"]
        )
        assert plan["query_type"] == "skill"
        assert "Python" in plan["entities"]

    def test_plan_with_rules_career_type(self):
        plan = self.agent._plan_with_rules(
           "What career roles and salary can I get?",
           ["ML Engineer"]
        )
        assert plan["query_type"] == "career"


    def test_plan_with_rules_project_type(self):
        plan = self.agent._plan_with_rules(
            "How do I build a RAG project?",
            ["RAG"]
        )
        assert plan["query_type"] == "project"

    def test_plan_with_rules_research_type(self):
        plan = self.agent._plan_with_rules(
            "What does this research paper say about transformers?",
            ["Transformer"]
        )
        assert plan["query_type"] == "research"

    def test_run_updates_state(self):
        state = GraphRAGState(
            question="What is Python?"
        )
        with patch.object(
            self.agent,
            "_plan_with_rules",
            return_value={
                "query_type": "skill",
                "entities": ["Python"],
                "sub_questions": [],
                "complexity": "simple",
                "requires_graph": True,
                "requires_multi_hop": False,
                "retrieval_strategies": [
                    "vector", "graph", "bm25"
                ]
            }
        ):
            result = self.agent.run(state)
            assert result.query_type == "skill"
            assert result.complexity in ["simple", "complex"]


# ── Evidence Fusion Agent Tests ───────────────────────────────

class TestEvidenceFusionAgent:

    def setup_method(self):
        self.agent = EvidenceFusionAgent()

    def _make_result(
        self,
        chunk_id: str,
        text: str,
        method: str
    ) -> dict:
        return {
            "chunk": DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                source="test",
                source_type="test"
            ),
            "score": 0.8,
            "retrieval_method": method,
            "rank": 1
        }

    def test_fuses_results(self):
        state = GraphRAGState(
            question="What is Python?",
            faiss_results=[
                self._make_result("c1", "Python ML", "faiss")
            ],
            bm25_results=[
                self._make_result("c2", "Python code", "bm25")
            ]
        )
        result = self.agent.run(state)
        assert len(result.fused_results) > 0

    def test_builds_context(self):
        state = GraphRAGState(
            question="What is Python?",
            faiss_results=[
                self._make_result(
                    "c1", "Python is a programming language", "faiss"
                )
            ],
            graph_context="Python is related to ML"
        )
        result = self.agent.run(state)
        assert len(result.context) > 0

    def test_empty_results(self):
        state = GraphRAGState(
            question="What is Python?",
            faiss_results=[],
            bm25_results=[],
            pinecone_results=[]
        )
        result = self.agent.run(state)
        assert result.fusion_confidence == 0.0

    def test_graph_context_included(self):
        state = GraphRAGState(
            question="What is Python?",
            faiss_results=[],
            graph_context="Python requires Statistics"
        )
        result = self.agent.run(state)
        assert "Python requires Statistics" in result.context


# ── Citation Builder Tests ────────────────────────────────────

class TestCitationBuilderAgent:

    def setup_method(self):
        self.agent = CitationBuilderAgent()

    def _make_fused_result(self, chunk_id: str) -> dict:
        return {
            "chunk": DocumentChunk(
                chunk_id=chunk_id,
                text="Test content about Python and ML",
                source="arxiv",
                source_type="arxiv"
            ),
            "score": 0.8,
            "retrieval_methods": ["faiss", "bm25"],
            "rank": 1,
            "multi_source": True
        }

    def test_builds_citations(self):
        state = GraphRAGState(
            question="What is Python?",
            answer="Python is a programming language.",
            fused_results=[
                self._make_fused_result("c1"),
                self._make_fused_result("c2")
            ],
            query_type="skill",
            reflection_score=0.8,
            reflection_loops=1,
            claim_score=0.9,
            fusion_confidence=0.8
        )
        result = self.agent.run(state)
        assert len(result.citations) > 0

    def test_assigns_verified_confidence(self):
        state = GraphRAGState(
            question="test",
            answer="test answer",
            fused_results=[self._make_fused_result("c1")],
            fusion_confidence=0.9,
            graph_coverage=0.9,
            claim_score=0.9,
            reflection_score=0.9,
            query_type="skill",
            reflection_loops=1
        )
        result = self.agent.run(state)
        assert result.confidence_band == "VERIFIED"

    def test_assigns_low_confidence(self):
        state = GraphRAGState(
            question="test",
            answer="test answer",
            fused_results=[],
            fusion_confidence=0.2,
            graph_coverage=0.2,
            claim_score=0.2,
            reflection_score=0.2,
            query_type="skill",
            reflection_loops=1
        )
        result = self.agent.run(state)
        assert result.confidence_band == "LOW"

    def test_builds_final_answer(self):
        state = GraphRAGState(
            question="test",
            answer="Python is great for ML.",
            fused_results=[self._make_fused_result("c1")],
            query_type="skill",
            reflection_score=0.8,
            reflection_loops=1,
            claim_score=0.9,
            fusion_confidence=0.8
        )
        result = self.agent.run(state)
        assert len(result.final_answer) > 0

    def test_graph_citation_added(self):
        state = GraphRAGState(
            question="test",
            answer="test answer",
            fused_results=[],
            graph_context="Python related to ML",
            graph_coverage=0.8,
            query_type="skill",
            reflection_score=0.7,
            reflection_loops=1,
            claim_score=0.8,
            fusion_confidence=0.0
        )
        result = self.agent.run(state)
        graph_citations = [
            c for c in result.citations
            if c["source_type"] == "graph"
        ]
        assert len(graph_citations) > 0


# ── Reflection Agent Tests ────────────────────────────────────

class TestReflectionAgent:

    @patch("agents.reflection.reflection_agent.anthropic")
    def setup_method(self, method, mock_anthropic):
        self.agent = ReflectionAgent()

    def test_skips_no_context_answer(self):
        state = GraphRAGState(
            question="test",
            answer="The available context does not contain "
                   "enough information to answer this accurately.",
            context="some context"
        )
        result = self.agent.run(state)
        assert result.reflection_score == 1.0
        assert result.reflection_passed is True

    def test_skips_empty_answer(self):
        state = GraphRAGState(
            question="test",
            answer="",
            context="some context"
        )
        result = self.agent.run(state)
        assert result.reflection_passed is False

    def test_max_loops_exits(self):
        from config.settings import settings
        state = GraphRAGState(
            question="test",
            answer="some answer",
            context="some context",
            reflection_loops=settings.reflection_max_loops
        )
        result = self.agent.run(state)
        assert result.reflection_passed is True


# ── Graph State Tests ─────────────────────────────────────────

class TestGraphRAGState:

    def test_default_state(self):
        state = GraphRAGState(question="test")
        assert state.question == "test"
        assert state.query_type == "skill"
        assert state.complexity == "simple"
        assert state.reflection_loops == 0
        assert state.error is None

    def test_state_update(self):
        state = GraphRAGState(question="test")
        state.query_type = "career"
        state.entities = ["Python", "PyTorch"]
        assert state.query_type == "career"
        assert len(state.entities) == 2

    def test_confidence_band_default(self):
        state = GraphRAGState(question="test")
        assert state.confidence_band == "LOW"

    def test_metadata_default(self):
        state = GraphRAGState(question="test")
        assert isinstance(state.metadata, dict)
        assert len(state.metadata) == 0
