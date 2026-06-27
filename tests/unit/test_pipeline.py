"""
Unit tests for LangGraph pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
from graph.state import GraphRAGState
from graph.graph_builder import should_rewrite, should_use_graph


class TestPipelineConditions:

    def test_should_rewrite_when_failed(self):
        state = GraphRAGState(
            question="test",
            answer="some answer",
            reflection_passed=False,
            reflection_loops=0,
            reflection_score=0.3
        )
        result = should_rewrite(state)
        assert result == "rewrite"

    def test_should_continue_when_passed(self):
        state = GraphRAGState(
            question="test",
            answer="some answer",
            reflection_passed=True,
            reflection_loops=0,
            reflection_score=0.9
        )
        result = should_rewrite(state)
        assert result == "continue"

    def test_should_continue_max_loops(self):
        from config.settings import settings
        state = GraphRAGState(
            question="test",
            answer="some answer",
            reflection_passed=False,
            reflection_loops=settings.reflection_max_loops,
            reflection_score=0.3
        )
        result = should_rewrite(state)
        assert result == "continue"

    def test_should_continue_no_context_answer(self):
        state = GraphRAGState(
            question="test",
            answer="The available context does not contain "
                   "enough information to answer this accurately.",
            reflection_passed=False,
            reflection_loops=0
        )
        result = should_rewrite(state)
        assert result == "continue"

    def test_should_use_graph_with_entities(self):
        state = GraphRAGState(
            question="test",
            requires_graph=True,
            entities=["Python", "PyTorch"]
        )
        result = should_use_graph(state)
        assert result == "graph"

    def test_should_skip_graph_no_entities(self):
        state = GraphRAGState(
            question="test",
            requires_graph=True,
            entities=[]
        )
        result = should_use_graph(state)
        assert result == "skip_graph"

    def test_should_skip_graph_not_required(self):
        state = GraphRAGState(
            question="test",
            requires_graph=False,
            entities=["Python"]
        )
        result = should_use_graph(state)
        assert result == "skip_graph"

    def test_rewrite_increments_loop_count(self):
        state = GraphRAGState(
            question="test",
            answer="some answer",
            reflection_passed=False,
            reflection_loops=0,
            reflection_score=0.3
        )
        initial_loops = state.reflection_loops
        should_rewrite(state)
        assert state.reflection_loops == initial_loops + 1

    def test_state_initialisation(self):
        state = GraphRAGState(question="What is Python?")
        assert state.question == "What is Python?"
        assert state.reflection_loops == 0
        assert state.confidence_band == "LOW"
        assert state.error is None

    def test_state_full_flow(self):
        state = GraphRAGState(question="test")
        state.query_type = "skill"
        state.entities = ["Python"]
        state.answer = "Python is great"
        state.reflection_score = 0.9
        state.reflection_passed = True
        state.confidence_band = "HIGH"

        assert state.query_type == "skill"
        assert state.reflection_passed is True
        assert state.confidence_band == "HIGH"
