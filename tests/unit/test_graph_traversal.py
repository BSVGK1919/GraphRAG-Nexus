"""
Unit tests for Graph Traversal Engine.
Uses mocking to avoid real Neo4j connection.
"""

import pytest
from unittest.mock import MagicMock, patch
from knowledge_graph.graph_traversal import GraphTraversalEngine


class TestGraphTraversalEngine:

    @patch("knowledge_graph.graph_traversal.GraphDatabase")
    def setup_method(self, method, mock_db):
        mock_driver = MagicMock()
        mock_db.driver.return_value = mock_driver
        self.engine = GraphTraversalEngine()
        self.engine.driver = mock_driver

    def _mock_session(self, return_data: list[dict]):
        """Helper to mock Neo4j session."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([
                MagicMock(**{"__iter__": MagicMock(
                    return_value=iter(row.items())
                ), "keys": MagicMock(return_value=row.keys()),
                   "__getitem__": MagicMock(
                       side_effect=row.__getitem__
                   )})
                for row in return_data
            ])
        )
        mock_session.run.return_value = mock_result
        self.engine.driver.session.return_value.__enter__ = (
            MagicMock(return_value=mock_session)
        )
        self.engine.driver.session.return_value.__exit__ = (
            MagicMock(return_value=False)
        )
        return mock_session

    def test_query_returns_structure(self):
        result = self.engine._empty_result("test question")
        assert "question" in result
        assert "graph_context" in result
        assert "graph_coverage" in result
        assert "entities_found" in result

    def test_empty_result_structure(self):
        result = self.engine._empty_result("What is Python?")
        assert result["graph_coverage"] == 0.0
        assert result["graph_context"] == ""
        assert result["entities_found"] == []

    def test_coverage_calculation_full(self):
        queries = [
            {"entity": "Python"},
            {"entity": "PyTorch"}
        ]
        results = [
            {"entity": "Python", "results": [{"data": 1}]},
            {"entity": "PyTorch", "results": [{"data": 2}]}
        ]
        coverage = self.engine._calculate_coverage(
            results, queries
        )
        assert coverage == 1.0

    def test_coverage_calculation_partial(self):
        queries = [
            {"entity": "Python"},
            {"entity": "PyTorch"}
        ]
        results = [
            {"entity": "Python", "results": [{"data": 1}]}
        ]
        coverage = self.engine._calculate_coverage(
            results, queries
        )
        assert coverage == 0.5

    def test_coverage_empty(self):
        coverage = self.engine._calculate_coverage([], [])
        assert coverage == 0.0

    def test_format_result_skills_for_role(self):
        result = self.engine._format_result(
            "skills_for_role",
            "ML Engineer",
            [{"skills": ["Python", "PyTorch", "TensorFlow"]}]
        )
        assert "ML Engineer" in result
        assert "Python" in result

    def test_format_result_roles_for_skill(self):
        result = self.engine._format_result(
            "roles_for_skill",
            "Python",
            [{"roles": ["ML Engineer", "Data Scientist"]}]
        )
        assert "Python" in result
        assert "ML Engineer" in result

    def test_format_result_empty_data(self):
        result = self.engine._format_result(
            "skills_for_role",
            "ML Engineer",
            []
        )
        assert result == ""

    def test_build_context_empty(self):
        context = self.engine._build_context([], "test")
        assert context == ""

    def test_build_context_with_results(self):
        results = [{
            "template": "skills_for_role",
            "entity": "ML Engineer",
            "results": [{"skills": ["Python", "PyTorch"]}]
        }]
        context = self.engine._build_context(
            results, "test question"
        )
        assert len(context) > 0
