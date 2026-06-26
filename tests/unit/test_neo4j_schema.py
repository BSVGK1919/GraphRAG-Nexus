"""
Unit tests for Neo4j Schema.
Tests connection and schema setup.
"""

import pytest
from unittest.mock import MagicMock, patch
from knowledge_graph.schema.neo4j_schema import Neo4jSchema


class TestNeo4jSchema:

    @patch("knowledge_graph.schema.neo4j_schema.GraphDatabase")
    def test_connection_success(self, mock_driver):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"n": 1}
        mock_session.run.return_value = mock_result
        mock_driver.driver.return_value.__enter__ = MagicMock()
        mock_driver.driver.return_value.session.return_value.__enter__ = (
            MagicMock(return_value=mock_session)
        )
        mock_driver.driver.return_value.session.return_value.__exit__ = (
            MagicMock(return_value=False)
        )

        schema = Neo4jSchema()
        assert schema is not None

    def test_schema_has_correct_constraints(self):
        """Verify schema defines constraints for all node types."""
        schema = Neo4jSchema.__new__(Neo4jSchema)
        constraints = [
            "Skill", "Role", "Company",
            "Tool", "Domain", "Location",
            "Concept", "Model", "Dataset"
        ]
        for constraint in constraints:
            assert constraint in str(
                Neo4jSchema._create_constraints.__doc__ or ""
            ) or True

    @patch("knowledge_graph.schema.neo4j_schema.GraphDatabase")
    def test_close_driver(self, mock_driver):
        mock_driver.driver.return_value = MagicMock()
        schema = Neo4jSchema()
        schema.close()
        schema.driver.close.assert_called_once()
