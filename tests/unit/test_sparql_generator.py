"""
Unit tests for SPARQL Query Generator.
"""

import pytest
from knowledge_graph.sparql.sparql_generator import SPARQLGenerator


class TestSPARQLGenerator:

    def setup_method(self):
        self.generator = SPARQLGenerator()

    def test_generates_queries_for_skill_type(self):
        queries = self.generator.generate(
            "What skills do I need for ML?",
            query_type="skill",
            entities=["Python"]
        )
        assert len(queries) > 0
        templates = [q["template"] for q in queries]
        assert "entity_neighbourhood" in templates

    def test_generates_queries_for_career_type(self):
        queries = self.generator.generate(
            "What roles require Python?",
            query_type="career",
            entities=["Python"]
        )
        assert len(queries) > 0
        templates = [q["template"] for q in queries]
        assert "roles_for_skill" in templates

    def test_generates_prerequisite_query_for_learning(self):
        queries = self.generator.generate(
            "What do I need to learn PyTorch?",
            query_type="skill",
            entities=["PyTorch"]
        )
        templates = [q["template"] for q in queries]
        assert "skill_prerequisites" in templates

    def test_generates_job_query_for_job_question(self):
        queries = self.generator.generate(
            "What jobs require Python skills?",
            query_type="career",
            entities=["Python"]
        )
        templates = [q["template"] for q in queries]
        assert "jobs_for_skills" in templates

    def test_extracts_entities_automatically(self):
        queries = self.generator.generate(
            "What skills do I need for machine learning?",
            query_type="skill"
        )
        assert len(queries) > 0

    def test_fallback_for_no_entities(self):
        queries = self.generator.generate(
            "hello",
            query_type="skill",
            entities=[]
        )
        assert isinstance(queries, list)

    def test_query_structure(self):
        queries = self.generator.generate(
            "Python for ML",
            query_type="skill",
            entities=["Python"]
        )
        for query in queries:
            assert "query" in query
            assert "params" in query
            assert "template" in query
            assert "entity" in query

    def test_multiple_entities(self):
        queries = self.generator.generate(
            "Compare Python and PyTorch",
            query_type="skill",
            entities=["Python", "PyTorch"]
        )
        entities = [q["entity"] for q in queries]
        assert "Python" in entities
        assert "PyTorch" in entities

    def test_project_query_type(self):
        queries = self.generator.generate(
            "What tools for NLP projects?",
            query_type="project",
            entities=["NLP"]
        )
        templates = [q["template"] for q in queries]
        assert "tools_for_domain" in templates

    def test_research_query_type(self):
        queries = self.generator.generate(
            "Research on transformer models",
            query_type="research",
            entities=["Transformer"]
        )
        assert len(queries) > 0
