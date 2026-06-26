"""
Unit tests for Entity Extractor.
"""

import pytest
from ingestion.extractors.entity_extractor import (
    EntityExtractor,
    Entity,
    Relationship
)


class TestEntityExtractor:

    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extracts_skills(self):
        text = "We need Python and PyTorch for this ML role."
        result = self.extractor.extract(text, source="test")
        entity_names = [e.name.lower() for e in result["entities"]]
        assert "python" in entity_names
        assert result["entity_count"] > 0

    def test_extracts_roles(self):
        text = "We are hiring a Senior ML Engineer in London."
        result = self.extractor.extract(text, source="test")
        entity_names = [e.name.lower() for e in result["entities"]]
        assert any("ml engineer" in name for name in entity_names)

    def test_extracts_tools(self):
        text = "The system uses Docker and MLflow for deployment."
        result = self.extractor.extract(text, source="test")
        entity_names = [e.name.lower() for e in result["entities"]]
        assert "docker" in entity_names
        assert "mlflow" in entity_names

    def test_extracts_domains(self):
        text = "This project focuses on NLP and computer vision."
        result = self.extractor.extract(text, source="test")
        entity_names = [e.name.lower() for e in result["entities"]]
        assert any("nlp" in name for name in entity_names)

    def test_extracts_location(self):
        text = "This role is based in London, UK."
        result = self.extractor.extract(text, source="test")
        entity_names = [e.name.lower() for e in result["entities"]]
        assert "london" in entity_names

    def test_extracts_salary(self):
        text = "Salary range is £80,000 to £120,000 per year."
        result = self.extractor.extract(text, source="test")
        salary_entities = [
            e for e in result["entities"]
            if e.entity_type == "Salary"
        ]
        assert len(salary_entities) > 0
        assert salary_entities[0].properties["min"] == 80000
        assert salary_entities[0].properties["max"] == 120000

    def test_deduplicates_entities(self):
        text = "Python is great. Python is widely used. Python rocks."
        result = self.extractor.extract(text, source="test")
        python_entities = [
            e for e in result["entities"]
            if e.name.lower() == "python"
        ]
        assert len(python_entities) == 1

    def test_empty_text(self):
        result = self.extractor.extract("", source="test")
        assert result["entity_count"] == 0
        assert result["relationship_count"] == 0

    def test_source_attached(self):
        text = "PyTorch is a deep learning framework."
        result = self.extractor.extract(text, source="arxiv")
        for entity in result["entities"]:
            assert entity.source == "arxiv"

    def test_returns_dict_structure(self):
        text = "Python and TensorFlow are used for ML."
        result = self.extractor.extract(text, source="test")
        assert "entities" in result
        assert "relationships" in result
        assert "entity_count" in result
        assert "relationship_count" in result
