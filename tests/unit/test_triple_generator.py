"""
Unit tests for RDF Triple Generator.
"""

import pytest
from ingestion.extractors.entity_extractor import Entity, Relationship
from ingestion.extractors.triple_generator import TripleGenerator


class TestTripleGenerator:

    def setup_method(self):
        self.generator = TripleGenerator()

    def test_add_single_entity(self):
        entity = Entity(
            name="Python",
            entity_type="Skill",
            confidence=0.9,
            source="test"
        )
        added = self.generator.add_entities([entity])
        assert added == 1
        assert len(self.generator.graph) > 0

    def test_add_multiple_entities(self):
        entities = [
            Entity(name="Python", entity_type="Skill",
                   confidence=0.9, source="test"),
            Entity(name="PyTorch", entity_type="Framework",
                   confidence=0.95, source="test"),
            Entity(name="ML Engineer", entity_type="Role",
                   confidence=0.9, source="test"),
        ]
        added = self.generator.add_entities(entities)
        assert added == 3

    def test_add_relationship(self):
        relationship = Relationship(
            subject="Python",
            subject_type="Skill",
            predicate="REQUIRED_FOR",
            object="ML Engineer",
            object_type="Role",
            confidence=0.9,
            source="test"
        )
        added = self.generator.add_relationships([relationship])
        assert added == 1

    def test_graph_stats(self):
        entity = Entity(
            name="TensorFlow",
            entity_type="Framework",
            confidence=0.95,
            source="test"
        )
        self.generator.add_entities([entity])
        stats = self.generator.get_stats()
        assert "total_triples" in stats
        assert stats["total_triples"] > 0

    def test_serialize_turtle(self, tmp_path):
        entity = Entity(
            name="NLP",
            entity_type="Domain",
            confidence=0.9,
            source="test"
        )
        self.generator.add_entities([entity])
        output_path = str(tmp_path / "test.ttl")
        self.generator.serialize(output_path, format="turtle")
        import os
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_entity_type_mapping(self):
        entities = [
            Entity(name="FAISS", entity_type="Tool",
                   confidence=0.9, source="test"),
            Entity(name="Neo4j", entity_type="Tool",
                   confidence=0.9, source="test"),
            Entity(name="London", entity_type="Location",
                   confidence=0.8, source="test"),
        ]
        added = self.generator.add_entities(entities)
        assert added == 3
        assert len(self.generator.graph) > 0

    def test_merge_graphs(self):
        gen1 = TripleGenerator()
        gen2 = TripleGenerator()

        gen1.add_entities([
            Entity(name="Python", entity_type="Skill",
                   confidence=0.9, source="test")
        ])
        gen2.add_entities([
            Entity(name="PyTorch", entity_type="Framework",
                   confidence=0.9, source="test")
        ])

        triples_before = len(gen1.graph)
        gen1.merge(gen2)
        assert len(gen1.graph) > triples_before

    def test_empty_entities(self):
        added = self.generator.add_entities([])
        assert added == 0
        assert len(self.generator.graph) == 0
