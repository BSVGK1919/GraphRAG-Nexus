"""
GraphRAG-Nexus — RDF Triple Generator
Converts extracted entities and relationships
into RDF triples using rdflib.
"""

import logging
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from knowledge_graph.schema.ontology import GRN, GRN_DATA, EntityType, RelationType
from ingestion.extractors.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)


def _make_uri(name: str, namespace) -> URIRef:
    """Convert a name to a valid URI."""
    clean = name.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
    return namespace[clean]


class TripleGenerator:
    """
    Converts extracted entities and relationships
    into RDF triples stored in an rdflib Graph.
    """

    def __init__(self):
        self.graph = Graph()
        self.graph.bind("grn", GRN)
        self.graph.bind("grn_data", GRN_DATA)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.triple_count = 0

    def add_entities(self, entities: list[Entity]) -> int:
        """Add entities as RDF triples."""
        added = 0
        for entity in entities:
            try:
                self._add_entity(entity)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add entity {entity.name}: {e}")
        self.triple_count += added
        return added

    def add_relationships(self, relationships: list[Relationship]) -> int:
        """Add relationships as RDF triples."""
        added = 0
        for rel in relationships:
            try:
                self._add_relationship(rel)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add relationship: {e}")
        self.triple_count += added
        return added

    def _add_entity(self, entity: Entity):
        """Add a single entity as RDF triples."""
        uri = _make_uri(entity.name, GRN_DATA)
        entity_type = self._get_rdf_type(entity.entity_type)

        # Type triple
        self.graph.add((uri, RDF.type, entity_type))

        # Label triple
        self.graph.add((uri, RDFS.label, Literal(entity.name)))

        # Source triple
        if entity.source:
            self.graph.add((
                uri,
                GRN.source,
                Literal(entity.source)
            ))

        # Confidence triple
        self.graph.add((
            uri,
            GRN.confidence,
            Literal(entity.confidence, datatype=XSD.float)
        ))

        # Additional properties
        for key, value in entity.properties.items():
            prop_uri = GRN[key]
            if isinstance(value, (int, float)):
                self.graph.add((
                    uri,
                    prop_uri,
                    Literal(value, datatype=XSD.float)
                ))
            else:
                self.graph.add((
                    uri,
                    prop_uri,
                    Literal(str(value))
                ))

    def _add_relationship(self, rel: Relationship):
        """Add a single relationship as RDF triple."""
        subject_uri = _make_uri(rel.subject, GRN_DATA)
        object_uri = _make_uri(rel.object, GRN_DATA)
        predicate_uri = self._get_rdf_predicate(rel.predicate)

        # Main relationship triple
        self.graph.add((subject_uri, predicate_uri, object_uri))

        # Reification — add metadata about the relationship
        stmt = BNode()
        self.graph.add((stmt, RDF.type, RDF.Statement))
        self.graph.add((stmt, RDF.subject, subject_uri))
        self.graph.add((stmt, RDF.predicate, predicate_uri))
        self.graph.add((stmt, RDF.object, object_uri))
        self.graph.add((
            stmt,
            GRN.confidence,
            Literal(rel.confidence, datatype=XSD.float)
        ))
        if rel.source:
            self.graph.add((stmt, GRN.source, Literal(rel.source)))

    def _get_rdf_type(self, entity_type: str) -> URIRef:
        """Map entity type string to RDF URI."""
        type_map = {
            "Skill": EntityType.SKILL,
            "Role": EntityType.ROLE,
            "Company": EntityType.COMPANY,
            "Tool": EntityType.TOOL,
            "Framework": EntityType.FRAMEWORK,
            "Domain": EntityType.DOMAIN,
            "Course": EntityType.COURSE,
            "Paper": EntityType.PAPER,
            "Job": EntityType.JOB,
            "Location": EntityType.LOCATION,
            "Concept": EntityType.CONCEPT,
            "Dataset": EntityType.DATASET,
            "Model": EntityType.MODEL,
            "Library": EntityType.LIBRARY,
        }
        return type_map.get(entity_type, GRN.Entity)

    def _get_rdf_predicate(self, predicate: str) -> URIRef:
        """Map predicate string to RDF URI."""
        predicate_map = {
            "REQUIRED_FOR": RelationType.REQUIRED_FOR,
            "PART_OF": RelationType.PART_OF,
            "RELATED_TO": RelationType.RELATED_TO,
            "PREREQUISITE_OF": RelationType.PREREQUISITE_OF,
            "ENABLES": RelationType.ENABLES,
            "OFFERED_BY": RelationType.OFFERED_BY,
            "REQUIRES_SKILL": RelationType.REQUIRES_SKILL,
            "LOCATED_IN": RelationType.LOCATED_IN,
            "IS_A": RelationType.IS_A,
            "DEVELOPED_BY": RelationType.DEVELOPED_BY,
            "USED_FOR": RelationType.USED_FOR,
            "USED_IN": RelationType.USED_IN,
            "ALTERNATIVE_TO": RelationType.ALTERNATIVE_TO,
            "BUILT_ON": RelationType.BUILT_ON,
            "SUPPORTS": RelationType.SUPPORTS,
            "SUBFIELD_OF": RelationType.SUBFIELD_OF,
            "APPLIES_TO": RelationType.APPLIES_TO,
            "TEACHES": RelationType.TEACHES,
            "INTRODUCES": RelationType.INTRODUCES,
            "REQUIRES": RelationType.REQUIRES,
            "POSTED_BY": RelationType.POSTED_BY,
            "HAS_SALARY": RelationType.HAS_SALARY,
        }
        return predicate_map.get(predicate, GRN[predicate.lower()])

    def serialize(self, output_path: str, format: str = "turtle") -> str:
        """Serialize the RDF graph to a file."""
        self.graph.serialize(destination=output_path, format=format)
        logger.info(
            f"Serialized {len(self.graph)} triples to {output_path}"
        )
        return output_path

    def get_stats(self) -> dict:
        """Return statistics about the graph."""
        return {
            "total_triples": len(self.graph),
            "entities_added": self.triple_count,
            "subjects": len(set(self.graph.subjects())),
            "predicates": len(set(self.graph.predicates())),
            "objects": len(set(self.graph.objects())),
        }

    def merge(self, other_graph: "TripleGenerator"):
        """Merge another triple generator's graph into this one."""
        self.graph += other_graph.graph
        logger.info(
            f"Merged graphs. Total triples: {len(self.graph)}"
        )
