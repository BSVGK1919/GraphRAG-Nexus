"""
GraphRAG-Nexus — Knowledge Graph Ontology
Defines all entities, relationships and RDF namespaces.
"""

from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

# ── RDF Namespaces ────────────────────────────────────────────
GRN = Namespace("https://graphrag-nexus.ai/ontology#")
GRN_DATA = Namespace("https://graphrag-nexus.ai/data#")

# ── Entity Types ─────────────────────────────────────────────
class EntityType:
    SKILL = GRN.Skill
    ROLE = GRN.Role
    COMPANY = GRN.Company
    TOOL = GRN.Tool
    FRAMEWORK = GRN.Framework
    DOMAIN = GRN.Domain
    COURSE = GRN.Course
    PAPER = GRN.Paper
    JOB = GRN.Job
    LOCATION = GRN.Location
    CONCEPT = GRN.Concept
    DATASET = GRN.Dataset
    MODEL = GRN.Model
    LIBRARY = GRN.Library

# ── Relationship Types ────────────────────────────────────────
class RelationType:
    # Skill relationships
    REQUIRED_FOR = GRN.required_for
    PART_OF = GRN.part_of
    RELATED_TO = GRN.related_to
    PREREQUISITE_OF = GRN.prerequisite_of
    ENABLES = GRN.enables

    # Role relationships
    OFFERED_BY = GRN.offered_by
    REQUIRES_SKILL = GRN.requires_skill
    PAYS_SALARY = GRN.pays_salary
    LOCATED_IN = GRN.located_in

    # Tool/Framework relationships
    IS_A = GRN.is_a
    DEVELOPED_BY = GRN.developed_by
    USED_FOR = GRN.used_for
    USED_IN = GRN.used_in
    ALTERNATIVE_TO = GRN.alternative_to
    BUILT_ON = GRN.built_on
    SUPPORTS = GRN.supports

    # Domain relationships
    SUBFIELD_OF = GRN.subfield_of
    APPLIES_TO = GRN.applies_to

    # Course/Paper relationships
    TEACHES = GRN.teaches
    INTRODUCES = GRN.introduces
    PUBLISHED_BY = GRN.published_by
    AUTHORED_BY = GRN.authored_by

    # Job relationships
    REQUIRES = GRN.requires
    POSTED_BY = GRN.posted_by
    HAS_SALARY = GRN.has_salary

# ── Neo4j Node Labels ─────────────────────────────────────────
NEO4J_LABELS = {
    "Skill": ["Skill"],
    "Role": ["Role"],
    "Company": ["Company"],
    "Tool": ["Tool"],
    "Framework": ["Framework", "Tool"],
    "Domain": ["Domain"],
    "Course": ["Course"],
    "Paper": ["Paper"],
    "Job": ["Job"],
    "Location": ["Location"],
    "Concept": ["Concept"],
    "Dataset": ["Dataset"],
    "Model": ["Model"],
    "Library": ["Library", "Tool"],
}

# ── Neo4j Relationship Types ──────────────────────────────────
NEO4J_RELATIONSHIPS = [
    "REQUIRED_FOR",
    "PART_OF",
    "RELATED_TO",
    "PREREQUISITE_OF",
    "ENABLES",
    "OFFERED_BY",
    "REQUIRES_SKILL",
    "LOCATED_IN",
    "IS_A",
    "DEVELOPED_BY",
    "USED_FOR",
    "USED_IN",
    "ALTERNATIVE_TO",
    "BUILT_ON",
    "SUPPORTS",
    "SUBFIELD_OF",
    "APPLIES_TO",
    "TEACHES",
    "INTRODUCES",
    "PUBLISHED_BY",
    "AUTHORED_BY",
    "REQUIRES",
    "POSTED_BY",
    "HAS_SALARY",
]

# ── Entity Properties ─────────────────────────────────────────
ENTITY_PROPERTIES = {
    "Skill": ["name", "domain", "level", "description", "source"],
    "Role": ["name", "seniority", "salary_min", "salary_max",
             "description", "source"],
    "Company": ["name", "location", "size", "industry", "source"],
    "Tool": ["name", "category", "version", "language", "source"],
    "Framework": ["name", "language", "use_case", "version", "source"],
    "Domain": ["name", "parent_domain", "description", "source"],
    "Course": ["name", "provider", "level", "url", "source"],
    "Paper": ["title", "authors", "year", "url", "abstract", "source"],
    "Job": ["title", "company", "location", "salary_min",
            "salary_max", "url", "source"],
    "Location": ["name", "country", "region"],
    "Concept": ["name", "domain", "description", "source"],
    "Dataset": ["name", "domain", "size", "url", "source"],
    "Model": ["name", "architecture", "domain", "url", "source"],
    "Library": ["name", "language", "category", "version", "source"],
}
