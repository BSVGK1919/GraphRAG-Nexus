"""
GraphRAG-Nexus — Entity Extractor
Extracts entities and relationships from text using spaCy NER
and rule-based matching for AI/ML domain knowledge.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    entity_type: str
    confidence: float = 1.0
    properties: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class Relationship:
    """Represents an extracted relationship between entities."""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 1.0
    source: str = ""


# ── AI/ML Domain Knowledge ────────────────────────────────────

ML_SKILLS = {
    "python", "pytorch", "tensorflow", "keras", "scikit-learn",
    "numpy", "pandas", "matplotlib", "seaborn", "scipy",
    "hugging face", "transformers", "bert", "gpt", "llm",
    "machine learning", "deep learning", "neural network",
    "nlp", "natural language processing", "computer vision",
    "reinforcement learning", "supervised learning",
    "unsupervised learning", "transfer learning",
    "fine-tuning", "prompt engineering", "rag",
    "retrieval augmented generation", "langchain", "langgraph",
    "faiss", "pinecone", "vector database", "embeddings",
    "attention mechanism", "transformer", "diffusion model",
    "generative ai", "large language model", "mlops",
    "feature engineering", "model deployment", "docker",
    "kubernetes", "aws", "azure", "gcp", "sql", "spark",
    "hadoop", "airflow", "mlflow", "wandb", "dvc",
    "git", "linux", "bash", "scala", "r", "julia",
    "xgboost", "lightgbm", "catboost", "random forest",
    "gradient boosting", "svm", "bayesian", "statistics",
    "linear algebra", "calculus", "probability",
    "data engineering", "etl", "data pipeline",
    "graph neural network", "gnn", "rdf", "sparql", "neo4j",
    "knowledge graph", "ontology", "semantic web",
}

ML_ROLES = {
    "machine learning engineer", "ml engineer",
    "data scientist", "data engineer", "ai engineer",
    "research scientist", "ai researcher",
    "nlp engineer", "computer vision engineer",
    "mlops engineer", "data analyst",
    "principal ml engineer", "senior ml engineer",
    "junior ml engineer", "staff ml engineer",
    "ai lead", "head of ai", "chief ai officer",
    "deep learning engineer", "llm engineer",
}

ML_TOOLS = {
    "pytorch": "Framework",
    "tensorflow": "Framework",
    "keras": "Framework",
    "scikit-learn": "Library",
    "numpy": "Library",
    "pandas": "Library",
    "matplotlib": "Library",
    "hugging face": "Platform",
    "transformers": "Library",
    "langchain": "Framework",
    "langgraph": "Framework",
    "faiss": "Tool",
    "pinecone": "Tool",
    "mlflow": "Tool",
    "wandb": "Tool",
    "docker": "Tool",
    "kubernetes": "Tool",
    "airflow": "Tool",
    "spark": "Framework",
    "neo4j": "Tool",
    "redis": "Tool",
    "postgresql": "Tool",
    "mongodb": "Tool",
    "xgboost": "Library",
    "lightgbm": "Library",
    "spacy": "Library",
    "nltk": "Library",
    "openai": "Platform",
    "anthropic": "Platform",
    "aws": "Platform",
    "azure": "Platform",
    "gcp": "Platform",
}

ML_DOMAINS = {
    "natural language processing", "nlp",
    "computer vision", "cv",
    "reinforcement learning", "rl",
    "time series", "recommender systems",
    "speech recognition", "generative ai",
    "robotics", "autonomous systems",
    "bioinformatics", "healthcare ai",
    "financial ml", "graph machine learning",
}

SALARY_PATTERNS = [
    r"[£$](\d{2,3}),?\d{3}\s*(?:to|-)\s*[£$]?(\d{2,3}),?\d{3}",
    r"(\d{2,3})[,k]\s*(?:to|-)\s*(\d{2,3})[,k]",
]


LOCATION_PATTERNS = [
    "london", "new york", "san francisco", "seattle",
    "berlin", "paris", "toronto", "singapore", "amsterdam",
    "boston", "chicago", "los angeles", "austin", "remote",
    "hybrid", "uk", "us", "usa", "europe",
]


class EntityExtractor:
    """
    Extracts entities and relationships from text
    using rule-based matching for AI/ML domain.
    """

    def __init__(self):
        self.nlp = None
        self._load_spacy()

    def _load_spacy(self):
        """Load spaCy model."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}. Using rule-based only.")
            self.nlp = None

    def extract(self, text: str, source: str = "") -> dict:
        """
        Extract entities and relationships from text.

        Returns:
            dict with 'entities' and 'relationships' lists
        """
        text_lower = text.lower()

        entities = []
        relationships = []

        # Extract skills
        for skill in ML_SKILLS:
            if skill in text_lower:
                entities.append(Entity(
                    name=skill.title(),
                    entity_type="Skill",
                    confidence=0.9,
                    source=source
                ))

        # Extract roles
        for role in ML_ROLES:
            if role in text_lower:
                entities.append(Entity(
                    name=role.title(),
                    entity_type="Role",
                    confidence=0.9,
                    source=source
                ))

        # Extract tools with their types
        for tool, tool_type in ML_TOOLS.items():
            if tool in text_lower:
                entities.append(Entity(
                    name=tool.title(),
                    entity_type=tool_type,
                    confidence=0.95,
                    source=source
                ))

        # Extract domains
        for domain in ML_DOMAINS:
            if domain in text_lower:
                entities.append(Entity(
                    name=domain.title(),
                    entity_type="Domain",
                    confidence=0.9,
                    source=source
                ))

        # Extract locations
        for location in LOCATION_PATTERNS:
            if location in text_lower:
                entities.append(Entity(
                    name=location.title(),
                    entity_type="Location",
                    confidence=0.8,
                    source=source
                ))

        # Extract salaries
        salary = self._extract_salary(text)
        if salary:
            entities.append(Entity(
                name=f"Salary_{salary['min']}_{salary['max']}",
                entity_type="Salary",
                confidence=0.95,
                properties=salary,
                source=source
            ))

        # Extract relationships using spaCy if available
        if self.nlp:
            spacy_relationships = self._extract_relationships_spacy(
                text, entities, source
            )
            relationships.extend(spacy_relationships)

        # Extract rule-based relationships
        rule_relationships = self._extract_relationships_rules(
            text_lower, entities, source
        )
        relationships.extend(rule_relationships)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return {
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships)
        }

    def _extract_salary(self, text: str) -> Optional[dict]:
        """Extract salary information from text."""
        for pattern in SALARY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                if min_val < 1000:
                    min_val *= 1000
                if max_val < 1000:
                    max_val *= 1000
                return {"min": min_val, "max": max_val}
        return None

    def _extract_relationships_rules(
        self,
        text: str,
        entities: list,
        source: str
    ) -> list:
        """Extract relationships using rule-based patterns."""
        relationships = []

        # Pattern: "X requires Y" or "X needs Y"
        requires_patterns = [
            r"(\w+(?:\s\w+)?)\s+(?:requires?|needs?)\s+(\w+(?:\s\w+)?)"
        ]

        # Pattern: "X is used for Y"
        used_for_patterns = [
            r"(\w+(?:\s\w+)?)\s+is\s+used\s+for\s+(\w+(?:\s\w+)?)"
        ]

        # Pattern: "X is a Y"
        is_a_patterns = [
            r"(\w+(?:\s\w+)?)\s+is\s+a(?:n)?\s+(\w+(?:\s\w+)?)"
        ]

        entity_names = {e.name.lower(): e for e in entities}

        for pattern in requires_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj = match.group(1).lower()
                obj = match.group(2).lower()
                if subj in entity_names and obj in entity_names:
                    relationships.append(Relationship(
                        subject=entity_names[subj].name,
                        subject_type=entity_names[subj].entity_type,
                        predicate="REQUIRES",
                        object=entity_names[obj].name,
                        object_type=entity_names[obj].entity_type,
                        confidence=0.8,
                        source=source
                    ))

        for pattern in is_a_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj = match.group(1).lower()
                obj = match.group(2).lower()
                if subj in entity_names and obj in entity_names:
                    relationships.append(Relationship(
                        subject=entity_names[subj].name,
                        subject_type=entity_names[subj].entity_type,
                        predicate="IS_A",
                        object=entity_names[obj].name,
                        object_type=entity_names[obj].entity_type,
                        confidence=0.75,
                        source=source
                    ))

        return relationships

    def _extract_relationships_spacy(
        self,
        text: str,
        entities: list,
        source: str
    ) -> list:
        """Extract relationships using spaCy dependency parsing."""
        relationships = []
        if not self.nlp:
            return relationships

        try:
            doc = self.nlp(text[:5000])
            entity_names = {e.name.lower(): e for e in entities}

            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subj = token.text.lower()
                    verb = token.head.text.lower()
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj", "attr"):
                            obj = child.text.lower()
                            if subj in entity_names and obj in entity_names:
                                predicate = self._map_verb_to_predicate(verb)
                                if predicate:
                                    relationships.append(Relationship(
                                        subject=entity_names[subj].name,
                                        subject_type=entity_names[subj].entity_type,
                                        predicate=predicate,
                                        object=entity_names[obj].name,
                                        object_type=entity_names[obj].entity_type,
                                        confidence=0.7,
                                        source=source
                                    ))
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")

        return relationships

    def _map_verb_to_predicate(self, verb: str) -> Optional[str]:
        """Map a verb to a relationship predicate."""
        verb_map = {
            "require": "REQUIRES",
            "need": "REQUIRES",
            "use": "USED_FOR",
            "support": "SUPPORTS",
            "enable": "ENABLES",
            "teach": "TEACHES",
            "include": "PART_OF",
            "develop": "DEVELOPED_BY",
            "build": "BUILT_ON",
            "apply": "APPLIES_TO",
        }
        return verb_map.get(verb.lower())

    def _deduplicate_entities(self, entities: list) -> list:
        """Remove duplicate entities keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())
