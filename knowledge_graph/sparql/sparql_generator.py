"""
GraphRAG-Nexus — SPARQL Query Generator
Generates SPARQL queries from natural language
using entity extraction and query templates.
"""

import logging
from config.settings import settings
from config.prompts import QUERY_PLANNER_SYSTEM_PROMPT
from config.prompts import QUERY_PLANNER_USER_PROMPT
from ingestion.extractors.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


# ── Cypher Query Templates ────────────────────────────────────

TEMPLATES = {

    "skills_for_role": """
        MATCH (r:Role)
        WHERE toLower(r.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (r)-[:REQUIRES_SKILL]->(s:Skill)
        RETURN r.name AS role,
               collect(DISTINCT s.name) AS skills
        LIMIT 10
    """,

    "roles_for_skill": """
        MATCH (s:Skill)
        WHERE toLower(s.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (r:Role)-[:REQUIRES_SKILL]->(s)
        RETURN s.name AS skill,
               collect(DISTINCT r.name) AS roles
        LIMIT 10
    """,

    "tools_for_domain": """
        MATCH (d:Domain)
        WHERE toLower(d.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (t:Tool)-[:USED_IN]->(d)
        OPTIONAL MATCH (f:Framework)-[:USED_IN]->(d)
        RETURN d.name AS domain,
               collect(DISTINCT t.name) AS tools,
               collect(DISTINCT f.name) AS frameworks
        LIMIT 10
    """,

    "related_skills": """
        MATCH (s:Skill)
        WHERE toLower(s.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (s)-[:RELATED_TO]-(related:Skill)
        OPTIONAL MATCH (s)-[:PART_OF]->(d:Domain)
                       <-[:PART_OF]-(other:Skill)
        RETURN s.name AS skill,
               collect(DISTINCT related.name) AS related_skills,
               collect(DISTINCT other.name) AS domain_skills
        LIMIT 10
    """,

    "skill_prerequisites": """
        MATCH (s:Skill)
        WHERE toLower(s.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (prereq:Skill)-[:PREREQUISITE_OF]->(s)
        RETURN s.name AS skill,
               collect(DISTINCT prereq.name) AS prerequisites
        LIMIT 10
    """,

    "multi_hop_career_path": """
        MATCH (s:Skill)
        WHERE toLower(s.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (r:Role)-[:REQUIRES_SKILL]->(s)
        OPTIONAL MATCH (r)-[:ENABLES]->(next_role:Role)
        OPTIONAL MATCH (next_role)-[:REQUIRES_SKILL]->(ns:Skill)
        RETURN s.name AS skill,
               collect(DISTINCT r.name) AS current_roles,
               collect(DISTINCT next_role.name) AS next_roles,
               collect(DISTINCT ns.name) AS next_role_skills
        LIMIT 5
    """,

    "entity_neighbourhood": """
        MATCH (e)
        WHERE toLower(e.name) CONTAINS toLower($entity)
        OPTIONAL MATCH (e)-[r1]-(n1)
        RETURN e.name AS entity,
               labels(e)[0] AS entity_type,
               collect(DISTINCT {{
                   name: n1.name,
                   type: labels(n1)[0],
                   relationship: type(r1)
               }}) AS neighbours
        LIMIT 1
    """,

    "fulltext_search": """
        CALL db.index.fulltext.queryNodes(
            $index_name, $query
        )
        YIELD node, score
        RETURN node.name AS name,
               labels(node)[0] AS type,
               score
        ORDER BY score DESC
        LIMIT $limit
    """,

    "jobs_for_skills": """
        MATCH (j:Job)
        OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
        WHERE toLower(s.name) CONTAINS toLower($entity)
        RETURN j.title AS title,
               j.company AS company,
               j.location AS location,
               collect(DISTINCT s.name) AS required_skills
        LIMIT 10
    """,
}


class SPARQLGenerator:
    """
    Generates Cypher/SPARQL queries from natural language.
    Uses entity extraction to identify query targets.
    """

    def __init__(self):
        self.extractor = EntityExtractor()

    def generate(
        self,
        question: str,
        query_type: str = "skill",
        entities: list[str] = None
    ) -> list[dict]:
        """
        Generate Cypher queries for a given question.

        Returns list of query dicts with:
        - query: Cypher query string
        - params: Query parameters
        - template: Template name used
        - entity: Target entity
        """
        if not entities:
            extracted = self.extractor.extract(question)
            entities = [
                e.name for e in extracted["entities"]
            ][:3]

        if not entities:
            logger.warning(
                f"No entities found in question: {question}"
            )
            return self._fallback_queries(question)

        queries = []

        for entity in entities:
            entity_queries = self._generate_for_entity(
                entity, query_type, question
            )
            queries.extend(entity_queries)

        logger.info(
            f"Generated {len(queries)} queries for: {question}"
        )
        return queries

    def _generate_for_entity(
        self,
        entity: str,
        query_type: str,
        question: str
    ) -> list[dict]:
        """Generate queries for a specific entity."""
        queries = []
        question_lower = question.lower()

        # Always include neighbourhood query
        queries.append({
            "query": TEMPLATES["entity_neighbourhood"],
            "params": {"entity": entity},
            "template": "entity_neighbourhood",
            "entity": entity
        })

        # Add type-specific queries
        if query_type == "skill":
            queries.append({
                "query": TEMPLATES["skills_for_role"],
                "params": {"entity": entity},
                "template": "skills_for_role",
                "entity": entity
            })
            queries.append({
                "query": TEMPLATES["related_skills"],
                "params": {"entity": entity},
                "template": "related_skills",
                "entity": entity
            })

        elif query_type == "career":
            queries.append({
                "query": TEMPLATES["roles_for_skill"],
                "params": {"entity": entity},
                "template": "roles_for_skill",
                "entity": entity
            })
            queries.append({
                "query": TEMPLATES["multi_hop_career_path"],
                "params": {"entity": entity},
                "template": "multi_hop_career_path",
                "entity": entity
            })

        elif query_type == "project":
            queries.append({
                "query": TEMPLATES["tools_for_domain"],
                "params": {"entity": entity},
                "template": "tools_for_domain",
                "entity": entity
            })

        elif query_type == "research":
            queries.append({
                "query": TEMPLATES["related_skills"],
                "params": {"entity": entity},
                "template": "related_skills",
                "entity": entity
            })

        # Add prerequisite query if learning path mentioned
        if any(word in question_lower for word in [
            "learn", "start", "begin", "prerequisite", "path"
        ]):
            queries.append({
                "query": TEMPLATES["skill_prerequisites"],
                "params": {"entity": entity},
                "template": "skill_prerequisites",
                "entity": entity
            })

        # Add jobs query if job-related
        if any(word in question_lower for word in [
            "job", "role", "hire", "salary", "position"
        ]):
            queries.append({
                "query": TEMPLATES["jobs_for_skills"],
                "params": {"entity": entity},
                "template": "jobs_for_skills",
                "entity": entity
            })

        return queries

    def _fallback_queries(self, question: str) -> list[dict]:
        """Fallback when no entities found."""
        words = [w for w in question.split() if len(w) > 4]
        if not words:
            return []

        return [{
            "query": TEMPLATES["entity_neighbourhood"],
            "params": {"entity": words[0]},
            "template": "entity_neighbourhood",
            "entity": words[0]
        }]
