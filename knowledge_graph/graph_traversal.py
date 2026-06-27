"""
GraphRAG-Nexus — Graph Traversal Engine
Executes Cypher queries against Neo4j and
performs multi-hop graph traversal.
"""

import logging
from neo4j import GraphDatabase
from config.settings import settings
from knowledge_graph.sparql.sparql_generator import SPARQLGenerator

logger = logging.getLogger(__name__)


class GraphTraversalEngine:
    """
    Executes graph queries and traversals against Neo4j.
    Supports multi-hop reasoning up to 3 hops.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.generator = SPARQLGenerator()

    def close(self):
        self.driver.close()

    def query(
        self,
        question: str,
        query_type: str = "skill",
        entities: list[str] = None
    ) -> dict:
        """
        Main entry point — query the knowledge graph.

        Returns structured graph context for the question.
        """
        queries = self.generator.generate(
            question, query_type, entities
        )

        if not queries:
            return self._empty_result(question)

        all_results = []
        for query_dict in queries:
            try:
                results = self._execute_query(
                    query_dict["query"],
                    query_dict["params"]
                )
                if results:
                    all_results.append({
                        "template": query_dict["template"],
                        "entity": query_dict["entity"],
                        "results": results
                    })
            except Exception as e:
                logger.warning(
                    f"Query failed for {query_dict['template']}: {e}"
                )

        graph_context = self._build_context(all_results, question)
        coverage = self._calculate_coverage(
            all_results, queries
        )

        return {
            "question": question,
            "graph_context": graph_context,
            "raw_results": all_results,
            "query_count": len(queries),
            "results_count": len(all_results),
            "graph_coverage": coverage,
            "entities_found": [
                q["entity"] for q in queries
            ]
        }

    def multi_hop_traverse(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = None
    ) -> dict:
        """
        Find path between two entities in the graph.
        """
        max_hops = max_hops or settings.max_graph_hops

        query = f"""
        MATCH path = (start)-[*1..{max_hops}]-(end)
        WHERE toLower(start.name) CONTAINS toLower($start)
          AND toLower(end.name) CONTAINS toLower($end)
        RETURN [node in nodes(path) | node.name] AS path_nodes,
               [rel in relationships(path) | type(rel)] AS relationships,
               length(path) AS hops
        ORDER BY hops
        LIMIT 3
        """

        try:
            results = self._execute_query(query, {
                "start": start_entity,
                "end": end_entity
            })
            return {
                "start": start_entity,
                "end": end_entity,
                "paths": results,
                "found": len(results) > 0
            }
        except Exception as e:
            logger.error(f"Multi-hop traversal failed: {e}")
            return {
                "start": start_entity,
                "end": end_entity,
                "paths": [],
                "found": False
            }

    def get_entity_subgraph(
        self,
        entity_name: str,
        hops: int = 2
    ) -> dict:
        """
        Extract subgraph around an entity.
        """
        query = f"""
        MATCH (e)
        WHERE toLower(e.name) CONTAINS toLower($entity)
        WITH e LIMIT 1
        CALL {{
            WITH e
            MATCH (e)-[r*1..{hops}]-(neighbour)
            RETURN collect(DISTINCT neighbour) AS neighbours,
                   collect(DISTINCT r) AS relationships
        }}
        RETURN e.name AS entity,
               labels(e)[0] AS entity_type,
               [n in neighbours | {{
                   name: n.name,
                   type: labels(n)[0]
               }}] AS neighbours
        """

        try:
            results = self._execute_query(
                query, {"entity": entity_name}
            )
            if results:
                return {
                    "entity": entity_name,
                    "subgraph": results[0],
                    "found": True
                }
            return {
                "entity": entity_name,
                "subgraph": {},
                "found": False
            }
        except Exception as e:
            logger.warning(f"Subgraph extraction failed: {e}")
            return {
                "entity": entity_name,
                "subgraph": {},
                "found": False
            }

    def _execute_query(
        self,
        query: str,
        params: dict
    ) -> list[dict]:
        """Execute a Cypher query and return results."""
        with self.driver.session(
            database=settings.neo4j_database
        ) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def _build_context(
        self,
        results: list[dict],
        question: str
    ) -> str:
        """Build a text context string from graph results."""
        if not results:
            return ""

        context_parts = []

        for result_group in results:
            template = result_group["template"]
            entity = result_group["entity"]
            data = result_group["results"]

            if not data:
                continue

            part = self._format_result(template, entity, data)
            if part:
                context_parts.append(part)

        return "\n\n".join(context_parts)

    def _format_result(
        self,
        template: str,
        entity: str,
        data: list[dict]
    ) -> str:
        """Format a query result as readable text."""
        if not data:
            return ""

        row = data[0]

        if template == "skills_for_role":
            skills = row.get("skills", [])
            if skills:
                return (
                    f"Skills required for {entity}: "
                    f"{', '.join(skills)}"
                )

        elif template == "roles_for_skill":
            roles = row.get("roles", [])
            if roles:
                return (
                    f"Roles that require {entity}: "
                    f"{', '.join(roles)}"
                )

        elif template == "related_skills":
            related = row.get("related_skills", [])
            domain_skills = row.get("domain_skills", [])
            all_related = list(set(related + domain_skills))
            if all_related:
                return (
                    f"Skills related to {entity}: "
                    f"{', '.join(all_related[:10])}"
                )

        elif template == "multi_hop_career_path":
            next_roles = row.get("next_roles", [])
            if next_roles:
                return (
                    f"Career progression from {entity} leads to: "
                    f"{', '.join(next_roles)}"
                )

        elif template == "entity_neighbourhood":
            neighbours = row.get("neighbours", [])
            if neighbours:
                neighbour_text = ", ".join([
                    f"{n['name']} ({n['type']})"
                    for n in neighbours[:5]
                    if n.get("name")
                ])
                if neighbour_text:
                    return (
                        f"Knowledge graph context for {entity}: "
                        f"{neighbour_text}"
                    )

        elif template == "skill_prerequisites":
            prereqs = row.get("prerequisites", [])
            if prereqs:
                return (
                    f"Prerequisites for {entity}: "
                    f"{', '.join(prereqs)}"
                )

        return ""

    def _calculate_coverage(
        self,
        results: list[dict],
        queries: list[dict]
    ) -> float:
        """
        Calculate graph coverage score.
        Higher = more entities found in graph.
        """
        if not queries:
            return 0.0

        entities_queried = set(q["entity"] for q in queries)
        entities_found = set(
            r["entity"] for r in results if r["results"]
        )

        if not entities_queried:
            return 0.0

        coverage = len(entities_found) / len(entities_queried)
        return round(coverage, 3)

    def _empty_result(self, question: str) -> dict:
        """Return empty result structure."""
        return {
            "question": question,
            "graph_context": "",
            "raw_results": [],
            "query_count": 0,
            "results_count": 0,
            "graph_coverage": 0.0,
            "entities_found": []
        }

    def test_connection(self) -> bool:
        """Test Neo4j connection."""
        try:
            with self.driver.session(
                database=settings.neo4j_database
            ) as session:
                result = session.run("RETURN 1 as n").single()
                return result["n"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
