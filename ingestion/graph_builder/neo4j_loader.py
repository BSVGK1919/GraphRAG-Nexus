"""
GraphRAG-Nexus — Neo4j Graph Loader
Loads extracted entities and relationships
into Neo4j knowledge graph.
"""

import logging
from neo4j import GraphDatabase
from config.settings import settings
from ingestion.extractors.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)


class Neo4jLoader:
    """
    Loads entities and relationships into Neo4j.
    Uses MERGE to avoid duplicates.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.nodes_created = 0
        self.relationships_created = 0

    def close(self):
        self.driver.close()

    def load_entities(self, entities: list[Entity]) -> int:
        """Load entities into Neo4j using MERGE."""
        loaded = 0
        with self.driver.session(
            database=settings.neo4j_database
        ) as session:
            for entity in entities:
                try:
                    session.execute_write(
                        self._merge_entity, entity
                    )
                    loaded += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to load entity {entity.name}: {e}"
                    )
        self.nodes_created += loaded
        logger.info(f"Loaded {loaded} entities into Neo4j")
        return loaded

    def load_relationships(
        self,
        relationships: list[Relationship]
    ) -> int:
        """Load relationships into Neo4j using MERGE."""
        loaded = 0
        with self.driver.session(
            database=settings.neo4j_database
        ) as session:
            for rel in relationships:
                try:
                    session.execute_write(
                        self._merge_relationship, rel
                    )
                    loaded += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to load relationship: {e}"
                    )
        self.relationships_created += loaded
        logger.info(
            f"Loaded {loaded} relationships into Neo4j"
        )
        return loaded

    @staticmethod
    def _merge_entity(tx, entity: Entity):
        """MERGE entity node into Neo4j."""
        query = f"""
        MERGE (n:{entity.entity_type} {{name: $name}})
        SET n.source = $source,
            n.confidence = $confidence
        """
        # Add extra properties
        for key, value in entity.properties.items():
            query += f"\nSET n.{key} = ${key}"

        params = {
            "name": entity.name,
            "source": entity.source,
            "confidence": entity.confidence,
            **entity.properties
        }
        tx.run(query, **params)

    @staticmethod
    def _merge_relationship(tx, rel: Relationship):
        """MERGE relationship between nodes."""
        query = f"""
        MERGE (a:{rel.subject_type} {{name: $subject}})
        MERGE (b:{rel.object_type} {{name: $object}})
        MERGE (a)-[r:{rel.predicate}]->(b)
        SET r.confidence = $confidence,
            r.source = $source
        """
        tx.run(
            query,
            subject=rel.subject,
            object=rel.object,
            confidence=rel.confidence,
            source=rel.source
        )

    def get_stats(self) -> dict:
        """Return Neo4j database statistics."""
        with self.driver.session(
            database=settings.neo4j_database
        ) as session:
            node_count = session.run(
                "MATCH (n) RETURN count(n) as count"
            ).single()["count"]

            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            label_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label,
                       count(n) as count
                ORDER BY count DESC
            """).data()

        return {
            "total_nodes": node_count,
            "total_relationships": rel_count,
            "nodes_created_this_session": self.nodes_created,
            "relationships_created_this_session": (
                self.relationships_created
            ),
            "label_breakdown": label_counts
        }

    def clear_database(self):
        """Clear all nodes and relationships — use carefully."""
        with self.driver.session(
            database=settings.neo4j_database
        ) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Neo4j database cleared")

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
