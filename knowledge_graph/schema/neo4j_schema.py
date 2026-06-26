"""
GraphRAG-Nexus — Neo4j Schema Setup
Creates constraints and indexes in Neo4j.
"""

import logging
from neo4j import GraphDatabase
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jSchema:
    """Sets up Neo4j schema — constraints and indexes."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def setup(self):
        """Run full schema setup."""
        logger.info("Setting up Neo4j schema...")
        self._create_constraints()
        self._create_indexes()
        self._verify()
        logger.info("Neo4j schema setup complete")

    def _create_constraints(self):
        """Create uniqueness constraints for all node types."""
        constraints = [
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT tool_name IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT model_name IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
        ]

        with self.driver.session(database=settings.neo4j_database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Constraint created: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

    def _create_indexes(self):
        """Create full-text and property indexes."""
        indexes = [
            # Full-text search indexes
            "CREATE FULLTEXT INDEX skill_search IF NOT EXISTS FOR (s:Skill) ON EACH [s.name, s.description]",
            "CREATE FULLTEXT INDEX role_search IF NOT EXISTS FOR (r:Role) ON EACH [r.name, r.description]",
            "CREATE FULLTEXT INDEX tool_search IF NOT EXISTS FOR (t:Tool) ON EACH [t.name]",
            "CREATE FULLTEXT INDEX domain_search IF NOT EXISTS FOR (d:Domain) ON EACH [d.name, d.description]",
            "CREATE FULLTEXT INDEX concept_search IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description]",

            # Property indexes
            "CREATE INDEX skill_domain IF NOT EXISTS FOR (s:Skill) ON (s.domain)",
            "CREATE INDEX role_seniority IF NOT EXISTS FOR (r:Role) ON (r.seniority)",
            "CREATE INDEX job_location IF NOT EXISTS FOR (j:Job) ON (j.location)",
            "CREATE INDEX tool_category IF NOT EXISTS FOR (t:Tool) ON (t.category)",
        ]

        with self.driver.session(database=settings.neo4j_database) as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Index created: {index[:50]}...")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")

    def _verify(self):
        """Verify schema was created correctly."""
        with self.driver.session(database=settings.neo4j_database) as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            logger.info(f"Total constraints: {len(constraints)}")

            result = session.run("SHOW INDEXES")
            indexes = list(result)
            logger.info(f"Total indexes: {len(indexes)}")

    def test_connection(self):
        """Test Neo4j connection."""
        try:
            with self.driver.session(
                database=settings.neo4j_database
            ) as session:
                result = session.run("RETURN 1 as n")
                record = result.single()
                if record["n"] == 1:
                    logger.info("Neo4j connection successful")
                    return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False


def setup_neo4j_schema():
    """Main function to setup Neo4j schema."""
    schema = Neo4jSchema()
    try:
        if schema.test_connection():
            schema.setup()
        else:
            raise ConnectionError("Cannot connect to Neo4j")
    finally:
        schema.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_neo4j_schema()
