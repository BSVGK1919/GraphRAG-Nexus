"""
GraphRAG-Nexus — Predefined SPARQL queries
for common knowledge graph operations.
"""

# ── Entity Queries ────────────────────────────────────────────

GET_SKILLS_FOR_ROLE = """
SELECT ?skill ?level
WHERE {{
    ?role rdf:type grn:Role .
    ?role rdfs:label "{role_name}" .
    ?role grn:requires_skill ?skill .
    ?skill rdf:type grn:Skill .
    OPTIONAL {{ ?skill grn:level ?level }}
}}
"""

GET_ROLES_FOR_SKILL = """
SELECT ?role ?seniority
WHERE {{
    ?skill rdf:type grn:Skill .
    ?skill rdfs:label "{skill_name}" .
    ?role grn:requires_skill ?skill .
    ?role rdf:type grn:Role .
    OPTIONAL {{ ?role grn:seniority ?seniority }}
}}
"""

GET_RELATED_SKILLS = """
SELECT ?related_skill
WHERE {{
    ?skill rdf:type grn:Skill .
    ?skill rdfs:label "{skill_name}" .
    {{
        ?skill grn:related_to ?related_skill .
    }} UNION {{
        ?skill grn:part_of ?domain .
        ?other_skill grn:part_of ?domain .
        FILTER(?other_skill != ?skill)
        BIND(?other_skill AS ?related_skill)
    }}
}}
LIMIT 10
"""

GET_TOOLS_FOR_DOMAIN = """
SELECT ?tool ?category
WHERE {{
    ?domain rdf:type grn:Domain .
    ?domain rdfs:label "{domain_name}" .
    ?tool grn:used_in ?domain .
    ?tool rdf:type grn:Tool .
    OPTIONAL {{ ?tool grn:category ?category }}
}}
"""

GET_JOBS_FOR_SKILL = """
SELECT ?job ?company ?location ?salary_min ?salary_max
WHERE {{
    ?skill rdf:type grn:Skill .
    ?skill rdfs:label "{skill_name}" .
    ?job grn:requires ?skill .
    ?job rdf:type grn:Job .
    OPTIONAL {{ ?job grn:posted_by ?company }}
    OPTIONAL {{ ?job grn:located_in ?location }}
    OPTIONAL {{ ?job grn:has_salary_min ?salary_min }}
    OPTIONAL {{ ?job grn:has_salary_max ?salary_max }}
}}
LIMIT 10
"""

# ── Multi-hop Queries ─────────────────────────────────────────

GET_LEARNING_PATH = """
SELECT ?skill ?prerequisite
WHERE {{
    ?skill rdf:type grn:Skill .
    ?skill rdfs:label "{skill_name}" .
    ?prerequisite grn:prerequisite_of ?skill .
    ?prerequisite rdf:type grn:Skill .
}}
"""

GET_CAREER_PATH = """
SELECT ?role ?next_role ?required_skills
WHERE {{
    ?role rdf:type grn:Role .
    ?role rdfs:label "{current_role}" .
    ?role grn:enables ?next_role .
    ?next_role rdf:type grn:Role .
    ?next_role grn:requires_skill ?required_skills .
}}
"""

GET_COMPANY_TECH_STACK = """
SELECT ?tool ?framework
WHERE {{
    ?company rdf:type grn:Company .
    ?company rdfs:label "{company_name}" .
    ?job grn:posted_by ?company .
    ?job grn:requires ?skill .
    {{
        ?skill rdf:type grn:Tool .
        BIND(?skill AS ?tool)
    }} UNION {{
        ?skill rdf:type grn:Framework .
        BIND(?skill AS ?framework)
    }}
}}
"""

# ── Neo4j Cypher Queries ──────────────────────────────────────

CYPHER_GET_SKILLS_FOR_ROLE = """
MATCH (r:Role {{name: $role_name}})-[:REQUIRES_SKILL]->(s:Skill)
RETURN s.name AS skill, s.level AS level, s.domain AS domain
ORDER BY s.level
"""

CYPHER_GET_RELATED_TOOLS = """
MATCH (t:Tool {{name: $tool_name}})
OPTIONAL MATCH (t)-[:ALTERNATIVE_TO]-(alt:Tool)
OPTIONAL MATCH (t)-[:BUILT_ON]->(base:Tool)
OPTIONAL MATCH (t)-[:USED_FOR]->(domain:Domain)
RETURN t.name AS tool,
       collect(DISTINCT alt.name) AS alternatives,
       collect(DISTINCT base.name) AS built_on,
       collect(DISTINCT domain.name) AS domains
"""

CYPHER_MULTI_HOP_SKILL_PATH = """
MATCH path = (start:Skill {{name: $start_skill}})
             -[:PREREQUISITE_OF*1..{max_hops}]->
             (end:Skill {{name: $end_skill}})
RETURN [node in nodes(path) | node.name] AS skill_path,
       length(path) AS hops
ORDER BY hops
LIMIT 1
"""

CYPHER_GET_ENTITY_NEIGHBOURHOOD = """
MATCH (e {{name: $entity_name}})
OPTIONAL MATCH (e)-[r1]-(n1)
OPTIONAL MATCH (n1)-[r2]-(n2)
WHERE n2 <> e
RETURN e,
       collect(DISTINCT {{
           node: n1.name,
           label: labels(n1)[0],
           relationship: type(r1)
       }}) AS direct_neighbours,
       collect(DISTINCT {{
           node: n2.name,
           label: labels(n2)[0],
           relationship: type(r2)
       }}) AS second_hop
LIMIT 50
"""

CYPHER_FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes(
    $index_name,
    $query
)
YIELD node, score
RETURN node.name AS name,
       labels(node)[0] AS type,
       score
ORDER BY score DESC
LIMIT $limit
"""
