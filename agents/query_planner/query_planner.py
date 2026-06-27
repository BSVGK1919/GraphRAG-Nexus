"""
GraphRAG-Nexus — Agent 1: Query Planner
Classifies query type, extracts entities,
decomposes complex queries into sub-questions.
"""

import json
import time
import logging
import anthropic
from graph.state import GraphRAGState
from config.settings import settings
from config.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    QUERY_PLANNER_USER_PROMPT
)
from ingestion.extractors.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

# Simple query keywords → Ollama
SIMPLE_KEYWORDS = [
    "what is", "what are", "define", "definition",
    "list", "name", "who is", "when was", "how many",
    "which", "example of"
]

# Complex query keywords → Claude
COMPLEX_KEYWORDS = [
    "analyse", "analyze", "compare", "difference",
    "skill gap", "roadmap", "career path", "recommend",
    "best way", "how should", "what should",
    "transition", "senior", "strategy",
]


class QueryPlannerAgent:
    """
    Agent 1 — Query Planner

    Responsibilities:
    - Classify query type (skill/career/project/research)
    - Extract entities from query
    - Determine complexity (simple/complex)
    - Decompose into sub-questions if needed
    - Select retrieval strategies
    """

    def __init__(self):
        self.extractor = EntityExtractor()
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run query planning agent."""
        start_time = time.time()
        logger.info(
            f"[QueryPlanner] Planning: {state.question[:60]}"
        )

        try:
            # Extract entities locally first (fast)
            extracted = self.extractor.extract(state.question)
            local_entities = [
                e.name for e in extracted["entities"]
            ][:5]

            # Determine complexity using rules
            complexity = self._classify_complexity(
                state.question
            )

            # For complex queries use Claude for better planning
            if complexity == "complex" and local_entities:
                plan = self._plan_with_claude(state.question)
            else:
                plan = self._plan_with_rules(
                    state.question,
                    local_entities
                )

            # Update state
            state.query_type = plan.get("query_type", "skill")
            state.entities = plan.get(
                "entities", local_entities
            )[:5]
            state.sub_questions = plan.get(
                "sub_questions", []
            )[:3]
            state.complexity = plan.get("complexity", complexity)
            state.requires_graph = plan.get(
                "requires_graph", True
            )
            state.requires_multi_hop = plan.get(
                "requires_multi_hop", False
            )
            state.retrieval_strategies = plan.get(
                "retrieval_strategies",
                ["vector", "graph", "bm25"]
            )

            # Set LLM provider based on complexity
            state.llm_provider = (
                "ollama" if complexity == "simple"
                else "claude"
            )

        except Exception as e:
            logger.error(f"[QueryPlanner] Failed: {e}")
            state.error = f"Query planning failed: {str(e)}"
            state.entities = []
            state.query_type = "skill"
            state.complexity = "complex"

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[QueryPlanner] type={state.query_type} "
            f"complexity={state.complexity} "
            f"entities={state.entities} "
            f"({latency:.0f}ms)"
        )
        return state

    def _classify_complexity(self, question: str) -> str:
        """Rule-based complexity classification."""
        q = question.lower()
        word_count = len(q.split())

        if word_count > 15:
            return "complex"

        for keyword in COMPLEX_KEYWORDS:
            if keyword in q:
                return "complex"

        for keyword in SIMPLE_KEYWORDS:
            if keyword in q:
                return "simple"

        return "complex"

    def _plan_with_rules(
        self,
        question: str,
        entities: list[str]
    ) -> dict:
        """Rule-based query planning."""
        q = question.lower()

        # Classify query type
        if any(w in q for w in [
            "skill", "learn", "library", "tool",
            "framework", "python", "pytorch"
        ]):
            query_type = "skill"
        elif any(w in q for w in [
            "job", "role", "career", "salary",
            "hire", "position", "engineer"
        ]):
            query_type = "career"
        elif any(w in q for w in [
            "build", "project", "create", "develop",
            "roadmap", "implement"
        ]):
            query_type = "project"
        elif any(w in q for w in [
            "research", "paper", "arxiv", "theory",
            "algorithm", "model"
        ]):
            query_type = "research"
        else:
            query_type = "skill"

        return {
            "query_type": query_type,
            "entities": entities,
            "sub_questions": [],
            "complexity": "simple",
            "requires_graph": True,
            "requires_multi_hop": False,
            "retrieval_strategies": ["vector", "graph", "bm25"]
        }

    def _plan_with_claude(self, question: str) -> dict:
        """Use Claude for complex query planning."""
        try:
            response = self.client.messages.create(
                model=settings.anthropic_model,
                max_tokens=512,
                system=QUERY_PLANNER_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": QUERY_PLANNER_USER_PROMPT.format(
                        query=question
                    )
                }]
            )
            text = response.content[0].text.strip()
            # Clean JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except Exception as e:
            logger.warning(
                f"Claude planning failed, using rules: {e}"
            )
            extracted = self.extractor.extract(question)
            entities = [
                e.name for e in extracted["entities"]
            ][:5]
            return self._plan_with_rules(question, entities)
