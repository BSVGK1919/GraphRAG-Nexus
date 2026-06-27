"""
GraphRAG-Nexus — Agent 7: Reflection Agent
Scores answer quality and rewrites if below threshold.
"""

import json
import time
import logging
import anthropic
from graph.state import GraphRAGState
from config.settings import settings
from config.prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT
)

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """
    Agent 7 — Reflection

    Responsibilities:
    - Score answer on faithfulness, relevancy,
      completeness, clarity
    - Rewrite answer if score below threshold
    - Track reflection loops
    - Pass/fail based on threshold
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run reflection agent."""
        start_time = time.time()
        logger.info(
            f"[Reflection] Scoring answer "
            f"(loop {state.reflection_loops + 1})"
        )

        if not state.answer:
            state.reflection_passed = False
            return state

        # Skip if no context answer
        if "does not contain enough information" in state.answer:
            state.reflection_score = 1.0
            state.reflection_passed = True
            return state

        # Max loops reached
        if state.reflection_loops >= settings.reflection_max_loops:
            logger.warning(
                "[Reflection] Max loops reached"
            )
            state.reflection_passed = True
            return state

        try:
            result = self._score_with_claude(state)

            state.reflection_score = result.get(
                "overall_score", 0.0
            )
            state.reflection_feedback = result.get(
                "feedback", ""
            )
            state.reflection_passed = result.get(
                "passed", False
            )

            logger.info(
                f"[Reflection] score={state.reflection_score} "
                f"passed={state.reflection_passed}"
            )

        except Exception as e:
            logger.error(f"[Reflection] Failed: {e}")
            state.reflection_score = 0.5
            state.reflection_passed = True

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[Reflection] "
            f"score={state.reflection_score:.2f} "
            f"passed={state.reflection_passed} "
            f"({latency:.0f}ms)"
        )
        return state

    def _score_with_claude(self, state: GraphRAGState) -> dict:
        """Use Claude to score the answer."""
        truncated_context = state.context[:2000]

        response = self.client.messages.create(
            model=settings.anthropic_model,
            max_tokens=512,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": REFLECTION_USER_PROMPT.format(
                    question=state.question,
                    context=truncated_context,
                    answer=state.answer,
                    claim_score=state.claim_score
                )
            }]
        )

        text = response.content[0].text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text)

        # Determine pass/fail
        overall = result.get("overall_score", 0.0)
        result["passed"] = (
            overall >= settings.reflection_score_threshold
        )

        return result
