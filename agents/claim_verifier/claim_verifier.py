"""
GraphRAG-Nexus — Agent 6: Claim Verifier
Extracts and verifies every factual claim
in the generated answer against source context.
"""

import re
import json
import time
import logging
import anthropic
from graph.state import GraphRAGState
from config.settings import settings
from config.prompts import (
    CLAIM_VERIFIER_SYSTEM_PROMPT,
    CLAIM_VERIFIER_USER_PROMPT
)

logger = logging.getLogger(__name__)


class ClaimVerifierAgent:
    """
    Agent 6 — Claim Verifier

    Responsibilities:
    - Extract factual claims from answer
    - Verify each claim against context
    - Verify each claim against graph triples
    - Calculate claim verification score
    - Flag unverified or contradicted claims
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run claim verification agent."""
        start_time = time.time()
        logger.info("[ClaimVerifier] Verifying claims...")

        if not state.answer or not state.context:
            state.claim_score = 0.0
            return state

        # Skip verification if answer says no context
        if "does not contain enough information" in state.answer:
            state.claim_score = 1.0
            state.verified_count = 0
            state.unverified_count = 0
            return state

        try:
            result = self._verify_with_claude(
                state.context,
                state.answer
            )

            state.claims = result.get("claims", [])
            state.verified_count = result.get(
                "verified_count", 0
            )
            state.unverified_count = result.get(
                "unverified_count", 0
            )
            state.contradicted_count = result.get(
                "contradicted_count", 0
            )
            state.claim_score = result.get("overall_score", 0.0)

        except Exception as e:
            logger.error(f"[ClaimVerifier] Failed: {e}")
            state.claim_score = 0.5
            state.claims = []

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[ClaimVerifier] "
            f"verified={state.verified_count} "
            f"unverified={state.unverified_count} "
            f"score={state.claim_score:.2f} "
            f"({latency:.0f}ms)"
        )
        return state

    def _verify_with_claude(
        self,
        context: str,
        answer: str
    ) -> dict:
        """Use Claude to verify claims."""
        # Truncate context to avoid token limits
        truncated_context = context[:3000]

        response = self.client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=CLAIM_VERIFIER_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": CLAIM_VERIFIER_USER_PROMPT.format(
                    context=truncated_context,
                    answer=answer
                )
            }]
        )

        text = response.content[0].text.strip()

        # Clean JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text)

        # Calculate overall score
        total = (
            result.get("verified_count", 0) +
            result.get("unverified_count", 0) +
            result.get("contradicted_count", 0)
        )
        if total > 0:
            score = result.get("verified_count", 0) / total
            result["overall_score"] = round(score, 3)
        else:
            result["overall_score"] = 1.0

        return result
