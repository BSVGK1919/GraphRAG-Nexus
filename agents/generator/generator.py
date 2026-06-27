"""
GraphRAG-Nexus — Agent 5: Generator
Generates grounded answers using LLM router.
Routes simple queries to Ollama, complex to Claude.
"""

import time
import logging
import requests
import anthropic
from graph.state import GraphRAGState
from config.settings import settings
from config.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_PROMPT
)

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    Agent 5 — Generator

    Responsibilities:
    - Route query to Ollama or Claude
    - Generate grounded answer from context
    - Handle fallback if Ollama fails
    - Store answer in state
    """

    def __init__(self):
        self.claude_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run generator agent."""
        start_time = time.time()
        logger.info(
            f"[Generator] Generating with "
            f"provider={state.llm_provider}"
        )

        if not state.context:
            state.answer = (
                "The available context does not contain "
                "enough information to answer this accurately."
            )
            state.llm_provider = "none"
            return state

        # Add reflection feedback if rewriting
        feedback_note = ""
        if state.reflection_feedback and state.reflection_loops > 0:
            feedback_note = (
                f"\n\nPrevious answer feedback: "
                f"{state.reflection_feedback}\n"
                f"Please improve based on this feedback."
            )

        try:
            if state.llm_provider == "ollama":
                answer = self._call_ollama(
                    state.question,
                    state.context,
                    feedback_note
                )
                if answer.startswith("__OLLAMA_FAILED__"):
                    logger.warning(
                        "[Generator] Ollama failed, "
                        "falling back to Claude"
                    )
                    answer = self._call_claude(
                        state.question,
                        state.context,
                        feedback_note
                    )
                    state.llm_provider = "claude"
                    state.llm_fallback_used = True
            else:
                answer = self._call_claude(
                    state.question,
                    state.context,
                    feedback_note
                )

            state.answer = answer

        except Exception as e:
            logger.error(f"[Generator] Failed: {e}")
            state.answer = (
                f"I encountered an error generating "
                f"the answer: {str(e)}"
            )
            state.error = f"Generator error: {str(e)}"

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[Generator] provider={state.llm_provider} "
            f"answer_len={len(state.answer)} "
            f"({latency:.0f}ms)"
        )
        return state

    def _call_ollama(
        self,
        question: str,
        context: str,
        feedback_note: str = ""
    ) -> str:
        """Call Ollama API."""
        prompt = (
            GENERATOR_USER_PROMPT.format(
                question=question,
                context=context
            ) + feedback_note
        )
        try:
            response = requests.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=settings.ollama_timeout
            )
            response.raise_for_status()
            return response.json().get(
                "response",
                "No response from Ollama."
            )
        except Exception as e:
            return f"__OLLAMA_FAILED__: {str(e)}"

    def _call_claude(
        self,
        question: str,
        context: str,
        feedback_note: str = ""
    ) -> str:
        """Call Claude API."""
        content = (
            GENERATOR_USER_PROMPT.format(
                question=question,
                context=context
            ) + feedback_note
        )
        response = self.claude_client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=GENERATOR_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        return response.content[0].text.strip()
