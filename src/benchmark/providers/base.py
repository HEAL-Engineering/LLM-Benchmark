"""Base protocol for LLM providers."""

from typing import Protocol

from benchmark.models import OutputMode, RunMetrics


class LLMProvider(Protocol):
    """Protocol that all LLM providers must satisfy."""

    def run(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_mode: OutputMode,
        max_tokens: int,
        temperature: float,
        test_case_id: str,
        response_schema: dict | None = None,
    ) -> RunMetrics:
        """
        Run a single model call and return metrics.

        Args:
            model: Model identifier (e.g., 'openai/gpt-4.1-nano')
            system_prompt: System prompt content
            user_prompt: User prompt content
            output_mode: How to constrain the response (text, json, json_schema)
            max_tokens: Max completion tokens
            temperature: Sampling temperature
            test_case_id: ID of the test case being run
            response_schema: JSON Schema dict (required when output_mode is json_schema)

        Returns:
            RunMetrics with timing, tokens, and response data
        """
        ...
