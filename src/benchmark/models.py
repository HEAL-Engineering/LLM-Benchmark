"""Pydantic models for benchmark configuration, results, and metrics."""

import json
import urllib.request
from enum import StrEnum

from pydantic import BaseModel, Field


class OutputMode(StrEnum):
    """How the model's response should be constrained."""

    TEXT = 'text'  # Free text, no constraints
    JSON = 'json'  # Valid JSON, no schema enforcement
    JSON_SCHEMA = 'json_schema'  # Strict JSON with schema validation
    # TOOL_CALL = 'tool_call'   # Future: structured via function/tool calling


class TestCase(BaseModel):
    """A single test case to run against each model."""

    id: str = Field(description='Unique identifier for this test case')
    description: str = Field(description='Human-readable description')
    user_prompt: str = Field(description='The user prompt to send to the model')


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration loaded from JSON."""

    name: str = Field(description='Name of this benchmark')
    system_prompt: str = Field(
        default='',
        description='Inline system prompt (or use system_prompt_file)',
    )
    system_prompt_file: str | None = Field(
        default=None,
        description='Path to a .md/.txt file for the system prompt (relative to config file)',
    )
    output_mode: OutputMode = Field(
        default=OutputMode.JSON_SCHEMA,
        description='Response constraint mode: text, json, or json_schema',
    )
    response_schema: dict | None = Field(
        default=None,
        description='JSON Schema for structured output (required when output_mode is json_schema)',
    )
    test_cases: list[TestCase] = Field(description='Test cases to run')
    models: list[str] = Field(description='OpenRouter model IDs to benchmark')
    evaluator_model: str = Field(
        default='anthropic/claude-sonnet-4-6',
        description='Model used to evaluate response quality',
    )
    max_tokens: int = Field(default=600, description='Max completion tokens per request')
    temperature: float = Field(default=0.0, description='Temperature for generation')
    runs_per_test: int = Field(default=1, description='Number of runs per test case per model')


class RunMetrics(BaseModel):
    """Metrics from a single model run against a single test case."""

    model: str
    test_case_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    schema_valid: bool
    parse_error: str | None = None
    raw_response: str = ''


class QualityScore(BaseModel):
    """AI-evaluated quality scores for a single response."""

    model: str
    test_case_id: str
    schema_compliance: int | None = Field(
        default=None, ge=0, le=10, description='Schema adherence (None for text mode)'
    )
    accuracy: int = Field(ge=0, le=10, description='Are the extracted values correct?')
    completeness: int = Field(ge=0, le=10, description='Did it capture all relevant information?')
    conciseness: int = Field(ge=0, le=10, description='Is the output appropriately concise?')
    reasoning: str = Field(description='Brief explanation of scores')

    @property
    def average(self) -> float:
        scores = [self.accuracy, self.completeness, self.conciseness]
        if self.schema_compliance is not None:
            scores.append(self.schema_compliance)
        return sum(scores) / len(scores)


class ModelSummary(BaseModel):
    """Aggregated summary for a single model across all test cases."""

    model: str
    avg_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    schema_pass_rate: float
    avg_quality_score: float
    estimated_cost_per_1k: float
    quality_scores: list[QualityScore] = Field(default_factory=list)
    run_metrics: list[RunMetrics] = Field(default_factory=list)


_pricing_cache: dict[str, tuple[float, float]] | None = None


def fetch_openrouter_pricing() -> dict[str, tuple[float, float]]:
    """
    Fetch live pricing from OpenRouter's /api/v1/models endpoint.

    Returns a dict mapping model ID to (input_price_per_mtok, output_price_per_mtok).
    Results are cached for the lifetime of the process.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    try:
        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/models',
            headers={'Accept': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        pricing = {}
        for model in data.get('data', []):
            model_id = model.get('id', '')
            model_pricing = model.get('pricing', {})
            input_price = model_pricing.get('prompt')
            output_price = model_pricing.get('completion')
            if input_price is not None and output_price is not None:
                # OpenRouter returns price per token; convert to per million tokens
                pricing[model_id] = (
                    float(input_price) * 1_000_000,
                    float(output_price) * 1_000_000,
                )

        _pricing_cache = pricing
        return pricing
    except Exception:
        _pricing_cache = {}
        return {}


def estimate_cost_per_1k(
    model: str, avg_prompt_tokens: float, avg_completion_tokens: float
) -> float:
    """
    Estimate cost per 1000 requests using live OpenRouter pricing.

    Returns -1.0 if pricing is unavailable for the model.
    """
    pricing = fetch_openrouter_pricing()
    model_price = pricing.get(model)
    if model_price is None:
        return -1.0
    input_price, output_price = model_price
    input_cost = (avg_prompt_tokens / 1_000_000) * input_price * 1000
    output_cost = (avg_completion_tokens / 1_000_000) * output_price * 1000
    return round(input_cost + output_cost, 4)
