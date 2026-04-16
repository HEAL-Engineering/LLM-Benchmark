"""Core benchmark orchestration - runs models against test cases and collects metrics."""

import logging

from benchmark.models import (
    BenchmarkConfig,
    ModelSummary,
    RunMetrics,
    estimate_cost_per_1k,
)
from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def run_benchmark(config: BenchmarkConfig, provider: LLMProvider) -> list[ModelSummary]:
    """
    Run all models against all test cases and return aggregated summaries.

    Args:
        config: Benchmark configuration with models, test cases, prompts
        provider: LLM provider to use for API calls

    Returns:
        List of ModelSummary, one per model, sorted by average latency
    """
    total_runs = len(config.models) * len(config.test_cases) * config.runs_per_test
    logger.info(
        'Starting benchmark "%s": %d models x %d test cases x %d runs = %d total calls',
        config.name,
        len(config.models),
        len(config.test_cases),
        config.runs_per_test,
        total_runs,
    )

    all_metrics: dict[str, list[RunMetrics]] = {model: [] for model in config.models}

    run_count = 0
    for model in config.models:
        logger.info('Benchmarking model: %s', model)
        for test_case in config.test_cases:
            for run_idx in range(config.runs_per_test):
                run_count += 1
                logger.debug(
                    'Run %d/%d: model=%s test=%s run=%d',
                    run_count,
                    total_runs,
                    model,
                    test_case.id,
                    run_idx + 1,
                )

                metrics = provider.run(
                    model=model,
                    system_prompt=config.system_prompt,
                    user_prompt=test_case.user_prompt,
                    output_mode=config.output_mode,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    test_case_id=test_case.id,
                    response_schema=config.response_schema,
                )
                all_metrics[model].append(metrics)

    logger.info('All runs complete. Aggregating results...')
    return _aggregate_summaries(all_metrics)


def _aggregate_summaries(all_metrics: dict[str, list[RunMetrics]]) -> list[ModelSummary]:
    """Aggregate per-run metrics into per-model summaries."""
    summaries = []

    for model, metrics_list in all_metrics.items():
        if not metrics_list:
            continue

        n = len(metrics_list)
        avg_latency = sum(m.latency_ms for m in metrics_list) / n
        avg_prompt = sum(m.prompt_tokens for m in metrics_list) / n
        avg_completion = sum(m.completion_tokens for m in metrics_list) / n
        avg_total = sum(m.total_tokens for m in metrics_list) / n
        schema_pass = sum(1 for m in metrics_list if m.schema_valid) / n

        cost = estimate_cost_per_1k(model, avg_prompt, avg_completion)

        summaries.append(
            ModelSummary(
                model=model,
                avg_latency_ms=round(avg_latency, 1),
                avg_prompt_tokens=round(avg_prompt, 1),
                avg_completion_tokens=round(avg_completion, 1),
                avg_total_tokens=round(avg_total, 1),
                schema_pass_rate=round(schema_pass, 2),
                avg_quality_score=0.0,  # Filled in by evaluator
                estimated_cost_per_1k=cost,
                run_metrics=metrics_list,
            )
        )

    summaries.sort(key=lambda s: s.avg_latency_ms)
    return summaries
