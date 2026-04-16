"""Shared fixtures for benchmark tests."""

import pytest

from benchmark.models import (
    BenchmarkConfig,
    ModelSummary,
    OutputMode,
    QualityScore,
    RunMetrics,
    TestCase,
)

SAMPLE_SCHEMA = {
    'type': 'object',
    'required': ['summary', 'topics'],
    'additionalProperties': False,
    'properties': {
        'summary': {'type': 'string'},
        'topics': {'type': 'array', 'items': {'type': 'string'}},
    },
}

SAMPLE_TEST_CASES = [
    TestCase(id='test_1', description='First test', user_prompt='Extract data from: Hello world'),
    TestCase(
        id='test_2',
        description='Second test',
        user_prompt='Extract data from: Goodbye world',
    ),
]


@pytest.fixture
def sample_config_json_schema() -> BenchmarkConfig:
    return BenchmarkConfig(
        name='Test Benchmark',
        system_prompt='You are a helpful assistant.',
        output_mode=OutputMode.JSON_SCHEMA,
        response_schema=SAMPLE_SCHEMA,
        test_cases=SAMPLE_TEST_CASES,
        models=['model-a', 'model-b'],
        evaluator_model='evaluator/model',
        max_tokens=100,
        temperature=0.0,
        runs_per_test=1,
    )


@pytest.fixture
def sample_config_text() -> BenchmarkConfig:
    return BenchmarkConfig(
        name='Text Benchmark',
        system_prompt='Summarize the following.',
        output_mode=OutputMode.TEXT,
        test_cases=SAMPLE_TEST_CASES,
        models=['model-a', 'model-b'],
        evaluator_model='evaluator/model',
    )


@pytest.fixture
def sample_config_json() -> BenchmarkConfig:
    return BenchmarkConfig(
        name='JSON Benchmark',
        system_prompt='Return valid JSON.',
        output_mode=OutputMode.JSON,
        test_cases=SAMPLE_TEST_CASES,
        models=['model-a'],
        evaluator_model='evaluator/model',
    )


def make_run_metrics(
    model: str = 'model-a',
    test_case_id: str = 'test_1',
    latency_ms: float = 150.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    schema_valid: bool = True,
    raw_response: str = '{"summary": "hello", "topics": ["greeting"]}',
) -> RunMetrics:
    return RunMetrics(
        model=model,
        test_case_id=test_case_id,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        schema_valid=schema_valid,
        raw_response=raw_response,
    )


def make_quality_score(
    model: str = 'model-a',
    test_case_id: str = 'test_1',
    schema_compliance: int | None = 9,
    accuracy: int = 8,
    completeness: int = 7,
    conciseness: int = 8,
    reasoning: str = 'Good response overall.',
) -> QualityScore:
    return QualityScore(
        model=model,
        test_case_id=test_case_id,
        schema_compliance=schema_compliance,
        accuracy=accuracy,
        completeness=completeness,
        conciseness=conciseness,
        reasoning=reasoning,
    )


@pytest.fixture
def sample_run_metrics() -> list[RunMetrics]:
    return [
        make_run_metrics(model='model-a', test_case_id='test_1', latency_ms=120.0),
        make_run_metrics(model='model-a', test_case_id='test_2', latency_ms=180.0),
    ]


@pytest.fixture
def sample_summaries_with_scores() -> list[ModelSummary]:
    scores_a = [
        make_quality_score(model='model-a', test_case_id='test_1'),
        make_quality_score(model='model-a', test_case_id='test_2', accuracy=9),
    ]
    scores_b = [
        make_quality_score(model='model-b', test_case_id='test_1', accuracy=6, completeness=5),
        make_quality_score(model='model-b', test_case_id='test_2', accuracy=7, completeness=6),
    ]
    return [
        ModelSummary(
            model='model-a',
            avg_latency_ms=150.0,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            schema_pass_rate=1.0,
            avg_quality_score=8.1,
            estimated_cost_per_1k=0.0150,
            quality_scores=scores_a,
            run_metrics=[
                make_run_metrics(model='model-a', test_case_id='test_1'),
                make_run_metrics(model='model-a', test_case_id='test_2'),
            ],
        ),
        ModelSummary(
            model='model-b',
            avg_latency_ms=200.0,
            avg_prompt_tokens=110.0,
            avg_completion_tokens=55.0,
            avg_total_tokens=165.0,
            schema_pass_rate=0.5,
            avg_quality_score=6.5,
            estimated_cost_per_1k=0.0200,
            quality_scores=scores_b,
            run_metrics=[
                make_run_metrics(model='model-b', test_case_id='test_1'),
                make_run_metrics(model='model-b', test_case_id='test_2', schema_valid=False),
            ],
        ),
    ]


@pytest.fixture
def sample_summaries_text_mode() -> list[ModelSummary]:
    scores = [
        make_quality_score(
            model='model-a',
            test_case_id='test_1',
            schema_compliance=None,
        ),
        make_quality_score(
            model='model-a',
            test_case_id='test_2',
            schema_compliance=None,
            accuracy=9,
        ),
    ]
    return [
        ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=80.0,
            avg_completion_tokens=40.0,
            avg_total_tokens=120.0,
            schema_pass_rate=1.0,
            avg_quality_score=7.8,
            estimated_cost_per_1k=0.0100,
            quality_scores=scores,
            run_metrics=[
                make_run_metrics(model='model-a', test_case_id='test_1', raw_response='A summary.'),
                make_run_metrics(model='model-a', test_case_id='test_2', raw_response='Another.'),
            ],
        ),
    ]
