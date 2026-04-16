"""Tests for Pydantic models, OutputMode, QualityScore, and pricing."""

from unittest.mock import patch

from benchmark.models import (
    BenchmarkConfig,
    OutputMode,
    QualityScore,
    TestCase,
    estimate_cost_per_1k,
    fetch_openrouter_pricing,
)


class TestOutputMode:
    def test_enum_values(self):
        assert OutputMode.TEXT == 'text'
        assert OutputMode.JSON == 'json'
        assert OutputMode.JSON_SCHEMA == 'json_schema'

    def test_enum_membership(self):
        assert 'text' in OutputMode.__members__.values()
        assert 'json' in OutputMode.__members__.values()
        assert 'json_schema' in OutputMode.__members__.values()


class TestQualityScore:
    def test_average_with_schema(self):
        score = QualityScore(
            model='m',
            test_case_id='t',
            schema_compliance=9,
            accuracy=8,
            completeness=7,
            conciseness=8,
            reasoning='test',
        )
        assert score.average == (9 + 8 + 7 + 8) / 4

    def test_average_without_schema(self):
        score = QualityScore(
            model='m',
            test_case_id='t',
            schema_compliance=None,
            accuracy=8,
            completeness=7,
            conciseness=9,
            reasoning='test',
        )
        assert score.average == (8 + 7 + 9) / 3

    def test_schema_compliance_none_in_text_mode(self):
        score = QualityScore(
            model='m',
            test_case_id='t',
            accuracy=5,
            completeness=5,
            conciseness=5,
            reasoning='ok',
        )
        assert score.schema_compliance is None


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig(
            name='Test',
            system_prompt='Hello',
            test_cases=[TestCase(id='t1', description='d', user_prompt='p')],
            models=['m1'],
        )
        assert config.output_mode == OutputMode.JSON_SCHEMA
        assert config.response_schema is None
        assert config.max_tokens == 600
        assert config.temperature == 0.0
        assert config.runs_per_test == 1
        assert config.evaluator_model == 'anthropic/claude-sonnet-4-6'

    def test_text_mode_no_schema(self):
        config = BenchmarkConfig(
            name='Text',
            system_prompt='Summarize.',
            output_mode=OutputMode.TEXT,
            test_cases=[TestCase(id='t1', description='d', user_prompt='p')],
            models=['m1'],
        )
        assert config.output_mode == OutputMode.TEXT
        assert config.response_schema is None


class TestPricing:
    def test_estimate_cost_per_1k_missing_model(self):
        with patch('benchmark.models.fetch_openrouter_pricing', return_value={}):
            cost = estimate_cost_per_1k('nonexistent/model', 100.0, 50.0)
            assert cost == -1.0

    def test_estimate_cost_per_1k_with_pricing(self):
        pricing = {'test/model': (1.0, 2.0)}  # $1/M input, $2/M output
        with patch('benchmark.models.fetch_openrouter_pricing', return_value=pricing):
            cost = estimate_cost_per_1k('test/model', 1000.0, 500.0)
            # input: (1000/1M) * 1.0 * 1000 = 1.0
            # output: (500/1M) * 2.0 * 1000 = 1.0
            assert cost == 2.0

    def test_fetch_openrouter_pricing_caches(self):
        import benchmark.models as models_mod

        original_cache = models_mod._pricing_cache
        try:
            models_mod._pricing_cache = {'cached/model': (0.5, 1.0)}
            result = fetch_openrouter_pricing()
            assert result == {'cached/model': (0.5, 1.0)}
        finally:
            models_mod._pricing_cache = original_cache
