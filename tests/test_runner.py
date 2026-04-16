"""Tests for the benchmark runner."""

from unittest.mock import patch

from benchmark.models import BenchmarkConfig, OutputMode, RunMetrics, TestCase
from benchmark.runner import run_benchmark
from tests.conftest import make_run_metrics


class MockProvider:
    """Mock LLM provider that returns canned RunMetrics."""

    def __init__(self, latency_base: float = 100.0):
        self.calls: list[dict] = []
        self.latency_base = latency_base
        self._call_count = 0

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
        self._call_count += 1
        self.calls.append(
            {
                'model': model,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'output_mode': output_mode,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'test_case_id': test_case_id,
                'response_schema': response_schema,
            }
        )
        return make_run_metrics(
            model=model,
            test_case_id=test_case_id,
            latency_ms=self.latency_base + self._call_count * 10,
            prompt_tokens=100,
            completion_tokens=50,
        )


@patch('benchmark.runner.estimate_cost_per_1k', return_value=0.01)
class TestRunner:
    def test_calls_provider_with_correct_args(self, _mock_cost, sample_config_json_schema):
        provider = MockProvider()
        run_benchmark(sample_config_json_schema, provider)

        assert len(provider.calls) == 4  # 2 models x 2 test cases x 1 run
        first = provider.calls[0]
        assert first['model'] == 'model-a'
        assert first['output_mode'] == OutputMode.JSON_SCHEMA
        assert first['response_schema'] == sample_config_json_schema.response_schema
        assert first['max_tokens'] == 100
        assert first['temperature'] == 0.0

    def test_aggregates_metrics(self, _mock_cost, sample_config_json_schema):
        provider = MockProvider()
        summaries = run_benchmark(sample_config_json_schema, provider)

        assert len(summaries) == 2
        for s in summaries:
            assert s.avg_prompt_tokens == 100.0
            assert s.avg_completion_tokens == 50.0
            assert s.avg_total_tokens == 150.0
            assert s.schema_pass_rate == 1.0

    def test_multiple_runs_per_test(self, _mock_cost):
        config = BenchmarkConfig(
            name='Multi Run',
            system_prompt='Test.',
            output_mode=OutputMode.TEXT,
            test_cases=[TestCase(id='t1', description='d', user_prompt='p')],
            models=['model-a'],
            runs_per_test=3,
        )
        provider = MockProvider()
        summaries = run_benchmark(config, provider)

        assert len(provider.calls) == 3  # 1 model x 1 test x 3 runs
        assert len(summaries) == 1
        assert len(summaries[0].run_metrics) == 3

    def test_sorts_by_latency(self, _mock_cost, sample_config_json_schema):
        provider = MockProvider(latency_base=100.0)
        summaries = run_benchmark(sample_config_json_schema, provider)

        latencies = [s.avg_latency_ms for s in summaries]
        assert latencies == sorted(latencies)
