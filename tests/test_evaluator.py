"""Tests for the AI quality evaluator."""

import json
from unittest.mock import MagicMock, patch

from openai import OpenAIError

from benchmark.evaluator import _load_eval_system_prompt, evaluate_responses
from benchmark.models import ModelSummary, OutputMode
from tests.conftest import make_run_metrics


def _mock_completion(response_data: dict) -> MagicMock:
    """Create a mock OpenAI completion object."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = json.dumps(response_data)
    return completion


class TestLoadEvalSystemPrompt:
    def test_json_mode_prompt(self):
        prompt = _load_eval_system_prompt(OutputMode.JSON_SCHEMA)
        assert 'schema_compliance' in prompt
        assert 'accuracy' in prompt

    def test_json_plain_mode_prompt(self):
        prompt = _load_eval_system_prompt(OutputMode.JSON)
        assert 'schema_compliance' in prompt

    def test_text_mode_prompt(self):
        prompt = _load_eval_system_prompt(OutputMode.TEXT)
        assert 'schema_compliance' not in prompt
        assert 'accuracy' in prompt


class TestEvaluateResponses:
    @patch('benchmark.evaluator.OpenAI')
    @patch('benchmark.evaluator.settings')
    def test_json_mode_includes_schema_compliance(self, mock_settings, mock_openai_cls):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.evaluator_max_tokens = 300
        mock_settings.evaluator_temperature = 0.0
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion(
            {
                'schema_compliance': 9,
                'accuracy': 8,
                'completeness': 7,
                'conciseness': 8,
                'reasoning': 'Good.',
            }
        )

        metrics = make_run_metrics(model='model-a', test_case_id='test_1')
        summary = ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            schema_pass_rate=1.0,
            avg_quality_score=0.0,
            estimated_cost_per_1k=0.01,
            run_metrics=[metrics],
        )

        result = evaluate_responses(
            [summary],
            api_key='test-key',
            evaluator_model='eval/model',
            output_mode=OutputMode.JSON_SCHEMA,
        )

        assert len(result[0].quality_scores) == 1
        qs = result[0].quality_scores[0]
        assert qs.schema_compliance == 9
        assert qs.accuracy == 8

    @patch('benchmark.evaluator.OpenAI')
    @patch('benchmark.evaluator.settings')
    def test_text_mode_excludes_schema_compliance(self, mock_settings, mock_openai_cls):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.evaluator_max_tokens = 300
        mock_settings.evaluator_temperature = 0.0
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion(
            {
                'accuracy': 8,
                'completeness': 7,
                'conciseness': 9,
                'reasoning': 'Good text.',
            }
        )

        metrics = make_run_metrics(
            model='model-a',
            test_case_id='test_1',
            raw_response='A summary.',
        )
        summary = ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=80.0,
            avg_completion_tokens=40.0,
            avg_total_tokens=120.0,
            schema_pass_rate=1.0,
            avg_quality_score=0.0,
            estimated_cost_per_1k=0.01,
            run_metrics=[metrics],
        )

        result = evaluate_responses(
            [summary],
            api_key='test-key',
            evaluator_model='eval/model',
            output_mode=OutputMode.TEXT,
        )

        qs = result[0].quality_scores[0]
        assert qs.schema_compliance is None
        assert qs.accuracy == 8

    @patch('benchmark.evaluator.OpenAI')
    @patch('benchmark.evaluator.settings')
    def test_skips_invalid_responses(self, mock_settings, mock_openai_cls):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.evaluator_max_tokens = 300
        mock_settings.evaluator_temperature = 0.0
        client = MagicMock()
        mock_openai_cls.return_value = client

        metrics = make_run_metrics(
            model='model-a',
            test_case_id='test_1',
            schema_valid=False,
        )
        summary = ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            schema_pass_rate=0.0,
            avg_quality_score=0.0,
            estimated_cost_per_1k=0.01,
            run_metrics=[metrics],
        )

        result = evaluate_responses(
            [summary],
            api_key='test-key',
            evaluator_model='eval/model',
            output_mode=OutputMode.JSON_SCHEMA,
        )

        qs = result[0].quality_scores[0]
        assert qs.accuracy == 0
        assert qs.completeness == 0
        assert qs.schema_compliance == 0
        assert 'Skipped' in qs.reasoning
        client.chat.completions.create.assert_not_called()

    @patch('benchmark.evaluator.OpenAI')
    @patch('benchmark.evaluator.settings')
    def test_handles_api_error(self, mock_settings, mock_openai_cls):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.evaluator_max_tokens = 300
        mock_settings.evaluator_temperature = 0.0
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = OpenAIError('API down')

        metrics = make_run_metrics(model='model-a', test_case_id='test_1')
        summary = ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            schema_pass_rate=1.0,
            avg_quality_score=0.0,
            estimated_cost_per_1k=0.01,
            run_metrics=[metrics],
        )

        result = evaluate_responses(
            [summary],
            api_key='test-key',
            evaluator_model='eval/model',
            output_mode=OutputMode.JSON_SCHEMA,
        )

        qs = result[0].quality_scores[0]
        assert qs.accuracy == 0
        assert 'error' in qs.reasoning.lower()

    @patch('benchmark.evaluator.OpenAI')
    @patch('benchmark.evaluator.settings')
    def test_updates_avg_quality_score(self, mock_settings, mock_openai_cls):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.evaluator_max_tokens = 300
        mock_settings.evaluator_temperature = 0.0
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion(
            {
                'schema_compliance': 10,
                'accuracy': 10,
                'completeness': 10,
                'conciseness': 10,
                'reasoning': 'Perfect.',
            }
        )

        metrics_1 = make_run_metrics(model='model-a', test_case_id='test_1')
        metrics_2 = make_run_metrics(model='model-a', test_case_id='test_2')
        summary = ModelSummary(
            model='model-a',
            avg_latency_ms=100.0,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            schema_pass_rate=1.0,
            avg_quality_score=0.0,
            estimated_cost_per_1k=0.01,
            run_metrics=[metrics_1, metrics_2],
        )

        result = evaluate_responses(
            [summary],
            api_key='test-key',
            evaluator_model='eval/model',
            output_mode=OutputMode.JSON_SCHEMA,
        )

        assert result[0].avg_quality_score == 10.0
