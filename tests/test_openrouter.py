"""Tests for the OpenRouter provider."""

from unittest.mock import MagicMock, patch

from openai import OpenAIError

from benchmark.models import OutputMode
from benchmark.providers.openrouter import OpenRouterProvider

SAMPLE_SCHEMA = {
    'type': 'object',
    'required': ['summary'],
    'properties': {'summary': {'type': 'string'}},
}


def _mock_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Create a mock completion response."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    completion.usage.total_tokens = prompt_tokens + completion_tokens
    return completion


class TestBuildKwargs:
    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_json_schema_mode(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"summary": "test"}')

        provider = OpenRouterProvider(api_key='test-key')
        provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.JSON_SCHEMA,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
            response_schema=SAMPLE_SCHEMA,
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs['response_format']['type'] == 'json_schema'
        assert call_kwargs['response_format']['json_schema']['strict'] is True
        assert call_kwargs['response_format']['json_schema']['schema'] == SAMPLE_SCHEMA

    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_json_mode(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"key": "val"}')

        provider = OpenRouterProvider(api_key='test-key')
        provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.JSON,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs['response_format'] == {'type': 'json_object'}

    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_text_mode(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('Hello world')

        provider = OpenRouterProvider(api_key='test-key')
        provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.TEXT,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert 'response_format' not in call_kwargs


class TestJsonValidation:
    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_invalid_json_in_json_mode(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('not valid json {{{')

        provider = OpenRouterProvider(api_key='test-key')
        result = provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.JSON,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
        )

        assert result.schema_valid is False
        assert result.parse_error is not None

    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_text_mode_always_valid(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion(
            'This is just free text, not JSON at all!'
        )

        provider = OpenRouterProvider(api_key='test-key')
        result = provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.TEXT,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
        )

        assert result.schema_valid is True
        assert result.parse_error is None

    @patch('benchmark.providers.openrouter.settings')
    @patch('benchmark.providers.openrouter.OpenAI')
    def test_api_error_returns_failed_metrics(self, mock_openai_cls, mock_settings):
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = OpenAIError('Connection failed')

        provider = OpenRouterProvider(api_key='test-key')
        result = provider.run(
            model='test/model',
            system_prompt='Hello',
            user_prompt='Test',
            output_mode=OutputMode.JSON,
            max_tokens=100,
            temperature=0.0,
            test_case_id='t1',
        )

        assert result.schema_valid is False
        assert 'API error' in result.parse_error
        assert result.total_tokens == 0
