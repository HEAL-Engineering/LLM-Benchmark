"""OpenRouter provider using the OpenAI SDK with base_url override."""

import json
import logging
import time

from openai import OpenAI, OpenAIError

from benchmark.config import settings
from benchmark.models import OutputMode, RunMetrics

logger = logging.getLogger(__name__)


class OpenRouterProvider:
    """LLM provider that routes through OpenRouter's unified API."""

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
        )
        logger.info('OpenRouter provider initialized')

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
        """Run a single model call via OpenRouter and return metrics."""
        logger.debug('Running model=%s test_case=%s mode=%s', model, test_case_id, output_mode)
        start = time.monotonic()

        # Build request kwargs based on output mode
        kwargs: dict = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        if output_mode == OutputMode.JSON_SCHEMA and response_schema:
            kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'benchmark_response',
                    'strict': True,
                    'schema': response_schema,
                },
            }
        elif output_mode == OutputMode.JSON:
            kwargs['response_format'] = {'type': 'json_object'}
        # OutputMode.TEXT: no response_format, free text

        try:
            completion = self.client.chat.completions.create(**kwargs)
        except OpenAIError as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error('API error for model=%s test_case=%s: %s', model, test_case_id, e)
            return RunMetrics(
                model=model,
                test_case_id=test_case_id,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                schema_valid=False,
                parse_error=f'API error: {e}',
                raw_response='',
            )

        latency_ms = (time.monotonic() - start) * 1000
        content = completion.choices[0].message.content or ''

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        logger.info(
            'Completed model=%s test_case=%s latency=%.0fms tokens=%d',
            model,
            test_case_id,
            latency_ms,
            total_tokens,
        )

        # Validate JSON parsability for JSON modes; text mode always passes
        schema_valid = True
        parse_error = None
        if output_mode in (OutputMode.JSON, OutputMode.JSON_SCHEMA):
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                schema_valid = False
                parse_error = f'JSON parse error: {e}'
                logger.warning(
                    'Invalid JSON from model=%s test_case=%s: %s', model, test_case_id, e
                )

        return RunMetrics(
            model=model,
            test_case_id=test_case_id,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            schema_valid=schema_valid,
            parse_error=parse_error,
            raw_response=content,
        )
