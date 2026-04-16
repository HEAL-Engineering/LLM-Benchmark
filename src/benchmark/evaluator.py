"""AI quality evaluator - uses a strong model (Sonnet) to score responses."""

import json
import logging
from pathlib import Path

from openai import OpenAI, OpenAIError

from benchmark.config import settings
from benchmark.models import ModelSummary, OutputMode, QualityScore, RunMetrics

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / 'prompts'


def _load_eval_system_prompt(output_mode: OutputMode) -> str:
    """Load the appropriate evaluator system prompt based on output mode."""
    if output_mode == OutputMode.TEXT:
        filename = 'evaluator_system_text.md'
    else:
        filename = 'evaluator_system_json.md'
    prompt_path = PROMPTS_DIR / filename
    return prompt_path.read_text().strip()


def evaluate_responses(
    summaries: list[ModelSummary],
    api_key: str,
    evaluator_model: str,
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
) -> list[ModelSummary]:
    """
    Use an AI evaluator to score each model's responses for quality.

    Mutates the summaries in-place, adding quality scores and updating avg_quality_score.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
    )

    system_prompt = _load_eval_system_prompt(output_mode)
    score_schema = output_mode != OutputMode.TEXT

    total_evals = sum(len([m for m in s.run_metrics if m.schema_valid]) for s in summaries)
    logger.info(
        'Evaluating %d responses with %s (mode=%s)', total_evals, evaluator_model, output_mode
    )

    eval_count = 0
    for summary in summaries:
        scores: list[QualityScore] = []

        for metrics in summary.run_metrics:
            if not metrics.schema_valid:
                scores.append(
                    QualityScore(
                        model=metrics.model,
                        test_case_id=metrics.test_case_id,
                        schema_compliance=0 if score_schema else None,
                        accuracy=0,
                        completeness=0,
                        conciseness=0,
                        reasoning='Skipped: invalid response',
                    )
                )
                continue

            eval_count += 1
            logger.debug(
                'Evaluating %d/%d: model=%s test=%s',
                eval_count,
                total_evals,
                metrics.model,
                metrics.test_case_id,
            )

            score = _evaluate_single(
                client=client,
                evaluator_model=evaluator_model,
                system_prompt=system_prompt,
                metrics=metrics,
                score_schema=score_schema,
            )
            scores.append(score)

        summary.quality_scores = scores
        if scores:
            summary.avg_quality_score = round(sum(s.average for s in scores) / len(scores), 2)

    logger.info('Quality evaluation complete')
    return summaries


def _evaluate_single(
    client: OpenAI,
    evaluator_model: str,
    system_prompt: str,
    metrics: RunMetrics,
    score_schema: bool,
) -> QualityScore:
    """Evaluate a single model response using the AI evaluator."""
    response_block = f'```\n{metrics.raw_response}\n```'
    user_prompt = (
        f'## Original Prompt\n{metrics.test_case_id}\n\n'
        f'## Model Response\n{response_block}\n\n'
        f'Score this response.'
    )

    try:
        completion = client.chat.completions.create(
            model=evaluator_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=settings.evaluator_max_tokens,
            temperature=settings.evaluator_temperature,
        )

        content = completion.choices[0].message.content or '{}'
        data = json.loads(content)

        return QualityScore(
            model=metrics.model,
            test_case_id=metrics.test_case_id,
            schema_compliance=_clamp(data['schema_compliance']) if score_schema else None,
            accuracy=_clamp(data.get('accuracy', 0)),
            completeness=_clamp(data.get('completeness', 0)),
            conciseness=_clamp(data.get('conciseness', 0)),
            reasoning=str(data.get('reasoning', 'No reasoning provided'))[:200],
        )

    except (OpenAIError, json.JSONDecodeError, KeyError) as e:
        logger.error(
            'Evaluator error for model=%s test=%s: %s', metrics.model, metrics.test_case_id, e
        )
        return QualityScore(
            model=metrics.model,
            test_case_id=metrics.test_case_id,
            schema_compliance=0 if score_schema else None,
            accuracy=0,
            completeness=0,
            conciseness=0,
            reasoning=f'Evaluator error: {e}',
        )


def _clamp(value: int) -> int:
    """Clamp a score to 0-10 range."""
    return max(0, min(10, int(value)))
