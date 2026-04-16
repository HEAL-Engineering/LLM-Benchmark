"""Report generation - Rich console, JSON, Markdown, and HTML outputs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from benchmark.models import BenchmarkConfig, ModelSummary, OutputMode

logger = logging.getLogger(__name__)


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment that loads templates from benchmark/templates/."""
    env = Environment(
        loader=PackageLoader('benchmark', 'templates'),
        autoescape=select_autoescape(['html']),
    )
    env.globals['score_class'] = _html_score_class
    return env


# ── Rich Console Output ─────────────────────────────────────────────────────


def print_console_report(config: BenchmarkConfig, summaries: list[ModelSummary]) -> None:
    """Print a rich formatted report to the terminal."""
    console = Console()
    has_schema = config.output_mode != OutputMode.TEXT
    console.print()

    # Header
    console.print(
        Panel(
            f'[bold]{config.name}[/bold]\n'
            f'{len(config.models)} models  |  {len(config.test_cases)} test cases  |  '
            f'{config.runs_per_test} run(s) each  |  mode: {config.output_mode}',
            title='LLM Benchmark Results',
            border_style='blue',
        )
    )
    console.print()

    # Summary table
    table = Table(title='Model Comparison', show_lines=True, header_style='bold cyan')
    table.add_column('Model', style='bold white', min_width=30)
    table.add_column('Avg Latency', justify='right')
    table.add_column('Avg Tokens', justify='right')
    if has_schema:
        table.add_column('Schema Pass', justify='right')
    table.add_column('Quality', justify='right')
    table.add_column('Cost/1K', justify='right')

    best_quality = max((s.avg_quality_score for s in summaries), default=0)
    best_latency = min((s.avg_latency_ms for s in summaries), default=0)

    for s in sorted(summaries, key=lambda x: x.avg_quality_score, reverse=True):
        latency_style = 'green' if s.avg_latency_ms == best_latency else ''
        quality_style = 'green bold' if s.avg_quality_score == best_quality else ''
        cost_str = f'${s.estimated_cost_per_1k:.4f}' if s.estimated_cost_per_1k >= 0 else 'N/A'

        row = [
            s.model,
            Text(f'{s.avg_latency_ms:.0f}ms', style=latency_style),
            str(int(s.avg_total_tokens)),
        ]
        if has_schema:
            schema_style = 'green' if s.schema_pass_rate == 1.0 else 'red'
            row.append(Text(f'{s.schema_pass_rate:.0%}', style=schema_style))
        row.append(Text(f'{s.avg_quality_score:.1f}/10', style=quality_style))
        row.append(cost_str)
        table.add_row(*row)

    console.print(table)
    console.print()

    # Quality breakdown per model
    for s in sorted(summaries, key=lambda x: x.avg_quality_score, reverse=True):
        if not s.quality_scores:
            continue

        qtable = Table(
            title=f'Quality Breakdown: {s.model}',
            show_lines=True,
            header_style='bold',
        )
        qtable.add_column('Test Case', style='dim')
        if has_schema:
            qtable.add_column('Schema', justify='center')
        qtable.add_column('Accuracy', justify='center')
        qtable.add_column('Complete', justify='center')
        qtable.add_column('Concise', justify='center')
        qtable.add_column('Avg', justify='center', style='bold')
        qtable.add_column('Reasoning', max_width=50)

        for qs in s.quality_scores:
            row = [qs.test_case_id]
            if has_schema:
                row.append(_rich_score(qs.schema_compliance or 0))
            row.extend(
                [
                    _rich_score(qs.accuracy),
                    _rich_score(qs.completeness),
                    _rich_score(qs.conciseness),
                    _rich_score_avg(qs.average),
                    qs.reasoning[:80],
                ]
            )
            qtable.add_row(*row)

        console.print(qtable)
        console.print()

    # Winner announcement
    if summaries:
        winner = max(summaries, key=lambda s: s.avg_quality_score)
        cost_str = (
            f'${winner.estimated_cost_per_1k:.4f}' if winner.estimated_cost_per_1k >= 0 else 'N/A'
        )
        stats = (
            f'Quality: {winner.avg_quality_score:.1f}/10  |  Latency: {winner.avg_latency_ms:.0f}ms'
        )
        if has_schema:
            stats += f'  |  Schema: {winner.schema_pass_rate:.0%}'
        stats += f'  |  Cost/1K: {cost_str}'
        console.print(
            Panel(
                f'[bold green]{winner.model}[/bold green]\n{stats}',
                title='Recommended Model',
                border_style='green',
            )
        )
        console.print()


# ── JSON Output ──────────────────────────────────────────────────────────────


def write_json_report(
    config: BenchmarkConfig, summaries: list[ModelSummary], output_path: Path
) -> None:
    """Write full results as a structured JSON file."""
    has_schema = config.output_mode != OutputMode.TEXT

    def _quality_dict(qs):
        d = {'test_case_id': qs.test_case_id}
        if has_schema:
            d['schema_compliance'] = qs.schema_compliance
        d.update(
            {
                'accuracy': qs.accuracy,
                'completeness': qs.completeness,
                'conciseness': qs.conciseness,
                'average': round(qs.average, 2),
                'reasoning': qs.reasoning,
            }
        )
        return d

    def _summary_dict(s):
        d = {
            'model': s.model,
            'avg_latency_ms': s.avg_latency_ms,
            'avg_prompt_tokens': s.avg_prompt_tokens,
            'avg_completion_tokens': s.avg_completion_tokens,
            'avg_total_tokens': s.avg_total_tokens,
        }
        if has_schema:
            d['schema_pass_rate'] = s.schema_pass_rate
        d.update(
            {
                'avg_quality_score': s.avg_quality_score,
                'estimated_cost_per_1k': s.estimated_cost_per_1k,
                'quality_scores': [_quality_dict(qs) for qs in s.quality_scores],
            }
        )
        return d

    report = {
        'benchmark_name': config.name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {
            'output_mode': config.output_mode,
            'models': config.models,
            'test_cases': len(config.test_cases),
            'runs_per_test': config.runs_per_test,
            'evaluator_model': config.evaluator_model,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
        },
        'summaries': [
            _summary_dict(s)
            for s in sorted(summaries, key=lambda x: x.avg_quality_score, reverse=True)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info('JSON report written to %s', output_path)


# ── Markdown Output ──────────────────────────────────────────────────────────


def write_markdown_report(
    config: BenchmarkConfig, summaries: list[ModelSummary], output_path: Path
) -> None:
    """Write a Markdown summary report."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    ranked = sorted(summaries, key=lambda x: x.avg_quality_score, reverse=True)
    has_schema = config.output_mode != OutputMode.TEXT

    lines = [
        f'# {config.name} - Benchmark Results',
        '',
        f'**Date:** {ts}  ',
        f'**Models:** {len(config.models)}  |  **Test Cases:** {len(config.test_cases)}'
        f'  |  **Runs:** {config.runs_per_test}  |  **Mode:** {config.output_mode}  ',
        f'**Evaluator:** {config.evaluator_model}',
        '',
        '## Summary',
        '',
    ]

    if has_schema:
        lines.append('| Rank | Model | Latency | Tokens | Schema | Quality | Cost/1K |')
        lines.append('|------|-------|---------|--------|--------|---------|---------|')
    else:
        lines.append('| Rank | Model | Latency | Tokens | Quality | Cost/1K |')
        lines.append('|------|-------|---------|--------|---------|---------|')

    for i, s in enumerate(ranked, 1):
        cost = f'${s.estimated_cost_per_1k:.4f}' if s.estimated_cost_per_1k >= 0 else 'N/A'
        if has_schema:
            lines.append(
                f'| {i} | {s.model} | {s.avg_latency_ms:.0f}ms | '
                f'{int(s.avg_total_tokens)} | {s.schema_pass_rate:.0%} | '
                f'{s.avg_quality_score:.1f}/10 | {cost} |'
            )
        else:
            lines.append(
                f'| {i} | {s.model} | {s.avg_latency_ms:.0f}ms | '
                f'{int(s.avg_total_tokens)} | '
                f'{s.avg_quality_score:.1f}/10 | {cost} |'
            )

    lines.append('')

    for s in ranked:
        if not s.quality_scores:
            continue
        lines.append(f'### {s.model}')
        lines.append('')
        if has_schema:
            lines.append('| Test Case | Schema | Accuracy | Complete | Concise | Avg | Reasoning |')
            lines.append('|-----------|--------|----------|----------|---------|-----|-----------|')
        else:
            lines.append('| Test Case | Accuracy | Complete | Concise | Avg | Reasoning |')
            lines.append('|-----------|----------|----------|---------|-----|-----------|')
        for qs in s.quality_scores:
            if has_schema:
                lines.append(
                    f'| {qs.test_case_id} | {qs.schema_compliance} | {qs.accuracy} | '
                    f'{qs.completeness} | {qs.conciseness} | {qs.average:.1f} | '
                    f'{qs.reasoning[:60]} |'
                )
            else:
                lines.append(
                    f'| {qs.test_case_id} | {qs.accuracy} | '
                    f'{qs.completeness} | {qs.conciseness} | {qs.average:.1f} | '
                    f'{qs.reasoning[:60]} |'
                )
        lines.append('')

    if ranked:
        w = ranked[0]
        cost = f'${w.estimated_cost_per_1k:.4f}' if w.estimated_cost_per_1k >= 0 else 'N/A'
        lines.append('## Recommendation')
        lines.append('')
        lines.append(
            f'**{w.model}** scored highest with an average quality of '
            f'**{w.avg_quality_score:.1f}/10** at **{w.avg_latency_ms:.0f}ms** latency '
            f'and **{cost}** per 1K requests.'
        )
        lines.append('')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines))
    logger.info('Markdown report written to %s', output_path)


# ── HTML Output (Jinja2) ────────────────────────────────────────────────────


def write_html_report(
    config: BenchmarkConfig, summaries: list[ModelSummary], output_path: Path
) -> None:
    """Render an HTML report from the Jinja2 template and write to disk."""
    ranked = sorted(summaries, key=lambda x: x.avg_quality_score, reverse=True)

    env = _get_jinja_env()
    template = env.get_template('report.html')

    html = template.render(
        config=config,
        ranked=ranked,
        winner=ranked[0] if ranked else None,
        has_schema=config.output_mode != OutputMode.TEXT,
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info('HTML report written to %s', output_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _rich_score(score: int) -> Text:
    """Format a 0-10 score with color for Rich console."""
    if score >= 8:
        return Text(str(score), style='green')
    if score >= 5:
        return Text(str(score), style='yellow')
    return Text(str(score), style='red')


def _rich_score_avg(score: float) -> Text:
    """Format an average score with color for Rich console."""
    if score >= 8:
        return Text(f'{score:.1f}', style='green bold')
    if score >= 5:
        return Text(f'{score:.1f}', style='yellow bold')
    return Text(f'{score:.1f}', style='red bold')


def _html_score_class(score: int) -> str:
    """Return a CSS class name for a 0-10 score. Used as a Jinja2 global function."""
    if score >= 8:
        return 'score-high'
    if score >= 5:
        return 'score-mid'
    return 'score-low'
