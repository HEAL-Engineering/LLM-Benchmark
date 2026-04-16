"""CLI entry point for the benchmark tool."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from benchmark.config import settings
from benchmark.evaluator import evaluate_responses
from benchmark.models import BenchmarkConfig, OutputMode
from benchmark.providers.openrouter import OpenRouterProvider
from benchmark.reporter import (
    print_console_report,
    write_html_report,
    write_json_report,
    write_markdown_report,
)
from benchmark.runner import run_benchmark

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='benchmark',
        description='Benchmark LLM models via OpenRouter',
    )
    subparsers = parser.add_subparsers(dest='command')

    # ── benchmark run ────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser('run', help='Run a benchmark from a config file')
    run_parser.add_argument(
        '--input',
        '-i',
        required=True,
        type=Path,
        help='Path to benchmark config JSON file',
    )
    run_parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=Path('results'),
        help='Output directory for report files (default: results/)',
    )
    run_parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip the AI quality evaluation step',
    )
    run_parser.add_argument(
        '--formats',
        nargs='+',
        default=['console', 'json', 'md', 'html'],
        choices=['console', 'json', 'md', 'html'],
        help='Output formats (default: all)',
    )

    # ── benchmark init ───────────────────────────────────────────────────────
    subparsers.add_parser('init', help='Interactive wizard to generate a benchmark config')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.command == 'init':
        _run_init_wizard()
    elif args.command == 'run':
        _run_benchmark(args)
    else:
        parser.print_help()


def _run_benchmark(args: argparse.Namespace) -> None:
    """Execute a benchmark run from a config file."""
    # Load config
    if not args.input.exists():
        logger.error('Config file not found: %s', args.input)
        sys.exit(1)

    raw = json.loads(args.input.read_text())
    config = BenchmarkConfig(**raw)

    # Resolve system_prompt_file relative to config JSON location
    if config.system_prompt_file:
        prompt_path = (args.input.parent / config.system_prompt_file).resolve()
        if not prompt_path.exists():
            logger.error('System prompt file not found: %s', prompt_path)
            sys.exit(1)
        config.system_prompt = prompt_path.read_text().strip()

    if not config.system_prompt:
        logger.error('Config must provide either "system_prompt" or "system_prompt_file"')
        sys.exit(1)

    # Validate json_schema mode has a schema
    if config.output_mode == OutputMode.JSON_SCHEMA and not config.response_schema:
        logger.error('output_mode "json_schema" requires a "response_schema" in the config')
        sys.exit(1)

    # Validate API key
    settings.validate_api_key()

    # Run benchmark
    provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    summaries = run_benchmark(config, provider)

    # Evaluate quality
    if not args.skip_eval:
        evaluate_responses(
            summaries=summaries,
            api_key=settings.openrouter_api_key,
            evaluator_model=config.evaluator_model,
            output_mode=config.output_mode,
        )

    # Generate reports
    output_dir: Path = args.output

    if 'console' in args.formats:
        print_console_report(config, summaries)

    if 'json' in args.formats:
        write_json_report(config, summaries, output_dir / 'report.json')

    if 'md' in args.formats:
        write_markdown_report(config, summaries, output_dir / 'report.md')

    if 'html' in args.formats:
        write_html_report(config, summaries, output_dir / 'report.html')

    written = [f for f in args.formats if f != 'console']
    if written:
        logger.info('Reports written to %s/', output_dir)


def _run_init_wizard() -> None:
    """Interactive wizard to generate a benchmark config file."""
    console = Console()
    console.print()
    console.print('[bold blue]LLM Benchmark Config Generator[/bold blue]')
    console.print('[dim]Answer the questions below to create your config file.[/dim]')
    console.print()

    # Benchmark name
    name = Prompt.ask('Benchmark name', default='My LLM Benchmark')

    # Output mode
    console.print()
    console.print('[bold]Output mode[/bold] - how should models respond?')
    console.print('  [cyan]text[/cyan]        - Free text, no constraints')
    console.print('  [cyan]json[/cyan]        - Valid JSON, no schema enforcement')
    console.print('  [cyan]json_schema[/cyan] - Strict JSON matching a schema')
    console.print()
    output_mode = Prompt.ask(
        'Output mode',
        choices=['text', 'json', 'json_schema'],
        default='json_schema',
    )

    # Schema file (only for json_schema)
    schema_file = None
    if output_mode == 'json_schema':
        console.print()
        schema_file = Prompt.ask('Path to JSON schema file (relative to config output location)')

    # System prompt
    console.print()
    console.print('[bold]System prompt[/bold] - instructions sent to every model')
    prompt_choice = Prompt.ask(
        'Provide as [cyan]file[/cyan] path or [cyan]inline[/cyan]?',
        choices=['file', 'inline'],
        default='file',
    )

    system_prompt = ''
    system_prompt_file = None
    if prompt_choice == 'file':
        system_prompt_file = Prompt.ask('Path to system prompt file (relative to config)')
    else:
        system_prompt = Prompt.ask('System prompt text')

    # Models
    console.print()
    console.print('[bold]Models[/bold] - OpenRouter model IDs to benchmark')
    console.print(
        '[dim]Comma-separated, e.g.: openai/gpt-4.1-nano, google/gemini-2.5-flash-lite[/dim]'
    )
    models_raw = Prompt.ask('Models')
    models = [m.strip() for m in models_raw.split(',') if m.strip()]

    # Runs per test
    runs = int(Prompt.ask('Runs per test case', default='1'))

    # Max tokens
    max_tokens = int(Prompt.ask('Max completion tokens', default='600'))

    # Evaluator
    evaluator = Prompt.ask('Evaluator model', default='anthropic/claude-sonnet-4-6')

    # Build config
    config: dict = {
        'name': name,
        'output_mode': output_mode,
    }

    if system_prompt_file:
        config['system_prompt_file'] = system_prompt_file
    elif system_prompt:
        config['system_prompt'] = system_prompt

    if schema_file:
        console.print()
        console.print(f'[dim]Loading schema from {schema_file}...[/dim]')
        schema_path = Path(schema_file)
        if schema_path.exists():
            config['response_schema'] = json.loads(schema_path.read_text())
        else:
            console.print(
                f'[yellow]Warning: {schema_file} not found. Add response_schema manually.[/yellow]'
            )
            config['response_schema'] = {}

    config['test_cases'] = [
        {
            'id': 'example_1',
            'description': 'Replace with your first test case',
            'user_prompt': 'Replace with the prompt to send to the model',
        }
    ]
    config['models'] = models
    config['evaluator_model'] = evaluator
    config['max_tokens'] = max_tokens
    config['temperature'] = 0.0
    config['runs_per_test'] = runs

    # Output location
    console.print()
    output_path = Path(Prompt.ask('Output config file path', default='benchmark_config.json'))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2))

    console.print()
    console.print(f'[green]Config written to {output_path}[/green]')
    console.print()
    console.print('Next steps:')
    console.print(
        f'  1. Add your test cases to the "test_cases" array in [cyan]{output_path}[/cyan]'
    )
    if system_prompt_file and not Path(system_prompt_file).exists():
        console.print(f'  2. Create your system prompt file: [cyan]{system_prompt_file}[/cyan]')
    console.print(
        f'  3. Run: [bold]OPENROUTER_API_KEY=sk-or-... benchmark run -i {output_path}[/bold]'
    )
    console.print()
