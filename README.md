# llm-benchmark

Generic LLM benchmarking tool powered by [OpenRouter](https://openrouter.ai/). Compare any number of models against your own data and get a side-by-side comparison of speed, cost, quality, and schema compliance.

Works for any evaluation task -- free text summarization, JSON extraction, strict schema-validated structured output, or anything else you can throw at an LLM.

## Quick Start

```bash
# 1. One-command setup (interactive: picks Docker or uv, installs deps)
task setup

# 2. Set your OpenRouter API key
#    Edit .env and add your key (get one at https://openrouter.ai/keys)

# 3. Generate a config (interactive wizard)
task init

# 4. Run the benchmark
task run -- -i your_config.json
```

Or without Taskfile:

```bash
uv sync
cp .env.example .env
# Edit .env → add OPENROUTER_API_KEY
uv run benchmark init
uv run benchmark run -i your_config.json
```

## How It Works

1. **Define** your test in a JSON config: system prompt, test cases, models to compare, and optionally a response schema
2. **Run** -- the tool sends each test case to every model via OpenRouter
3. **Measure** -- latency, token usage, schema compliance, and estimated cost per 1K requests (live pricing from OpenRouter)
4. **Evaluate** -- an AI evaluator (default: Claude Sonnet) scores each response for accuracy, completeness, and conciseness
5. **Report** -- results as a Rich terminal table, JSON file, Markdown file, and a self-contained HTML report

## Output Modes

The tool supports three output modes, configured per benchmark:

| Mode | Description | Schema Column | Evaluator Dimensions |
|------|-------------|---------------|---------------------|
| `text` | Free text, no constraints | Hidden | accuracy, completeness, conciseness |
| `json` | Valid JSON, no schema enforcement | Shown (JSON parsability) | + schema_compliance |
| `json_schema` | Strict JSON matching a schema | Shown (schema validation) | + schema_compliance |

## Config Format

```json
{
  "name": "My Benchmark",
  "output_mode": "json_schema",
  "system_prompt_file": "system_prompt.md",
  "response_schema": { ... },
  "test_cases": [
    {
      "id": "test_1",
      "description": "What this tests",
      "user_prompt": "The input to send to the model"
    }
  ],
  "models": [
    "openai/gpt-4.1-nano",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-haiku-4-5"
  ],
  "evaluator_model": "anthropic/claude-sonnet-4-6",
  "max_tokens": 600,
  "temperature": 0.0,
  "runs_per_test": 1
}
```

**Config fields:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | -- | Benchmark name (shown in reports) |
| `output_mode` | No | `json_schema` | `text`, `json`, or `json_schema` |
| `system_prompt` | One of these | -- | Inline system prompt |
| `system_prompt_file` | One of these | -- | Path to prompt file (relative to config) |
| `response_schema` | If json_schema | -- | JSON Schema for structured output |
| `test_cases` | Yes | -- | Array of `{id, description, user_prompt}` |
| `models` | Yes | -- | OpenRouter model IDs to compare |
| `evaluator_model` | No | `anthropic/claude-sonnet-4-6` | Model that scores quality |
| `max_tokens` | No | `600` | Max completion tokens |
| `temperature` | No | `0.0` | Generation temperature |
| `runs_per_test` | No | `1` | Runs per test case per model |

## CLI Usage

```bash
# Run a benchmark with all output formats
task run -- -i config.json

# Specify output directory and formats
task run -- -i config.json -o results/ --formats console json md html

# Skip AI quality evaluation (just collect metrics)
task run -- -i config.json --skip-eval

# Generate a new config interactively
task init
```

Or directly with `uv run benchmark run -i config.json` / `uv run benchmark init`.

## Task Commands

Requires [go-task](https://taskfile.dev/) (`brew install go-task`). The Taskfile supports Docker and uv modes via a `.mode` file.

| Task | Description |
|------|-------------|
| `task setup` | Interactive first-time setup |
| `task run -- -i config.json` | Run benchmark (mode-aware) |
| `task init` | Config wizard (mode-aware) |
| `task lint` | Ruff check + format check |
| `task lint:fix` | Auto-fix lint and formatting |
| `task test` | Run pytest test suite |
| `task deps` | Install/sync dependencies |
| `task build` | Build Docker image |
| `task clean` | Remove results/, caches |
| `task clean:all` | Full cleanup (.venv, Docker images) |
| `task mode` | Print current mode |
| `task mode:docker` / `task mode:uv` | Switch runtime mode |

## Reports

Every run can generate up to 4 report formats:

- **Console** -- Rich-formatted tables in the terminal with color-coded scores
- **JSON** -- Machine-readable results with all metrics and quality scores
- **Markdown** -- GitHub-friendly tables for pasting into issues/PRs
- **HTML** -- Self-contained dark-themed report (double-click to open in browser)

The HTML report includes a "Recommended Model" card, comparison table, and per-model quality breakdowns with color-coded scores.

## Example: Email Extraction

An included example benchmarks 7 models on email insight extraction:

```bash
uv run benchmark run -i data/examples/email_extraction/config.json
```

Models tested: GPT-4.1 Nano, GPT-4.1 Mini, Gemini 2.5 Flash Lite, Gemini 2.5 Flash, Gemini 3.1 Flash Lite, Gemini 3 Flash, Claude Haiku 4.5

12 test cases covering: promotional emails, order confirmations, flight bookings, credit card statements, appointment reminders, GitHub notifications, utility bills, investment alerts, and more.

## Docker

```bash
# Set your API key in .env, then:
docker compose run --rm benchmark run -i data/examples/email_extraction/config.json
```

Results are written to `./results/` on the host via volume mount.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Lint
task lint
# or: uv run ruff check . && uv run ruff format --check .

# Auto-fix
task lint:fix

# Run tests (44 tests, all mocked -- no API key needed)
task test
# or: uv run pytest tests/ -v
```

## Project Structure

```
src/benchmark/
  cli.py              - CLI entry point (argparse subcommands)
  runner.py            - Runs models against test cases, collects metrics
  models.py            - Pydantic models (config, metrics, scores, summaries)
  evaluator.py         - AI quality evaluation
  reporter.py          - 4 output formats (console, JSON, Markdown, HTML)
  config.py            - Environment variable settings
  providers/
    base.py            - LLMProvider Protocol
    openrouter.py      - OpenRouter provider (OpenAI SDK with base_url)
  prompts/             - Evaluator system prompts (json vs text modes)
  templates/           - Jinja2 HTML template

data/examples/         - Example benchmark configs and data
tests/                 - 44 integration tests (mocked API, no real calls)
Taskfile.yml           - Task runner (mode-aware docker/uv)
tasks/helpers.yml      - Internal task helpers
```

## License

MIT
