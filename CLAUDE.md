# CLAUDE.md

## Project Overview

Generic, open-source LLM benchmarking tool powered by [OpenRouter](https://openrouter.ai/). Designed for anyone to compare models on their own data -- not just JSON extraction, but any evaluation task (free text, structured JSON, strict schema-validated JSON).

The tool runs each model against every test case, collects latency/token/cost metrics, optionally validates schema compliance, then uses an AI evaluator to score response quality. Results are output as Rich console tables, JSON, Markdown, and a self-contained HTML report.

## Build & Run

```bash
# One-time setup (interactive: picks docker or uv, installs deps)
task setup

# Or manually:
uv sync
cp .env.example .env
# Edit .env → add OPENROUTER_API_KEY

# Run a benchmark
task run -- -i data/examples/email_extraction/config.json

# Or directly:
uv run benchmark run -i data/examples/email_extraction/config.json

# Config wizard
task init

# Lint & test
task lint
task test

# Docker
docker compose run --rm benchmark run -i data/examples/email_extraction/config.json
```

## Task Commands (Taskfile.yml)

Requires [go-task](https://taskfile.dev/). All tasks are mode-aware (docker vs uv) via `.mode` file.

| Task | Description |
|------|-------------|
| `task` / `task help` | List all tasks |
| `task setup` | Interactive first-time setup (choose Docker or uv, install deps) |
| `task run -- -i config.json` | Run benchmark (mode-aware, passes args through) |
| `task init` | Interactive config wizard (mode-aware) |
| `task lint` | Ruff check + format check |
| `task lint:fix` | Auto-fix lint and formatting |
| `task test` | Run pytest suite (`tests/ -v`) |
| `task deps` | Install/sync dependencies (mode-aware) |
| `task build` | Build Docker image |
| `task clean` | Remove results/, caches, build artifacts |
| `task clean:all` | Full cleanup (.venv, Docker images, all artifacts) |
| `task mode` | Print current runtime mode |
| `task mode:docker` | Switch to Docker mode |
| `task mode:uv` | Switch to uv mode |

### Runtime Mode Detection

- `.mode` (gitignored) -- user override, contains `docker` or `uv`
- `.mode.default` (checked in) -- defaults to `uv`
- Fallback chain: `.mode` → `.mode.default` → `"uv"`
- `task run`, `task init`, `task deps` use mode to decide `uv run` vs `docker compose run`
- `task lint`, `task test` always use uv (linting/testing is local)

## Architecture

```
src/benchmark/
  __main__.py       - Entry point (python -m benchmark)
  cli.py            - CLI with argparse subcommands: `run` and `init`
  runner.py         - Core orchestration: runs models against test cases
  models.py         - Pydantic models (BenchmarkConfig, RunMetrics, QualityScore, ModelSummary)
  evaluator.py      - AI quality evaluation (evaluator model scores responses)
  reporter.py       - 4 output formats: Rich console, JSON, Markdown, HTML (Jinja2)
  config.py         - BenchmarkSettings via pydantic-settings (env vars)
  providers/
    base.py         - LLMProvider Protocol
    openrouter.py   - OpenRouter via openai SDK (base_url override)
  prompts/
    evaluator_system_json.md   - Evaluator prompt for json/json_schema modes (4 dimensions)
    evaluator_system_text.md   - Evaluator prompt for text mode (3 dimensions, no schema_compliance)
  templates/
    report.html     - Jinja2 HTML template (dark theme, self-contained)

data/examples/
  email_extraction/
    config.json        - 12 test cases, 7 models, json_schema mode
    config_mini.json   - 3 test cases, 2 cheap models (for fast testing)
    system_prompt.md   - Email insight extraction prompt

tests/                       - 44 pytest integration tests (all mocked, no real API calls)
  conftest.py                - Shared fixtures, sample configs, helper factories
  test_models.py             - OutputMode, QualityScore, BenchmarkConfig, pricing
  test_runner.py             - Runner with mock provider, aggregation, sorting
  test_evaluator.py          - Evaluator prompt loading, scoring, error handling
  test_reporter.py           - All 4 report formats, schema column visibility
  test_cli.py                - CLI arg parsing, config loading, validation
  test_openrouter.py         - Provider kwargs per OutputMode, JSON validation

Taskfile.yml                 - Task runner (mode-aware docker/uv)
tasks/helpers.yml            - Internal helpers (mode detection, setup, cleanup)
```

### Key Design Decisions

- **OutputMode enum** (`text` / `json` / `json_schema`): Controls response_format at the provider level, JSON validation, evaluator prompts (3 vs 4 scoring dimensions), and report column visibility. A future `tool_call` mode is stubbed.
- **system_prompt_file**: Config JSON can reference an external .md file for the system prompt (resolved relative to config file directory). Keeps large prompts out of JSON.
- **QualityScore.schema_compliance** is `None` in text mode; the `average` property dynamically computes over 3 or 4 dimensions.
- **Dynamic pricing**: Fetched live from OpenRouter's `/api/v1/models` endpoint, cached per process.
- **Evaluator uses separate prompts**: `evaluator_system_json.md` scores schema_compliance + accuracy + completeness + conciseness. `evaluator_system_text.md` scores only accuracy + completeness + conciseness.
- **HTML reports** use Jinja2 with `PackageLoader` from `benchmark/templates/`.
- **TestCase naming collision**: The Pydantic `TestCase` model name collides with pytest collection. Suppressed via `filterwarnings` in `pyproject.toml`.

### CLI Subcommands

| Command | Description |
|---------|-------------|
| `benchmark run -i config.json` | Run benchmark, generate reports |
| `benchmark run -i config.json -o results/ --formats console json md html` | Full options |
| `benchmark run -i config.json --skip-eval` | Skip AI quality evaluation |
| `benchmark init` | Interactive wizard to generate a config file |

## Code Conventions

- Python 3.13+, UV for dependency management
- Pydantic for all data models
- Rich for terminal output
- OpenAI SDK with `base_url` override for OpenRouter
- Jinja2 for HTML templating
- Type annotations everywhere
- `ruff` for linting and formatting (single quotes, 100 char line length)
- No OOP classes for business logic -- keep it functional
- All tests use mocked APIs (no real API calls in tests)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (get at openrouter.ai/keys) |

Stored in `.env` (gitignored). Template at `.env.example`. Loaded automatically by pydantic-settings.

## Testing

```bash
# Run all 44 tests
task test
# or: uv run pytest tests/ -v

# Lint
task lint
# or: uv run ruff check . && uv run ruff format --check .

# Auto-fix
task lint:fix
```

All tests use mocked OpenRouter/OpenAI API calls -- no real API key needed.
Test fixtures in `tests/conftest.py` provide sample configs for all 3 output modes,
pre-built RunMetrics/ModelSummary/QualityScore objects, and helper factory functions
(`make_run_metrics`, `make_quality_score`).
