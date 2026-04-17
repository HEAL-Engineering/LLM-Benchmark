"""Configuration for the benchmark tool. Loads from environment variables via Pydantic."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    """Global settings loaded from environment variables."""

    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    openrouter_models_url: str = 'https://openrouter.ai/api/v1/models'

    default_evaluator_model: str = 'anthropic/claude-sonnet-4-6'
    default_max_tokens: int = 600
    default_temperature: float = 0.0
    default_runs_per_test: int = 1

    evaluator_max_tokens: int = 300
    evaluator_temperature: float = 0.0

    log_level: str = 'INFO'

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='',
        case_sensitive=False,
        extra='ignore',
    )

    def validate_api_key(self) -> None:
        """Raise if the API key is not set."""
        if not self.openrouter_api_key:
            raise ValueError(
                'OPENROUTER_API_KEY environment variable is required. '
                'Get one at https://openrouter.ai/keys'
            )


settings = BenchmarkSettings()
