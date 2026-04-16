"""Tests for CLI arg parsing, config loading, and validation."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestRunSubcommand:
    @patch('benchmark.cli.run_benchmark')
    @patch('benchmark.cli.OpenRouterProvider')
    @patch('benchmark.cli.settings')
    def test_loads_config(self, mock_settings, mock_provider_cls, mock_run, tmp_path):
        mock_settings.openrouter_api_key = 'test-key'
        mock_settings.log_level = 'INFO'
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.validate_api_key = MagicMock()
        mock_run.return_value = []

        config_data = {
            'name': 'CLI Test',
            'system_prompt': 'Hello',
            'output_mode': 'text',
            'test_cases': [{'id': 't1', 'description': 'd', 'user_prompt': 'p'}],
            'models': ['model-a'],
        }
        config_path = tmp_path / 'test_config.json'
        config_path.write_text(json.dumps(config_data))

        with patch('sys.argv', ['benchmark', 'run', '-i', str(config_path), '--skip-eval']):
            from benchmark.cli import main

            main()

        mock_run.assert_called_once()
        call_config = mock_run.call_args[0][0]
        assert call_config.name == 'CLI Test'

    @patch('benchmark.cli.settings')
    def test_system_prompt_file_resolved_relative_to_config(self, mock_settings, tmp_path):
        mock_settings.openrouter_api_key = 'test-key'
        mock_settings.log_level = 'INFO'
        mock_settings.openrouter_base_url = 'https://test.api/v1'
        mock_settings.validate_api_key = MagicMock()

        prompt_file = tmp_path / 'prompt.md'
        prompt_file.write_text('System prompt from file.')

        config_data = {
            'name': 'File Prompt Test',
            'system_prompt_file': 'prompt.md',
            'output_mode': 'text',
            'test_cases': [{'id': 't1', 'description': 'd', 'user_prompt': 'p'}],
            'models': ['model-a'],
        }
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps(config_data))

        with (
            patch('sys.argv', ['benchmark', 'run', '-i', str(config_path), '--skip-eval']),
            patch('benchmark.cli.run_benchmark', return_value=[]) as mock_run,
            patch('benchmark.cli.OpenRouterProvider'),
        ):
            from benchmark.cli import main

            main()

        call_config = mock_run.call_args[0][0]
        assert call_config.system_prompt == 'System prompt from file.'

    @patch('benchmark.cli.settings')
    def test_json_schema_without_schema_exits(self, mock_settings, tmp_path):
        mock_settings.openrouter_api_key = 'test-key'
        mock_settings.log_level = 'INFO'
        mock_settings.validate_api_key = MagicMock()

        config_data = {
            'name': 'No Schema',
            'system_prompt': 'Hello',
            'output_mode': 'json_schema',
            'test_cases': [{'id': 't1', 'description': 'd', 'user_prompt': 'p'}],
            'models': ['model-a'],
        }
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps(config_data))

        with (
            patch('sys.argv', ['benchmark', 'run', '-i', str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            from benchmark.cli import main

            main()

        assert exc_info.value.code == 1

    def test_missing_config_exits(self, tmp_path):
        with (
            patch('sys.argv', ['benchmark', 'run', '-i', str(tmp_path / 'nonexistent.json')]),
            patch('benchmark.cli.settings') as mock_settings,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_settings.log_level = 'INFO'
            mock_settings.validate_api_key = MagicMock()
            from benchmark.cli import main

            main()

        assert exc_info.value.code == 1

    def test_no_subcommand_prints_help(self, capsys):
        with patch('sys.argv', ['benchmark']), patch('benchmark.cli.settings') as mock_settings:
            mock_settings.log_level = 'INFO'
            from benchmark.cli import main

            main()

        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower() or 'benchmark' in captured.out.lower()
