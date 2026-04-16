"""Tests for all 4 report formats."""

import json
from io import StringIO

from rich.console import Console

from benchmark.reporter import (
    print_console_report,
    write_html_report,
    write_json_report,
    write_markdown_report,
)


class TestConsoleReport:
    def test_json_mode_has_schema_column(
        self, sample_config_json_schema, sample_summaries_with_scores
    ):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        # Monkey-patch Console in the reporter to capture output
        import benchmark.reporter as reporter_mod

        original = reporter_mod.Console
        reporter_mod.Console = lambda: console
        try:
            print_console_report(sample_config_json_schema, sample_summaries_with_scores)
        finally:
            reporter_mod.Console = original
        output = buf.getvalue()
        assert 'Schema Pass' in output

    def test_text_mode_no_schema_column(self, sample_config_text, sample_summaries_text_mode):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        import benchmark.reporter as reporter_mod

        original = reporter_mod.Console
        reporter_mod.Console = lambda: console
        try:
            print_console_report(sample_config_text, sample_summaries_text_mode)
        finally:
            reporter_mod.Console = original
        output = buf.getvalue()
        assert 'Schema Pass' not in output


class TestHtmlReport:
    def test_renders_without_error(
        self, sample_config_json_schema, sample_summaries_with_scores, tmp_path
    ):
        output_path = tmp_path / 'report.html'
        write_html_report(sample_config_json_schema, sample_summaries_with_scores, output_path)
        content = output_path.read_text()
        assert '<html' in content
        assert '</html>' in content

    def test_json_mode_has_schema_header(
        self, sample_config_json_schema, sample_summaries_with_scores, tmp_path
    ):
        output_path = tmp_path / 'report.html'
        write_html_report(sample_config_json_schema, sample_summaries_with_scores, output_path)
        content = output_path.read_text()
        assert '<th>Schema</th>' in content

    def test_text_mode_no_schema_header(
        self, sample_config_text, sample_summaries_text_mode, tmp_path
    ):
        output_path = tmp_path / 'report.html'
        write_html_report(sample_config_text, sample_summaries_text_mode, output_path)
        content = output_path.read_text()
        assert '<th>Schema</th>' not in content


class TestJsonReport:
    def test_structure(self, sample_config_json_schema, sample_summaries_with_scores, tmp_path):
        output_path = tmp_path / 'report.json'
        write_json_report(sample_config_json_schema, sample_summaries_with_scores, output_path)
        data = json.loads(output_path.read_text())
        assert 'benchmark_name' in data
        assert 'timestamp' in data
        assert 'config' in data
        assert 'summaries' in data
        s = data['summaries'][0]
        assert 'model' in s
        assert 'avg_latency_ms' in s
        assert 'quality_scores' in s

    def test_json_mode_has_schema_fields(
        self, sample_config_json_schema, sample_summaries_with_scores, tmp_path
    ):
        output_path = tmp_path / 'report.json'
        write_json_report(sample_config_json_schema, sample_summaries_with_scores, output_path)
        data = json.loads(output_path.read_text())
        s = data['summaries'][0]
        assert 'schema_pass_rate' in s
        qs = s['quality_scores'][0]
        assert 'schema_compliance' in qs

    def test_text_mode_no_schema_fields(
        self, sample_config_text, sample_summaries_text_mode, tmp_path
    ):
        output_path = tmp_path / 'report.json'
        write_json_report(sample_config_text, sample_summaries_text_mode, output_path)
        data = json.loads(output_path.read_text())
        s = data['summaries'][0]
        assert 'schema_pass_rate' not in s
        qs = s['quality_scores'][0]
        assert 'schema_compliance' not in qs


class TestMarkdownReport:
    def test_correct_columns_json_mode(
        self, sample_config_json_schema, sample_summaries_with_scores, tmp_path
    ):
        output_path = tmp_path / 'report.md'
        write_markdown_report(sample_config_json_schema, sample_summaries_with_scores, output_path)
        content = output_path.read_text()
        # json_schema mode: Rank | Model | Latency | Tokens | Schema | Quality | Cost/1K = 7 cols
        header_line = [line for line in content.split('\n') if line.startswith('| Rank')][0]
        assert header_line.count('|') == 8  # 7 cols = 8 pipes

    def test_correct_columns_text_mode(
        self, sample_config_text, sample_summaries_text_mode, tmp_path
    ):
        output_path = tmp_path / 'report.md'
        write_markdown_report(sample_config_text, sample_summaries_text_mode, output_path)
        content = output_path.read_text()
        # text mode: Rank | Model | Latency | Tokens | Quality | Cost/1K = 6 cols
        header_line = [line for line in content.split('\n') if line.startswith('| Rank')][0]
        assert header_line.count('|') == 7  # 6 cols = 7 pipes


class TestReportFileCreation:
    def test_all_reports_write_files(
        self, sample_config_json_schema, sample_summaries_with_scores, tmp_path
    ):
        write_json_report(
            sample_config_json_schema,
            sample_summaries_with_scores,
            tmp_path / 'report.json',
        )
        write_markdown_report(
            sample_config_json_schema,
            sample_summaries_with_scores,
            tmp_path / 'report.md',
        )
        write_html_report(
            sample_config_json_schema,
            sample_summaries_with_scores,
            tmp_path / 'report.html',
        )

        assert (tmp_path / 'report.json').exists()
        assert (tmp_path / 'report.md').exists()
        assert (tmp_path / 'report.html').exists()
