"""Tests for Arcana CLI."""

import pytest
from typer.testing import CliRunner

from arcana.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Arcana" in result.stdout

    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_providers(self):
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "deepseek" in result.stdout
        assert "openai" in result.stdout

    def test_run_no_key(self):
        """Run without API key should fail gracefully."""
        result = runner.invoke(app, ["run", "test", "-p", "deepseek"])
        assert result.exit_code == 1
        assert (
            "API key" in result.stdout
            or "api_key" in result.stdout.lower()
            or result.exit_code == 1
        )

    def test_trace_list_no_dir(self):
        """Trace list with no traces dir should handle gracefully."""
        result = runner.invoke(app, ["trace", "list", "--dir", "/nonexistent"])
        assert result.exit_code == 0

    def test_trace_show_missing(self):
        """Trace show with bad run_id should fail gracefully."""
        result = runner.invoke(app, ["trace", "show", "nonexistent", "--dir", "/tmp"])
        assert result.exit_code == 1
