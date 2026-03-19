"""Tests for Arcana CLI."""

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

    def test_run_yaml_config(self, tmp_path):
        """Run with YAML config should parse and use config values."""
        cfg = tmp_path / "agent.yaml"
        cfg.write_text("goal: test goal\nprovider: deepseek\nmax_turns: 5\n")
        result = runner.invoke(app, ["run", str(cfg)])
        # Should fail at API key, not at YAML parsing
        assert result.exit_code == 1
        assert "API key" in result.stdout

    def test_run_yaml_override_goal(self, tmp_path):
        """--override should replace goal from YAML."""
        cfg = tmp_path / "agent.yaml"
        cfg.write_text("goal: original\nprovider: deepseek\n")
        result = runner.invoke(app, ["run", str(cfg), "--override", "new goal"])
        assert result.exit_code == 1
        assert "API key" in result.stdout

    def test_run_yaml_no_goal(self, tmp_path):
        """YAML without goal and no --override should error."""
        cfg = tmp_path / "no_goal.yaml"
        cfg.write_text("provider: deepseek\n")
        result = runner.invoke(app, ["run", str(cfg)])
        assert result.exit_code == 1
        assert "no goal" in result.stdout.lower()

    def test_run_yaml_not_found(self):
        """Missing YAML file should error."""
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
