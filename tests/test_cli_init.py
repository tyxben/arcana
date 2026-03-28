"""Tests for 'arcana init' CLI command."""

from typer.testing import CliRunner

from arcana.cli.main import app

runner = CliRunner()


class TestInitCommand:
    def test_init_creates_files_in_current_dir(self, tmp_path):
        """Init with default '.' should create all three files."""
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / ".env.example").exists()
        assert (tmp_path / "agent.yaml").exists()

    def test_init_creates_files_in_named_dir(self, tmp_path):
        """Init with a named directory should create it and populate."""
        target = tmp_path / "my-agent"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0
        assert target.is_dir()
        assert (target / "main.py").exists()
        assert (target / ".env.example").exists()
        assert (target / "agent.yaml").exists()

    def test_init_creates_nested_dir(self, tmp_path):
        """Init should create parent directories as needed."""
        target = tmp_path / "a" / "b" / "c"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0
        assert (target / "main.py").exists()

    def test_init_file_contents(self, tmp_path):
        """Generated files should contain expected content."""
        runner.invoke(app, ["init", str(tmp_path)])

        main_py = (tmp_path / "main.py").read_text()
        assert "import arcana" in main_py
        assert "@arcana.tool" in main_py
        assert "arcana.Runtime" in main_py

        env = (tmp_path / ".env.example").read_text()
        assert "DEEPSEEK_API_KEY" in env

        yaml = (tmp_path / "agent.yaml").read_text()
        assert "provider: deepseek" in yaml
        assert "arcana run agent.yaml" in yaml

    def test_init_skips_existing_files(self, tmp_path):
        """Existing files should not be overwritten without --force."""
        (tmp_path / "main.py").write_text("# my custom code")
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        # main.py should be untouched
        assert (tmp_path / "main.py").read_text() == "# my custom code"
        # Other files should still be created
        assert (tmp_path / ".env.example").exists()
        assert (tmp_path / "agent.yaml").exists()
        # Output should mention the skip
        assert "exists" in result.stdout

    def test_init_force_overwrites(self, tmp_path):
        """--force should overwrite existing files."""
        (tmp_path / "main.py").write_text("# old code")
        result = runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert result.exit_code == 0
        content = (tmp_path / "main.py").read_text()
        assert "import arcana" in content
        assert "# old code" not in content

    def test_init_shows_next_steps(self, tmp_path):
        """Output should include next steps instructions."""
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert "Next steps" in result.stdout
        assert "arcana run agent.yaml" in result.stdout

    def test_init_shows_created_files(self, tmp_path):
        """Output should list the created files."""
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert "main.py" in result.stdout
        assert ".env.example" in result.stdout
        assert "agent.yaml" in result.stdout

    def test_init_idempotent_with_force(self, tmp_path):
        """Running init --force twice should produce the same result."""
        runner.invoke(app, ["init", str(tmp_path), "--force"])
        runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / ".env.example").exists()
        assert (tmp_path / "agent.yaml").exists()

    def test_init_all_skipped(self, tmp_path):
        """If all files exist, all should be skipped without --force."""
        for name in ("main.py", ".env.example", "agent.yaml"):
            (tmp_path / name).write_text("existing")
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        # All three should still have original content
        assert (tmp_path / "main.py").read_text() == "existing"
        assert (tmp_path / ".env.example").read_text() == "existing"
        assert (tmp_path / "agent.yaml").read_text() == "existing"
