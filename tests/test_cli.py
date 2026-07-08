from typer.testing import CliRunner

from lab.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "config" in result.output


def test_config_show_default_profile() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "SmolLM2-135M-Instruct" in result.output


def test_config_show_with_overrides() -> None:
    result = runner.invoke(app, ["config", "show", "profile=colab_a100", "lora.r=8"])
    assert result.exit_code == 0
    assert "Qwen/Qwen3-8B" in result.output
    assert "r: 8" in result.output
