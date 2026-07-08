import typer
from omegaconf import OmegaConf

from lab.config import LabConfig, load_config
from lab.logging import configure_logging, get_logger

app = typer.Typer(name="lab", no_args_is_help=True, pretty_exceptions_show_locals=False)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config", help="Configuration inspection")

logger = get_logger(__name__)


def setup(overrides: list[str]) -> LabConfig:
    configure_logging()
    config = load_config(overrides)
    logger.info("config_loaded", profile=config.profile_name)
    return config


@config_app.command("show")
def config_show(overrides: list[str] = typer.Argument(None)) -> None:
    config = setup(overrides or [])
    typer.echo(OmegaConf.to_yaml(OmegaConf.structured(config)))


data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Dataset preparation")


@data_app.command("prepare")
def data_prepare(overrides: list[str] = typer.Argument(None)) -> None:
    from lab.data.prepare import prepare_dataset

    config = setup(overrides or [])
    prepare_dataset(config)


@data_app.command("stats")
def data_stats(overrides: list[str] = typer.Argument(None)) -> None:
    from lab.data.stats import report_stats

    config = setup(overrides or [])
    typer.echo(report_stats(config))


train_app = typer.Typer(no_args_is_help=True)
app.add_typer(train_app, name="train", help="SFT and DPO training")


@train_app.command("sft")
def train_sft(overrides: list[str] = typer.Argument(None)) -> None:
    from lab.training.sft import run_sft

    config = setup(overrides or [])
    run_sft(config)


@train_app.command("dpo")
def train_dpo(
    adapter_path: str = typer.Option("", help="SFT adapter path, defaults to SFT output"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.training.dpo import run_dpo

    config = setup(overrides or [])
    run_dpo(config, adapter_path=adapter_path or None)


@train_app.command("pairs")
def train_pairs(
    base_url: str = typer.Option("http://localhost:8000/v1"),
    model: str = typer.Option("base"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.training.pairs import generate_pairs

    config = setup(overrides or [])
    count = generate_pairs(config, base_url=base_url, model=model)
    typer.echo(f"collected {count} preference pairs")


eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval", help="Evaluation suites")


@eval_app.command("run")
def eval_run(
    model_path: str = typer.Option("", help="Model or adapter path, defaults to SFT output"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.eval.runner import run_eval

    config = setup(overrides or [])
    report = run_eval(config, model_path=model_path or None)
    typer.echo(
        f"{report.model_label}: overall={report.overall_accuracy:.3f} "
        f"exec={report.execution_accuracy:.3f} valid={report.valid_sql_rate:.3f}"
    )


@eval_app.command("perplexity")
def eval_perplexity(
    model_path: str = typer.Option(..., help="Model path"),
    split: str = typer.Option("val"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.eval.perplexity import compute_perplexity

    config = setup(overrides or [])
    value = compute_perplexity(config, model_path, split=split)
    typer.echo(f"perplexity({split}) = {value:.3f}")


if __name__ == "__main__":
    app()
