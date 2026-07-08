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


if __name__ == "__main__":
    app()
