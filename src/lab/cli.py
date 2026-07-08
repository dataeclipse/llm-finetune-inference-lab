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


export_app = typer.Typer(no_args_is_help=True)
app.add_typer(export_app, name="export", help="Quantized exports")


@export_app.command("merge")
def export_merge(
    adapter_path: str = typer.Option(""),
    output_dir: str = typer.Option(""),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.quantization.merge import merge_adapter

    config = setup(overrides or [])
    target = merge_adapter(config, adapter_path=adapter_path or None, output_dir=output_dir or None)
    typer.echo(f"merged model saved to {target}")


@export_app.command("awq")
def export_awq(
    model_path: str = typer.Option(""),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.quantization.awq import export_awq_model

    config = setup(overrides or [])
    target = export_awq_model(config, model_path=model_path or None)
    typer.echo(f"awq model saved to {target}")


@export_app.command("gguf")
def export_gguf(
    model_path: str = typer.Option(""),
    quant_type: str = typer.Option("Q4_K_M"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.quantization.gguf import export_gguf_model

    config = setup(overrides or [])
    target = export_gguf_model(config, model_path=model_path or None, quant_type=quant_type)
    typer.echo(f"gguf model saved to {target}")


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


serve_app = typer.Typer(no_args_is_help=True)
app.add_typer(serve_app, name="serve", help="Inference serving")
bench_app = typer.Typer(no_args_is_help=True)
app.add_typer(bench_app, name="bench", help="Benchmarks and A/B tests")


@serve_app.command("vllm")
def serve_vllm(overrides: list[str] = typer.Argument(None)) -> None:
    from lab.serving.vllm_server import launch_vllm

    config = setup(overrides or [])
    process = launch_vllm(config)
    process.wait()


@bench_app.command("latency")
def bench_latency(
    base_url: str = typer.Option("http://localhost:8000/v1"),
    concurrency: int = typer.Option(8),
    requests: int = typer.Option(64),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.serving.bench import run_benchmark

    config = setup(overrides or [])
    report = run_benchmark(
        config, base_url=base_url, concurrency=concurrency, total_requests=requests
    )
    typer.echo(
        f"p50={report.latency_p50:.3f}s p95={report.latency_p95:.3f}s "
        f"tps={report.throughput_tokens_per_second:.1f}"
    )


@bench_app.command("ab")
def bench_ab(
    base_url_a: str = typer.Option(..., help="OpenAI-compatible endpoint for model A"),
    base_url_b: str = typer.Option(..., help="OpenAI-compatible endpoint for model B"),
    model_a: str = typer.Option("model-a"),
    model_b: str = typer.Option("model-b"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    from lab.serving.ab_test import run_ab_test

    config = setup(overrides or [])
    result = run_ab_test(
        config, base_url_a=base_url_a, base_url_b=base_url_b, model_a=model_a, model_b=model_b
    )
    typer.echo(f"{result.model_a}={result.accuracy_a:.3f} {result.model_b}={result.accuracy_b:.3f}")


@bench_app.command("gpu")
def bench_gpu(
    output: str = typer.Option("reports/gpu_samples.csv"),
    duration: float = typer.Option(60.0),
    interval: float = typer.Option(5.0),
) -> None:
    from pathlib import Path

    from lab.monitoring.gpu import sample_to_csv

    configure_logging()
    samples = sample_to_csv(Path(output), duration_seconds=duration, interval_seconds=interval)
    typer.echo(f"captured {samples} samples to {output}")


if __name__ == "__main__":
    app()
