from prefect import flow, task

from lab.config import LabConfig, load_config
from lab.logging import configure_logging, get_logger

logger = get_logger(__name__)


@task(retries=1)
def prepare_data(config: LabConfig) -> int:
    from lab.data.prepare import prepare_dataset

    summary = prepare_dataset(config)
    return summary.train_examples


@task
def train_sft_task(config: LabConfig) -> str:
    from lab.training.sft import run_sft

    return run_sft(config)


@task
def train_dpo_task(config: LabConfig, adapter_path: str) -> str:
    from pathlib import Path

    from lab.training.dpo import run_dpo

    if not Path(config.dpo.pairs_path).exists():
        logger.warning("dpo_skipped", reason="no preference pairs file")
        return adapter_path
    return run_dpo(config, adapter_path=adapter_path)


@task
def eval_task(config: LabConfig, model_path: str) -> float:
    from lab.eval.runner import run_eval

    report = run_eval(config, model_path=model_path)
    return report.overall_accuracy


@task
def merge_task(config: LabConfig, adapter_path: str) -> str:
    from lab.quantization.merge import merge_adapter

    return merge_adapter(config, adapter_path=adapter_path)


@flow(name="finetune-pipeline")
def finetune_pipeline(profile: str = "colab_a100") -> None:
    configure_logging()
    config = load_config([f"profile={profile}"])
    train_rows = prepare_data(config)
    logger.info("pipeline_data_ready", train_rows=train_rows)
    sft_dir = train_sft_task(config)
    final_adapter = train_dpo_task(config, sft_dir)
    accuracy = eval_task(config, final_adapter)
    merged = merge_task(config, final_adapter)
    logger.info("pipeline_finished", accuracy=accuracy, merged=merged)


if __name__ == "__main__":
    finetune_pipeline()
