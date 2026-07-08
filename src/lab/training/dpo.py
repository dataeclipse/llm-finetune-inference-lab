from pathlib import Path

from lab.config import LabConfig
from lab.exceptions import TrainingError
from lab.logging import get_logger
from lab.training.common import (
    configure_wandb,
    load_base_model,
    load_tokenizer,
    resolve_checkpoint,
)

logger = get_logger(__name__)


def run_dpo(config: LabConfig, adapter_path: str | None = None) -> str:
    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    pairs_path = Path(config.dpo.pairs_path)
    if not pairs_path.exists():
        raise TrainingError(f"dpo pairs not found at {pairs_path}, run pair generation first")

    report_to = configure_wandb(config.wandb)
    tokenizer = load_tokenizer(config.model.name)
    base_model = load_base_model(config.model)
    source_adapter = adapter_path or config.sft.output_dir
    model = PeftModel.from_pretrained(base_model, source_adapter, is_trainable=True)
    dataset = load_dataset("json", data_files=str(pairs_path), split="train")
    if config.dpo.max_pairs > 0:
        dataset = dataset.select(range(min(config.dpo.max_pairs, len(dataset))))

    dpo_config = DPOConfig(
        output_dir=config.dpo.output_dir,
        beta=config.dpo.beta,
        num_train_epochs=config.dpo.epochs,
        max_steps=config.dpo.max_steps,
        per_device_train_batch_size=config.dpo.per_device_batch_size,
        gradient_accumulation_steps=config.dpo.gradient_accumulation,
        learning_rate=config.dpo.learning_rate,
        bf16=config.dpo.bf16,
        gradient_checkpointing=config.dpo.gradient_checkpointing,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        max_length=config.model.max_seq_length,
        report_to=report_to,
        run_name=config.wandb.run_name or None,
        seed=config.data.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    checkpoint = resolve_checkpoint(config.dpo.output_dir, resume=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(config.dpo.output_dir)
    tokenizer.save_pretrained(config.dpo.output_dir)
    logger.info(
        "dpo_finished",
        output_dir=config.dpo.output_dir,
        train_loss=round(float(result.training_loss), 4),
        steps=int(result.global_step),
    )
    return config.dpo.output_dir
