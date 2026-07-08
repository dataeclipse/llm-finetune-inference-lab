from lab.config import LabConfig
from lab.logging import get_logger
from lab.training.common import (
    build_lora_config,
    configure_wandb,
    load_base_model,
    load_split_dataset,
    load_tokenizer,
    resolve_checkpoint,
)

logger = get_logger(__name__)


def run_sft(config: LabConfig) -> str:
    from trl import SFTConfig, SFTTrainer

    report_to = configure_wandb(config.wandb)
    tokenizer = load_tokenizer(config.model.name)
    model = load_base_model(config.model)
    train_dataset = load_split_dataset(config, "train")
    val_dataset = load_split_dataset(config, "val")

    sft_config = SFTConfig(
        output_dir=config.sft.output_dir,
        num_train_epochs=config.sft.epochs,
        max_steps=config.sft.max_steps,
        per_device_train_batch_size=config.sft.per_device_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation,
        learning_rate=config.sft.learning_rate,
        lr_scheduler_type=config.sft.lr_scheduler,
        warmup_ratio=config.sft.warmup_ratio,
        bf16=config.sft.bf16,
        gradient_checkpointing=config.sft.gradient_checkpointing,
        logging_steps=config.sft.logging_steps,
        save_steps=config.sft.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if len(val_dataset) else "no",
        eval_steps=config.sft.save_steps,
        max_length=config.model.max_seq_length,
        report_to=report_to,
        run_name=config.wandb.run_name or None,
        seed=config.data.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) else None,
        processing_class=tokenizer,
        peft_config=build_lora_config(config.lora),
    )
    checkpoint = resolve_checkpoint(config.sft.output_dir, config.sft.resume_from_checkpoint)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(config.sft.output_dir)
    tokenizer.save_pretrained(config.sft.output_dir)
    logger.info(
        "sft_finished",
        output_dir=config.sft.output_dir,
        train_loss=round(float(result.training_loss), 4),
        steps=int(result.global_step),
    )
    return config.sft.output_dir
