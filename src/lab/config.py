from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-8B"
    fallback: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"


@dataclass
class DataConfig:
    dataset_name: str = "gretelai/synthetic_text_to_sql"
    output_dir: str = "data"
    target_examples: int = 4000
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    max_prompt_chars: int = 4000
    max_sql_chars: int = 2000
    seed: int = 42


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class SFTSettings:
    epochs: float = 2.0
    max_steps: int = -1
    per_device_batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    output_dir: str = "checkpoints/sft"
    resume_from_checkpoint: bool = True


@dataclass
class DPOSettings:
    beta: float = 0.1
    epochs: float = 1.0
    max_steps: int = -1
    per_device_batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 5.0e-6
    bf16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "checkpoints/dpo"
    pairs_path: str = "data/dpo_pairs.jsonl"
    max_pairs: int = 1500


@dataclass
class EvalSettings:
    num_examples: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.0
    report_path: str = "reports/eval.md"
    judge_base_url: str = ""
    judge_model: str = ""


@dataclass
class ServeSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "checkpoints/merged"
    served_model_name: str = "qwen3-8b-sql"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    backend: str = "vllm"


@dataclass
class WandbSettings:
    enabled: bool = True
    project: str = "llm-finetune-inference-lab"
    run_name: str = ""


@dataclass
class LabConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    sft: SFTSettings = field(default_factory=SFTSettings)
    dpo: DPOSettings = field(default_factory=DPOSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    serve: ServeSettings = field(default_factory=ServeSettings)
    wandb: WandbSettings = field(default_factory=WandbSettings)
    drive_root: str = ""
    work_root: str = ""
    profile_name: str = "default"


def register_schema() -> None:
    store = ConfigStore.instance()
    store.store(name="lab_schema", node=LabConfig)


def load_config(overrides: list[str] | None = None) -> LabConfig:
    register_schema()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        composed = compose(config_name="config", overrides=overrides or [])
    return cast(LabConfig, OmegaConf.to_object(composed))
