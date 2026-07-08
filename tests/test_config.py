from lab.config import LabConfig, load_config


def test_default_profile_is_cpu_smoke() -> None:
    config = load_config()
    assert isinstance(config, LabConfig)
    assert config.profile_name == "cpu_smoke"
    assert config.model.name == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert config.model.load_in_4bit is False
    assert config.sft.max_steps == 2
    assert config.wandb.enabled is False


def test_colab_profile_overrides() -> None:
    config = load_config(["profile=colab_a100"])
    assert config.profile_name == "colab_a100"
    assert config.model.name == "Qwen/Qwen3-8B"
    assert config.model.load_in_4bit is True
    assert config.sft.bf16 is True
    assert config.sft.output_dir.startswith("/content/drive")
    assert config.dpo.pairs_path.startswith("/content/drive")


def test_cli_style_overrides() -> None:
    config = load_config(["profile=colab_a100", "sft.learning_rate=1e-4", "lora.r=8"])
    assert config.sft.learning_rate == 1e-4
    assert config.lora.r == 8


def test_lora_defaults_target_all_projections() -> None:
    config = load_config()
    assert "q_proj" in config.lora.target_modules
    assert "down_proj" in config.lora.target_modules
    assert config.lora.alpha == 2 * config.lora.r


def test_data_split_fractions_valid() -> None:
    config = load_config(["profile=colab_a100"])
    assert 0 < config.data.val_fraction < 0.5
    assert 0 < config.data.test_fraction < 0.5
