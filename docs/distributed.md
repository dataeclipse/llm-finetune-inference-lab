# Distributed Training Configurations

Tested on a single A100 40GB; the configs in `configs/distributed/` are ready
for multi-GPU and multi-node runs without code changes, because both trainers
go through `accelerate`.

## Launching

```bash
accelerate launch --config_file configs/distributed/accelerate_fsdp.yaml \
  -m lab.cli train sft profile=colab_a100

accelerate launch --config_file configs/distributed/accelerate_zero3.yaml \
  -m lab.cli train sft profile=colab_a100
```

`num_processes` must match the GPU count; for multi-node add
`--num_machines`, `--machine_rank` and `--main_process_ip`.

## What ZeRO stages shard

Data-parallel training keeps a full copy of parameters, gradients and
optimizer state on every GPU. ZeRO removes that redundancy incrementally:

| Stage | Shards | Memory per GPU (model of size M, Adam) | Communication overhead |
|-------|--------|----------------------------------------|------------------------|
| ZeRO-1 | optimizer state | 4M + 12M/N | none beyond DDP |
| ZeRO-2 | + gradients | 2M + 14M/N | reduce-scatter instead of all-reduce |
| ZeRO-3 | + parameters | 16M/N | all-gather of layer params per forward/backward |

N = number of GPUs. With Adam in mixed precision, optimizer state dominates
(12 bytes per parameter versus 2 for bf16 weights), so ZeRO-1/2 already cover
most single-node cases; ZeRO-3 is what makes models that do not fit on one
GPU trainable at all, at the cost of parameter all-gathers on every layer.

## FSDP versus DeepSpeed ZeRO-3

Both implement full parameter sharding. FSDP (`FULL_SHARD` ≈ ZeRO-3) is
native PyTorch: fewer dependencies, `SHARDED_STATE_DICT` checkpoints scale to
large models, first-class `torch.compile` support. DeepSpeed adds mature CPU
and NVMe offload (`offload_optimizer.device: cpu`) which trades step time for
capacity — the practical route to fine-tune 70B-class models on limited GPU
counts.

## Offload

Both configs ship with offload disabled: on a single A100 40GB, QLoRA already
fits the 8B model and offload would only slow the step. To trade speed for
capacity set `offload_optimizer.device` and `offload_param.device` to `cpu`
in `deepspeed_zero3.json`, or `fsdp_offload_params: true` in the FSDP config.

## How this scales

- Single A100 (this project): QLoRA, no sharding needed; configs act as a
  validated starting point.
- Single node, 8 GPUs: either config as-is; prefer FSDP for simplicity.
- Multi-node: same configs plus rendezvous flags; DeepSpeed with CPU offload
  when the model no longer fits even sharded.
