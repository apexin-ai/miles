"""Train the full 1017 decoder plus native vision tower, or a vision-only control.

Requires the original multimodal checkpoint and an allocated external Ray cluster.
The image SFT contract is messages plus readable image paths, one <image> per path.
Native Engram tables and discrete routing-selection biases are fixed assets.

Args:
    hf_checkpoint: Native DeepSeek-V4.1-Flash-1017 checkpoint (including vision).
    prompt_data: Image SFT JSONL with messages and images fields.
    vision_trainable: none, projector or full; full is joint language/vision SFT.
    freeze_language: Freeze the decoder for a causal vision-only control.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_deepseek_v4_1_vision_sft.py \
        --hf-checkpoint /root/models/DeepSeek-V4.1-Flash-1017 \
        --prompt-data /root/datasets/image-sft/train.jsonl --num-nodes 24 \
        --output-dir /root/runs/dsv41-vision
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    hf_checkpoint: str = "/root/models/DeepSeek-V4.1-Flash-1017"
    prompt_data: str = "/root/datasets/image-sft/train.jsonl"
    num_nodes: int = 24
    num_gpus_per_node: int = 8
    tp_size: int = 8
    pp_size: int = 6
    ep_size: int = 32
    first_stage_layers: int | None = 6
    last_stage_layers: int | None = 6
    vision_trainable: Literal["none", "projector", "full"] = "full"
    freeze_language: bool = False
    num_rollout: int = 32
    rollout_batch_size: int = 32
    global_batch_size: int = 32
    seq_length: int = 4096
    lr: float = 1e-6
    save_interval: int = 32
    optimizer_offload: bool = True
    load: str | None = None
    rollout_seed: int = 42
    run_id: str = field(default_factory=U.create_run_id)
    extra_args: str = ""
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        if self.freeze_language and self.vision_trainable == "none":
            raise ValueError("Vision-only training requires a trainable visual scope")
        if self.num_nodes * self.num_gpus_per_node % (self.tp_size * self.pp_size):
            raise ValueError("GPU count must be divisible by TP times PP")
        if self.global_batch_size != self.rollout_batch_size:
            raise ValueError("This recipe uses exactly one optimizer update per rollout batch")


def execute(args: ScriptArgs):
    checkpoint_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint} "
    checkpoint_args += f"--save {args.output_dir}/{args.run_id}/checkpoints --save-interval {args.save_interval} "
    if args.load:
        checkpoint_args += f"--load {args.load} "
    rollout_args = (
        f"--prompt-data {args.prompt_data} --input-key messages "
        '--multimodal-keys \'{"image":"images"}\' --rollout-shuffle '
        f"--rollout-seed {args.rollout_seed} --num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} --n-samples-per-prompt 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--rollout-function-path miles_plugins.models.deepseek_v4_1.sft.generate_rollout "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tp_size} --pipeline-model-parallel-size {args.pp_size} "
        f"--expert-model-parallel-size {args.ep_size} --expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 --sequence-parallel --qkv-format bshd --micro-batch-size 1 "
        f"--seq-length {args.seq_length} --recompute-granularity full --recompute-method uniform "
        "--recompute-num-layers 1 --transformer-impl transformer_engine --dsv4-impl miles "
    )
    for flag, value in (
        ("decoder-first-pipeline-num-layers", args.first_stage_layers),
        ("decoder-last-pipeline-num-layers", args.last_stage_layers),
    ):
        if value is not None:
            perf_args += f"--{flag} {value} "
    algorithm_args = (
        "--debug-train-only --loss-type sft_loss "
        "--bf16 --calculate-per-token-loss --disable-compute-advantages-and-returns "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --model-name deepseekv41 "
        "--custom-model-provider-path miles_plugins.models.deepseek_v4_1.multimodal.deepseek_v41_vlm_model_provider "
        f"--dsv41-vision-trainable {args.vision_trainable} --dsv41-vision-recompute "
        "--moe-router-bias-update-rate 0 --moe-aux-loss-coeff 0 "
    )
    if args.freeze_language:
        algorithm_args += "--dsv41-freeze-language "
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 --grad-reduce-in-bf16 --main-grads-dtype bf16 "
    )
    if args.optimizer_offload:
        optimizer_args += "--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer "
    args.create_backend().execute_train(
        megatron_model_type="deepseek-v4.1",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_path=args.megatron_path,
        extra_env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        train_args=checkpoint_args
        + rollout_args
        + perf_args
        + algorithm_args
        + optimizer_args
        + U.get_default_wandb_args(__file__, run_id=args.run_id)
        + args.extra_args,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
