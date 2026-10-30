"""Train Qwen3.8-Flash-Next's complete native vision tower, optionally with language.

Requires the full HF checkpoint and an allocated, already-running Ray cluster.
No layer slicing, LoRA or generated training targets are used by this recipe.

Args:
    hf_checkpoint: Complete native Qwen/Qwen3.8-Flash-Next directory.
    prompt_data: JSONL with messages and readable images, using one <image> per path.
    vision_trainable: none, projector or full (the entire native ViT and merger).
    freeze_language: Freeze the complete decoder before constructing its optimizer.
    tp_size: Tensor parallelism; TP2/PP4/EP2 is the single-node bring-up topology.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_8_next_vision_sft.py \
        --hf-checkpoint /root/models/Qwen3.8-Flash-Next \
        --prompt-data /root/datasets/image-sft/train.jsonl \
        --output-dir /root/runs/qwen38-vision --freeze-language
"""

import json
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    skip_process_cleanup: bool = True
    hf_checkpoint: str = "/root/models/Qwen3.8-Flash-Next"
    prompt_data: str = "/root/datasets/image-sft/train.jsonl"
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    tp_size: int = 2
    pp_size: int = 4
    ep_size: int = 2
    vision_trainable: Literal["none", "projector", "full"] = "full"
    freeze_language: bool = False
    optimizer_offload: bool = True
    num_rollout: int = 32
    rollout_batch_size: int = 32
    global_batch_size: int = 32
    seq_length: int = 4096
    lr: float = 1e-6
    weight_decay: float = 0.0
    save_interval: int = 32
    visual_snapshot_interval: int = 8
    rollout_seed: int = 42
    enable_thinking: bool = False
    reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh"
    final_answer_format: Literal["dataset", "integer-only"] = "dataset"
    audit: bool = True
    checkpoint_reload_check: bool = True
    load: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    run_id: str = field(default_factory=U.create_run_id)
    extra_args: str = ""

    def __post_init__(self):
        world = self.num_nodes * self.num_gpus_per_node
        if min(self.tp_size, self.pp_size, self.ep_size) < 1 or world % (self.tp_size * self.pp_size):
            raise ValueError("GPU count must be divisible by TP times PP")
        if world % (self.pp_size * self.ep_size) or 512 % self.ep_size or 48 % self.pp_size:
            raise ValueError("Expert and pipeline layouts must divide the full 48-layer, 512-expert model")
        if self.freeze_language and self.vision_trainable == "none":
            raise ValueError("Vision-only SFT requires trainable visual parameters")
        if self.global_batch_size != self.rollout_batch_size:
            raise ValueError("This recipe takes one optimizer step per supervised batch")
        if min(self.num_rollout, self.global_batch_size, self.seq_length, self.save_interval) < 1:
            raise ValueError("Training lengths, batch sizes and save interval must be positive")
        if self.visual_snapshot_interval < 0 or (self.visual_snapshot_interval and not self.audit):
            raise ValueError("Visual snapshots require an audit and a nonnegative interval")
        if self.checkpoint_reload_check and not self.audit:
            raise ValueError("Native checkpoint qualification requires an audit directory")


def execute(args: ScriptArgs):
    run = f"{args.output_dir}/{args.run_id}"
    checkpoint_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint} "
        f"--save {run}/checkpoints --save-interval {args.save_interval} --no-save-optim --no-save-rng "
    )
    if args.load:
        checkpoint_args += f"--load {args.load} --no-load-optim --no-load-rng "
    template = json.dumps({"enable_thinking": args.enable_thinking, "reasoning_effort": args.reasoning_effort})
    rollout_args = (
        f'--prompt-data {args.prompt_data} --input-key messages --multimodal-keys \'{{"image":"images"}}\' '
        f"--apply-chat-template-kwargs '{template}' --rollout-shuffle --rollout-seed {args.rollout_seed} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--global-batch-size {args.global_batch_size} --n-samples-per-prompt 1 --num-steps-per-rollout 1 "
        "--rollout-function-path miles_plugins.models.qwen3_8_next.vision_sft.generate_rollout "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tp_size} --pipeline-model-parallel-size {args.pp_size} "
        f"--expert-model-parallel-size {args.ep_size} --expert-tensor-parallel-size 1 "
        "--sequence-parallel --context-parallel-size 1 --qkv-format thd --micro-batch-size 1 "
        f"--seq-length {args.seq_length} --recompute-granularity full --recompute-method uniform "
        "--recompute-num-layers 1 --linear-attention-backend fla "
    )
    algorithm_args = (
        "--debug-train-only --loss-type sft_loss --calculate-per-token-loss "
        "--disable-compute-advantages-and-returns --bf16 --model-name qwen4_exp "
        "--custom-model-provider-path miles_plugins.models.qwen3_8_next.vision.get_qwen38_vision_model_provider "
        f"--qwen38-vision-trainable {args.vision_trainable} --qwen38-vision-recompute "
        f"--qwen38-sft-final-answer-format {args.final_answer_format} "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --attention-dropout 0 --hidden-dropout 0 "
    )
    if args.freeze_language:
        algorithm_args += "--qwen38-freeze-language "
    if args.audit:
        algorithm_args += (
            f"--qwen38-audit-dir {run}/audit --qwen38-visual-snapshot-interval {args.visual_snapshot_interval} "
            "--custom-megatron-before-train-step-hook-path miles_plugins.models.qwen3_8_next.vision_audit.before_step "
        )
    if args.checkpoint_reload_check:
        algorithm_args += (
            "--custom-megatron-init-path miles_plugins.models.qwen3_8_next.vision_audit.install_final_save_check "
        )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay {args.weight_decay} "
        "--adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1 --seed 42 "
    )
    if args.optimizer_offload:
        optimizer_args += "--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer "
    args.create_backend().execute_train(
        megatron_model_type="qwen3.8-flash-next-vision",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_path=args.megatron_path,
        extra_env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"},
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
