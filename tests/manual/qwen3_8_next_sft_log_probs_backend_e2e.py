"""One SFT step of the Qwen3.8-Flash-Next 4-layer model per ``--log-probs-backend``, to compare step 1.

The two runs differ only in ``--log-probs-backend``: same converted checkpoint, same multi-turn SFT data
(``prepare`` writes it), no shuffle, the same seed. Compare ``train/loss`` and ``train/grad_norm`` of rollout 0.
At TP 2 Megatron does not pad the 248320-token vocabulary, so both backends normalize over the same columns;
run ``torch`` twice to see the run-to-run noise first.

python tests/manual/qwen3_8_next_sft_log_probs_backend_e2e.py prepare
python tests/manual/qwen3_8_next_sft_log_probs_backend_e2e.py train --log-probs-backend torch
python tests/manual/qwen3_8_next_sft_log_probs_backend_e2e.py train --log-probs-backend fused
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MEGATRON_MODEL_TYPE = "qwen3.8-flash-next-4layer"
_DATA_FILE = "qwen3_8_next_sft_backend_check.jsonl"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    log_probs_backend: Literal["torch", "fused"] = "torch"
    num_gpus_per_node: int = 8
    hf_checkpoint: str = "/root/models/Qwen3.8-Flash-Next-4layer"
    ckpt_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_samples: int = 64
    extra_args: str = ""


def _conversation(i: int) -> dict:
    """Two user/assistant rounds: the assistant turns are trained, the rest is masked."""
    numbers = [(7 * i + 3 * k) % 97 for k in range(4 + i % 5)]
    steps = " ".join(f"Add {n}: the running total is {sum(numbers[: k + 1])}." for k, n in enumerate(numbers))
    return {
        "messages": [
            {"role": "system", "content": "You are a careful assistant. Show each step."},
            {"role": "user", "content": f"Add these numbers one at a time: {', '.join(map(str, numbers))}."},
            {"role": "assistant", "content": f"{steps} The sum is {sum(numbers)}."},
            {"role": "user", "content": "Now double it and explain why doubling is the same as adding it to itself."},
            {
                "role": "assistant",
                "content": f"Doubling {sum(numbers)} gives {2 * sum(numbers)}, because x + x = 2x. " * (1 + i % 3),
            },
        ]
    }


def prepare(args: ScriptArgs) -> None:
    data = Path(args.data_dir) / _DATA_FILE
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("".join(json.dumps(_conversation(i)) + "\n" for i in range(args.num_samples)))
    os.environ["CONVERT_KEEP_PP1"] = "1"
    args.create_backend().convert_checkpoint(
        model_name=Path(args.hf_checkpoint).name,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1",
    )


def train(args: ScriptArgs) -> None:
    torch_dist = f"{args.ckpt_dir}/{Path(args.hf_checkpoint).name}_torch_dist"
    sft_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {torch_dist} --ref-load {torch_dist} --no-load-optim --no-load-rng "
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.data_dir}/{_DATA_FILE} "
        "--input-key messages "
        "--loss-mask-type qwen3 "
        "--num-rollout 1 "
        f"--rollout-batch-size {args.num_samples} "
        f"--global-batch-size {args.num_samples} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
        "--seed 1234 "
        f"--log-probs-backend {args.log_probs_backend} "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--qkv-format thd "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--model-name qwen4_exp "
        "--linear-attention-backend flashqla "
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider "
    )
    args.create_backend().execute_train(
        train_args=f"{sft_args} {perf_args} {optimizer_args} {misc_args} {args.extra_args}",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@app.command(name="prepare")
@U.dataclass_cli
def prepare_command(args: ScriptArgs):
    prepare(args)


@app.command(name="train")
@U.dataclass_cli
def train_command(args: ScriptArgs):
    train(args)


if __name__ == "__main__":
    app()
