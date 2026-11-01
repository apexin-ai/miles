from argparse import ArgumentParser

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import A, Arg, BaseConfig


class CustomMegatronPluginsConfig(BaseConfig):
    """
    Add custom Megatron plugins arguments.
    This is a placeholder for any additional arguments that might be needed.
    """

    custom_megatron_init_path: A[
        CustomFunctionConfig | None,
        Arg(),
    ] = None
    custom_megatron_before_log_prob_hook_path: A[
        CustomFunctionConfig | None,
        Arg(),
    ] = None
    custom_megatron_before_train_step_hook_path: A[
        CustomFunctionConfig | None,
        Arg(),
    ] = None


class Dsv4MegatronPluginsConfig(BaseConfig):
    dsv4_impl: A[
        str,
        Arg(
            choices=["miles", "megatron"],
            help=(
                "Which DeepSeek-V4 attention implementation to train with. 'miles' is the plugin path "
                "(BSHD, sparse context parallelism, tilelang kernels, miles' hyper-connections) and is "
                "the only one that supports tensor parallelism. 'megatron' is Megatron's native "
                "dsv4_hybrid path (THD, cuDNN or unfused kernels, native hyper-connections). The two "
                "read the same HuggingFace checkpoint but their torch_dist checkpoints are not "
                "interchangeable."
            ),
        ),
    ] = "megatron"
    dsv41_vision_trainable: A[
        str,
        Arg(
            choices=["none", "projector", "full"],
            help="V4.1 visual scope; requires the multimodal model provider and an image SFT dataset.",
        ),
    ] = "none"
    dsv41_vision_recompute: A[bool, Arg(help="Recompute trainable ViT blocks.")] = False
    dsv41_freeze_language: A[
        bool,
        Arg(help="Freeze language parameters before DDP/optimizer construction for a vision-only control."),
    ] = False


class Qwen38VisionMegatronPluginsConfig(BaseConfig):
    """Qwen3.8-Flash-Next's opt-in native image SFT controls."""

    qwen38_vision_trainable: A[str, Arg(choices=["none", "projector", "full"])] = "full"
    qwen38_vision_data: A[
        str,
        Arg(
            choices=["image", "text", "mixed"],
            help=(
                "What the training data holds. image: every sample has images. text: no sample has images; the "
                "vision tower is held but frozen (--qwen38-vision-trainable none) and never run. mixed: each sample "
                "may have images or not; text samples skip the vision tower."
            ),
        ),
    ] = "image"
    qwen38_freeze_language: A[bool, Arg()] = False
    qwen38_vision_recompute: A[bool, Arg()] = False
    qwen38_sft_final_answer_format: A[str, Arg(choices=["dataset", "integer-only"])] = "dataset"
    qwen38_audit_dir: A[str | None, Arg()] = None
    qwen38_audit_interval: A[
        int,
        Arg(
            help=(
                "With --qwen38-audit-dir: probe gradients, hash the vision tower and write a record every N "
                "updates; the final update is always audited and images are counted on every update."
            )
        ),
    ] = 1
    qwen38_visual_snapshot_interval: A[int, Arg()] = 0
    qwen38_vision_load_from_scope: A[
        str | None,
        Arg(
            choices=["none", "projector"],
            help=(
                "Once, on the resume that switches the vision scope up (e.g. projector -> full): the checkpoint was "
                "saved with this --qwen38-vision-trainable. Its optimizer state is loaded across the DDP layout "
                "change; the newly trainable vision parameters start with zero Adam moments "
                "(miles_plugins/models/qwen3_8_next/vision_scope_switch.py). Only Megatron's dp_reshardable format."
            ),
        ),
    ] = None
    qwen38_vit_lr_mult: A[
        float,
        Arg(
            help=(
                "With --qwen38-vision-trainable full: the ViT's lr (every vision parameter except the merger) = the "
                "language lr schedule times this (max and min lr both scaled). The merger keeps the language lr. "
                "1.0 keeps Megatron's standard param groups."
            )
        ),
    ] = 1.0
    qwen38_vit_wd_mult: A[
        float,
        Arg(
            help=(
                "With --qwen38-vision-trainable full: the ViT's weight decay = the language weight decay times this; "
                "ViT biases and 1-D parameters stay undecayed. 1.0 keeps Megatron's standard param groups."
            )
        ),
    ] = 1.0
    qwen38_vision_lazy: A[
        str,
        Arg(
            choices=["off", "merger", "vision"],
            help=(
                "Lazy Adam on steps whose global batch has no image: the merger (merger) or every trainable vision "
                "parameter (vision) keeps its weights, fp32 master and Adam moments, with no weight decay; its Adam "
                "step still follows the global one. Needs --optimizer-cpu-offload --optimizer-offload-fraction 1.0 "
                "(miles_plugins/models/qwen3_8_next/vision_lazy.py)."
            ),
        ),
    ] = "off"


class DsaMegatronPluginsConfig(BaseConfig):
    dsa_impl: A[str, Arg(choices=["miles", "megatron"])] = "miles"
    miles_dsa_topk_backend: A[
        str,
        Arg(
            choices=["canonical", "torch", "flashinfer"],
            help=(
                "DSA indexer top-k backend for both raw DSA implementations. 'canonical' is the exact top-k with "
                "ties at the k-th score broken toward the smaller key, like the rollout's top-k, returned in "
                "ascending key order; native Megatron DSA runs it as its exact 'torch' top-k."
            ),
        ),
    ] = "canonical"

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        if any(action.dest == "cp_comm_type" for action in parser._actions):
            parser.set_defaults(cp_comm_type=None)
