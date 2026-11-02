import argparse
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
    qwen38_reload_check_iterations: A[
        list[int] | None,
        Arg(
            type_parser=int,
            nargs="+",
            help=(
                "With --custom-megatron-init-path miles_plugins.models.qwen3_8_next.vision_audit."
                "install_final_save_check: also run the save -> reload -> per-parameter compare check after the saves "
                "of these iterations (rollout ids), not only after the final save. Results go to "
                "<--qwen38-audit-dir>/checkpoint-reload/iter_NNNNNNN/; a failure stops the run on every rank."
            ),
        ),
    ] = None
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
                "step still follows the global one. Needs --optimizer-cpu-offload (any --optimizer-offload-fraction > 0) "
                "(miles_plugins/models/qwen3_8_next/vision_lazy.py)."
            ),
        ),
    ] = "off"
    qwen38_gdn_head_sharded: A[
        bool,
        Arg(
            help=(
                "Run Qwen3.8-Next's GDN layers head-sharded across TP (Qwen3.5's layer: each TP rank computes 1 / TP "
                "of the heads) instead of replicated on every TP rank. Same parameter names, but q/k/v and conv rows "
                "are group-major and TP-sharded, so torch_dist checkpoints of the two layouts are not interchangeable; "
                "HF load and export convert either way."
            )
        ),
    ] = False
    qwen38_indexer_score_dtype: A[
        str,
        Arg(
            choices=["fp32", "bf16"],
            help=(
                "Operand precision of the frozen QSA indexer's block scores. fp32: fp32 einsum (SIMT SGEMM on "
                "Blackwell, the reference). bf16: one fused Triton kernel, bf16 operands with fp32 accumulation and "
                "fp32 logits, as sglang's prefill indexer; the indexer's q and keys are bf16 already, so only the "
                "fp32 summation order differs. Selection only; no parameter or checkpoint change."
            ),
        ),
    ] = "fp32"
    qwen38_indexer_tp_shard: A[
        bool,
        Arg(
            help=(
                "Split the QSA indexer's query rows across TP (rank r scores rows r, r + TP, ...) and all-gather the "
                "selected block ids, instead of every TP rank scoring the whole SP-gathered sequence. Same selection; "
                "no parameter or checkpoint change."
            )
        ),
    ] = False
    qwen38_ple_deterministic_conv_grad: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Compute the PLE causal conv's weight gradient from per-token-block fp32 partial sums and a "
                "fixed-order reduction instead of fp32 atomics over all tokens, so it is bitwise reproducible from "
                "run to run and closer to the fp64 sum (miles/kernels/embedding/ple_gate_conv.py). Also faster: "
                "the atomics serialize on W * K addresses. Other gradients and the forward are unchanged; no "
                "parameter or checkpoint change. --no-qwen38-ple-deterministic-conv-grad restores the atomics."
            ),
        ),
    ] = True
    qwen38_hc_matmul_precision: A[
        str,
        Arg(
            choices=["fp32", "tf32", "bf16"],
            help=(
                "Precision of the hyper-connection gate chain's matmuls (low-rank mix down/up projections, inject "
                "gate, their input and weight gradients; miles/kernels/hyper_connection/hc.py). fp32: fp32 operands "
                "and math (SIMT SGEMM on Blackwell), the reference. tf32: the same fp32 matmuls with TF32 tensor "
                "cores allowed around these calls only. bf16: bf16 operands (the HC weights are bf16 already; the "
                "normed activations are rounded) with fp32 accumulation and fp32 results, as sglang's fused hc_mix. "
                "Elementwise math stays fp32. No parameter or checkpoint change."
            ),
        ),
    ] = "fp32"
    qwen38_fla_conv_pinned: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Run fla's NB-keyed Triton kernels on the GDN path -- causal_conv1d (the short convolution), "
                "layer_norm_gated (FusedRMSNormGated) and l2norm (q/k norm) -- with one fixed config each and a "
                "constant NB instead of autotuning per row-length bucket "
                "(miles_plugins/models/qwen3_8_next/ops/fla_conv_tuning.py). Without it every new row length "
                "compiles and benchmarks 16 conv configs (~15 s forward, ~35 s backward; CP2 has its own key) "
                "and every new norm bucket 9-25 configs, and near-tied configs can differ between ranks and runs. "
                "The pinned configs are the ones autotune picks on B300; outputs equal that config's bit for bit. "
                "No parameter or checkpoint change."
            ),
        ),
    ] = False


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
