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
