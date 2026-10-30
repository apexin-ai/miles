"""Qwen3.8-Flash-Next's opt-in native image SFT controls."""


def add_qwen38_vision_arguments(parser):
    group = parser.add_argument_group("Qwen3.8-Flash-Next image SFT")
    group.add_argument("--qwen38-vision-trainable", choices=("none", "projector", "full"), default="full")
    group.add_argument("--qwen38-freeze-language", action="store_true")
    group.add_argument("--qwen38-vision-recompute", action="store_true")
    group.add_argument("--qwen38-sft-final-answer-format", choices=("dataset", "integer-only"), default="dataset")
    group.add_argument("--qwen38-audit-dir", type=str, default=None)
    group.add_argument("--qwen38-visual-snapshot-interval", type=int, default=0)
    return parser
