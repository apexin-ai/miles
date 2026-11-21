"""Identify the custom language-model spec shared by Qwen3.8 text and image SFT."""

from collections.abc import Sequence


def is_qwen38_spec(spec: Sequence[str] | None) -> bool:
    return spec is not None and tuple(spec) == (
        "miles_plugins.models.qwen3_8_next.qwen3_8_next",
        "get_qwen3_8_next_spec",
    )


def validate_qwen38_mtp_config(text_config) -> None:
    """Match the checkpoint's draft architecture before binding its weights to the training head."""
    mtp = text_config.mtp
    if (
        text_config.mtp_num_hidden_layers != 1
        or text_config.mtp_use_dedicated_embeddings
        or not isinstance(mtp, dict)
        or mtp.get("num_hidden_layers") != 1
        or mtp.get("layer_types") != ["full_attention"]
        or mtp.get("mtp_use_hidden_state_from_layer") is not None
        or mtp.get("rope_theta") != text_config.rope_parameters["rope_theta"]
    ):
        raise ValueError(
            "Qwen3.8 MTP requires one full-attention layer using final HC states, shared embeddings and RoPE"
        )
