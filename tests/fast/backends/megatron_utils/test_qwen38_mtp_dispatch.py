"""The multimodal provider and Megatron must agree on the native Qwen3.8 spec."""

import pytest

from miles_plugins.models.qwen3_8_next.mtp_config import is_qwen38_spec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (["miles_plugins.models.qwen3_8_next.qwen3_8_next", "get_qwen3_8_next_spec"], True),
        (("miles_plugins.models.qwen3_8_next.qwen3_8_next", "get_qwen3_8_next_spec"), True),
        (["other.qwen3_8_next", "get_qwen3_8_next_spec"], False),
        (["miles_plugins.models.qwen3_8_next.qwen3_8_next", "other_spec"], False),
        (None, False),
    ],
)
def test_qwen38_spec_selection(spec, expected):
    assert is_qwen38_spec(spec) is expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layer_types", ["linear_attention"]),
        ("num_hidden_layers", 2),
        ("mtp_use_hidden_state_from_layer", 16),
        ("rope_theta", 1000),
    ],
)
def test_reject_incompatible_checkpoint_draft_contract(field, value):
    from types import SimpleNamespace

    from miles_plugins.models.qwen3_8_next.mtp_config import validate_qwen38_mtp_config

    mtp = {"num_hidden_layers": 1, "layer_types": ["full_attention"], "rope_theta": 10000000}
    config = SimpleNamespace(
        mtp=mtp, mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False, rope_parameters={"rope_theta": 10000000}
    )
    validate_qwen38_mtp_config(config)
    config.mtp[field] = value
    with pytest.raises(ValueError, match="Qwen3.8 MTP"):
        validate_qwen38_mtp_config(config)
