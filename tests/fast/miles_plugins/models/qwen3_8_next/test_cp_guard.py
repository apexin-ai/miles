"""Qwen3.8-Next has no context-parallel path, so CP>1 must fail at model build.

QSA, the indexer and PLE all treat the local CP shard as the whole sequence: on a
zigzag shard they place positions wrong and only see local K/V, silently, or index
past the shard when a pad segment is present.
"""

import importlib
from types import ModuleType, SimpleNamespace

import pytest
from tests.fast.miles_plugins.models.qwen3_8_next.conftest import gdn

__all__ = ["gdn"]


class _ReachedBlockSpec(Exception):
    pass


def _text_config():
    return SimpleNamespace(
        layer_types=["linear_attention", "full_attention"],
        hc_count=4,
        hc_lowrank=2,
        ple_layer_ids=[],
        ple_embed_dim=8,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=17,
        split_ngram_parts=1,
        ple_conv_kernel_size=3,
        eos_token_id=7,
        indexer_budget=11,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_head_dim=4,
        indexer_kv_heads=1,
    )


def _build(monkeypatch: pytest.MonkeyPatch, **config_fields):
    qwen3_8_next = importlib.import_module("miles_plugins.models.qwen3_8_next.qwen3_8_next")

    def block_spec(*args, **kwargs):
        raise _ReachedBlockSpec

    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: _text_config())
    monkeypatch.setattr(qwen3_8_next, "get_gpt_decoder_block_spec", block_spec)
    args = SimpleNamespace(
        hf_checkpoint="/qwen-checkpoint", position_embedding_type="rope", backend=SimpleNamespace(num_experts=4)
    )
    config = SimpleNamespace(num_layers=2, moe_layer_freq=[1, 0], pipeline_model_parallel_layout=None, **config_fields)
    return qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)


@pytest.mark.parametrize("cp", [2, 4])
def test_context_parallel_is_rejected_before_any_layer_is_built(
    monkeypatch: pytest.MonkeyPatch, cp: int, gdn: ModuleType
) -> None:
    with pytest.raises(NotImplementedError, match="context parallel"):
        _build(monkeypatch, context_parallel_size=cp)


@pytest.mark.parametrize("fields", [{"context_parallel_size": 1}, {}])
def test_cp1_or_unset_builds(monkeypatch: pytest.MonkeyPatch, fields: dict, gdn: ModuleType) -> None:
    with pytest.raises(_ReachedBlockSpec):
        _build(monkeypatch, **fields)
