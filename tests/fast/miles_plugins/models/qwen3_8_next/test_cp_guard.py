"""Qwen3.8-Next supports CP only under allgather-CP, so zigzag must fail at model build.

QSA, the indexer and PLE all need whole-sequence natural order. Under allgather-CP each
rank holds one contiguous global range, so a single affine offset maps local to global.
Under the default zigzag layout a rank holds two disjoint ranges per sequence and no such
offset exists: positions would be placed wrong and the indexer would select over local keys
only, silently, or index past the shard when a pad segment is present.

The generic ``verify_allgather_cp_support`` (upstream #4471) only checks ``--allgather-cp``
and returns early without it; it cannot stand in for this guard, which must still reject
zigzag CP under its own message, naming the Qwen3.8-specific reason.
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


def _build_args(allgather_cp: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="/qwen-checkpoint",
        allgather_cp=allgather_cp,
        backend=SimpleNamespace(num_experts=4, position_embedding_type="rope"),
    )


def _build_config(**config_fields) -> SimpleNamespace:
    return SimpleNamespace(num_layers=2, moe_layer_freq=[1, 0], pipeline_model_parallel_layout=None, **config_fields)


def _build(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace, allgather_cp: bool = False) -> None:
    """Run the spec up to the block spec, which always raises. ``config`` is mutated in place
    and supplied by the caller (not constructed here and returned): a statement assigning a
    call's result never executes if the call itself raises before returning, so a caller that
    needs to inspect ``config`` after catching the expected exception must already hold its own
    reference from before the call, not rely on a return value it will never receive.
    """
    qwen3_8_next = importlib.import_module("miles_plugins.models.qwen3_8_next.qwen3_8_next")

    def block_spec(*args, **kwargs):
        raise _ReachedBlockSpec

    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: _text_config())
    monkeypatch.setattr(qwen3_8_next, "get_gpt_decoder_block_spec", block_spec)
    qwen3_8_next.get_qwen3_8_next_spec(args=_build_args(allgather_cp), config=config)


@pytest.mark.parametrize("cp", [2, 4])
def test_zigzag_context_parallel_is_rejected_before_any_layer_is_built(
    monkeypatch: pytest.MonkeyPatch, cp: int, gdn: ModuleType
) -> None:
    with pytest.raises(NotImplementedError, match="allgather-cp"):
        _build(monkeypatch, _build_config(context_parallel_size=cp), allgather_cp=False)


@pytest.mark.parametrize("cp", [2, 4])
def test_allgather_context_parallel_builds(monkeypatch: pytest.MonkeyPatch, cp: int, gdn: ModuleType) -> None:
    with pytest.raises(_ReachedBlockSpec):
        _build(monkeypatch, _build_config(context_parallel_size=cp), allgather_cp=True)


@pytest.mark.parametrize("allgather_cp", [False, True], ids=["zigzag", "allgather"])
@pytest.mark.parametrize("fields", [{"context_parallel_size": 1}, {}])
def test_cp1_or_unset_builds(
    monkeypatch: pytest.MonkeyPatch, fields: dict, allgather_cp: bool, gdn: ModuleType
) -> None:
    with pytest.raises(_ReachedBlockSpec):
        _build(monkeypatch, _build_config(**fields), allgather_cp=allgather_cp)


@pytest.mark.parametrize("allgather_cp", [False, True], ids=["zigzag", "allgather"])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_the_flag_is_set_on_every_build_not_only_when_cp_is_on(
    monkeypatch: pytest.MonkeyPatch, cp: int, allgather_cp: bool, gdn: ModuleType
) -> None:
    """``ops/cp.py``'s ``cp_enabled`` reads ``qwen3_8_next_allgather_cp`` unconditionally, so a
    CP=1 build must still set it explicitly -- "only set when CP>1" would leave it missing on
    every CP=1 build, which happens to be harmless today only because CP=1 never calls
    ``cp_enabled``. A future caller that does would hit a missing attribute instead of False.
    """
    if cp > 1 and not allgather_cp:
        pytest.skip("zigzag CP>1 never reaches the point where the flag is set")
    config = _build_config(context_parallel_size=cp)
    with pytest.raises(_ReachedBlockSpec):
        _build(monkeypatch, config, allgather_cp=allgather_cp)
    assert config.qwen3_8_next_allgather_cp is allgather_cp
