"""The PLE n-gram contexts take the pack boundaries from miles' host copy (packed_seq_params.cu_seqlens_host): no
``cu_seqlens.tolist()`` every forward, the same contexts as before."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.qwen3_8_next import model_provider
from miles_plugins.models.qwen3_8_next.ops import ple


def _old_build_ngram_contexts_packed(tokens, cu_seqlens, ngram_size, eos_token_id):
    """build_ngram_contexts_packed as of 8325e8bd."""
    if cu_seqlens is None:
        return ple.build_ngram_contexts(tokens, ngram_size, eos_token_id)
    bounds = cu_seqlens.tolist()
    parts = [
        ple.build_ngram_contexts(tokens[lo:hi], ngram_size, eos_token_id)
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        if hi > lo
    ]
    return torch.cat(parts, dim=0)


@pytest.mark.parametrize("bounds", [(0, 5, 9, 16), (0, 16), (0, 3, 3, 16), (0, 1, 2, 3, 16)])
@pytest.mark.parametrize("ngram_size", [2, 3])
def test_host_bounds_give_the_same_contexts(bounds, ngram_size):
    torch.manual_seed(0)
    tokens = torch.randint(0, 50, (16,))
    tokens[7] = 1  # an EOS inside a row
    cu = torch.tensor(bounds, dtype=torch.int32)
    old = _old_build_ngram_contexts_packed(tokens, cu, ngram_size, eos_token_id=1)
    new = ple.build_ngram_contexts_packed(tokens, cu, ngram_size, eos_token_id=1, cu_seqlens_host=bounds)
    assert torch.equal(new, old)
    assert torch.equal(ple.build_ngram_contexts_packed(tokens, cu, ngram_size, eos_token_id=1), old)


def test_the_device_boundaries_are_not_read():
    """Meta tensors hold no data: cu_seqlens.tolist() on one raises."""
    tokens = torch.arange(10, 26)
    cu = torch.empty(3, dtype=torch.int32, device="meta")
    with pytest.raises(NotImplementedError):
        _old_build_ngram_contexts_packed(tokens, cu, 3, eos_token_id=1)  # positive control
    out = ple.build_ngram_contexts_packed(tokens, cu, 3, eos_token_id=1, cu_seqlens_host=(0, 6, 16))
    assert torch.equal(out, _old_build_ngram_contexts_packed(tokens, torch.tensor([0, 6, 16]), 3, eos_token_id=1))


def test_the_pre_hook_passes_the_host_copy(monkeypatch):
    seen = {}

    class _Embedding:
        ngram_size, eos_token_id = 3, 1

        def compute_ngram_ids(self, contexts):
            seen["contexts"] = contexts
            return contexts.sum(-1, keepdim=True)

    monkeypatch.setattr(model_provider, "_find_ple_embedding", lambda model: _Embedding())
    monkeypatch.setattr(model_provider, "publish_ple_batch", lambda ids, cu: seen.update(ids=ids, cu=cu))

    class _Model(torch.nn.Module):
        def forward(self, input_ids=None, packed_seq_params=None):
            return input_ids

    model = _Model()
    model_provider._install_ple_context_hooks(model)
    tokens = torch.arange(10, 26)[None]
    cu = torch.empty(3, dtype=torch.int32, device="meta")  # read it and the forward raises
    model(input_ids=tokens, packed_seq_params=SimpleNamespace(cu_seqlens_q=cu, cu_seqlens_host=(0, 6, 16)))
    expected = _old_build_ngram_contexts_packed(tokens[0], torch.tensor([0, 6, 16]), 3, eos_token_id=1)
    assert torch.equal(seen["contexts"], expected) and seen["cu"] is cu
