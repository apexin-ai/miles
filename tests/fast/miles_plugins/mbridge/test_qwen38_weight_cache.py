"""The Qwen3.8 bridge's device cache must never hand one HF weight's upload to another (G-gaps 8.2).

The cache keys on the source tensor's address. Before 0aa233fb the source was dropped after the upload,
so a later expert matrix allocated at the freed address loaded the previous one's values silently.
"""

import pytest
import torch

pytest.importorskip("mbridge")


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch):
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    # CPU stand-ins: "upload" is a copy, and the parent's mcore formatting passes the uploads through
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(Qwen3_5Bridge, "_weight_to_mcore_format", lambda self, name, weights: weights)
    return Qwen38NextBridge.__new__(Qwen38NextBridge)


def test_weights_read_one_after_another_never_get_a_stale_upload(bridge) -> None:
    """Each read is a fresh tensor that dies after use, so the allocator reuses addresses; every
    upload must still carry its own weight's values."""
    seen_addresses = set()
    for value in range(64):
        source = torch.full((1024,), float(value))
        seen_addresses.add(source.data_ptr())
        [uploaded] = bridge._weight_to_mcore_format(f"decoder.layers.{value}.mlp.experts.linear_fc1.weight", [source])
        assert torch.equal(uploaded, torch.full((1024,), float(value))), value
        del source, uploaded
    # the scenario is real only if the allocator did hand out an address again
    assert len(seen_addresses) < 64


def test_live_views_with_the_same_address_and_shape_are_told_apart(bridge) -> None:
    """A square matrix and its transpose share address, shape and dtype but not the layout."""
    matrix = torch.arange(16.0).reshape(4, 4)
    plain, transposed = bridge._weight_to_mcore_format("w", [matrix, matrix.t()])
    assert torch.equal(plain, matrix)
    assert torch.equal(transposed, matrix.t())


def test_a_repeated_read_of_the_same_weight_reuses_the_upload(bridge) -> None:
    """The cache still saves the upload when consecutive experts slice one fused HF tensor."""
    fused = torch.randn(8, 6)
    first = bridge._weight_to_mcore_format("decoder.layers.0.mlp.experts.linear_fc1.weight0", [fused])[0]
    second = bridge._weight_to_mcore_format("decoder.layers.0.mlp.experts.linear_fc1.weight1", [fused])[0]
    assert second is first
