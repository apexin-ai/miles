"""The PLE conv segment bounds pass output_size to repeat_interleave: no device sync to read the lengths' sum
every forward, the same bounds as before."""

import pytest
import torch

from miles.kernels.embedding import ple_gate_conv


def _old_seg_bounds(T, cu_seqlens, device):
    """_seg_bounds as of 8325e8bd."""
    if cu_seqlens is None:
        lo = torch.zeros(T, dtype=torch.int32, device=device)
        hi = torch.full((T,), T, dtype=torch.int32, device=device)
        return lo, hi
    cu = cu_seqlens.to(torch.long)
    lens = (cu[1:] - cu[:-1]).clamp_min(0)
    lo = torch.repeat_interleave(cu[:-1], lens).to(torch.int32)
    hi = torch.repeat_interleave(cu[1:], lens).to(torch.int32)
    return lo, hi


@pytest.mark.parametrize("bounds", [(0, 5, 9, 16), (0, 16), (0, 3, 3, 16), (0, 0, 16), (0, 1, 2, 3, 15, 16), None])
def test_the_bounds_are_unchanged(bounds):
    cu = None if bounds is None else torch.tensor(bounds, dtype=torch.int32)
    for old, new in zip(_old_seg_bounds(16, cu, "cpu"), ple_gate_conv._seg_bounds(16, cu, "cpu"), strict=True):
        assert new.dtype == old.dtype and torch.equal(new, old)


def test_the_lengths_are_not_read_on_the_host():
    """A meta tensor holds no data: repeat_interleave without output_size refuses it (it must read the sum)."""
    cu = torch.empty(4, dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="output_size"):
        _old_seg_bounds(16, cu, "meta")  # positive control
    lo, hi = ple_gate_conv._seg_bounds(16, cu, "meta")
    assert lo.shape == hi.shape == (16,) and lo.dtype == hi.dtype == torch.int32
