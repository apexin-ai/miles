"""loss_function's log record built without a device-to-host copy per 0-dim tensor (GBSMIG host-sync work,
2026-11-02): stack_log_values must give exactly what torch.tensor(items) gives (dtype and values)."""

import itertools

import pytest
import torch

from miles.backends.training_utils.loss.objective import stack_log_values

SCALARS = [
    3,
    0,
    262144,
    2.5,
    -0.0,
    1e-30,
    torch.tensor(7),
    torch.tensor(7, dtype=torch.int32),
    torch.tensor(131072, dtype=torch.int64),
    torch.tensor(0.3125),
    torch.tensor(1.2345678),
    torch.tensor(2.75, dtype=torch.bfloat16),
    torch.tensor(0.1, dtype=torch.float16),
    torch.tensor(0.1, dtype=torch.float64),
    torch.tensor(True),
    torch.tensor([4.0]),  # one element, 1-dim: torch.tensor keeps its value
]


@pytest.mark.parametrize(
    "items",
    [
        list(c)
        for n in (1, 2, 3)
        for c in itertools.product(SCALARS, repeat=n)
        if any(isinstance(v, torch.Tensor) for v in c)
    ][:400],
)
def test_same_as_torch_tensor(items):
    want = torch.tensor(items)
    got = stack_log_values(items, device="cpu")
    assert got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want), (items, got, want)


def test_python_only_and_multi_element_keep_torch_tensor():
    assert torch.equal(stack_log_values([1, 2.5], device="cpu"), torch.tensor([1, 2.5]))
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        torch.tensor([torch.tensor([1.0, 2.0])])
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        stack_log_values([torch.tensor([1.0, 2.0])], device="cpu")  # same error as torch.tensor


def test_the_sft_record_shape():
    # loss_function: [num_tokens (sum of loss masks, a 0-dim tensor)] + the loss's detached values
    masks = [torch.tensor([0, 1, 1, 1]), torch.tensor([1, 1, 0])]
    num_tokens = sum(m.sum() for m in masks)
    items = [num_tokens, torch.tensor(1.7311234, dtype=torch.float32)]
    assert torch.equal(stack_log_values(items, device="cpu"), torch.tensor(items))
    assert stack_log_values(items, device="cpu").dtype == torch.float32
