"""loss_function builds its metric values on the device: no ``.item()`` per metric and no blocking H2D every
microbatch, with the dtype and bits that ``torch.tensor(values, device=...)`` gave."""

import pytest
import torch

from miles.backends.training_utils.loss.objective import _metric_values

_BITS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}

CASES = {
    "sft-per-token": lambda: [torch.tensor(37), torch.tensor(1.2345678)],
    "count-and-specials": lambda: [
        5,
        torch.tensor(0.1),
        torch.tensor(-0.0),
        torch.tensor(float("nan")),
        torch.tensor(float("-inf")),
    ],
    "bf16-metric": lambda: [3, torch.tensor(2.5078125, dtype=torch.bfloat16)],
    "count-past-fp32-precision": lambda: [2**40 + 1, torch.tensor(1.0)],
    "int32-and-fp64": lambda: [torch.tensor(7, dtype=torch.int32), torch.tensor(1 / 3, dtype=torch.float64)],
    "python-float-and-bool": lambda: [4, 0.3, torch.tensor(True)],
    "all-integer": lambda: [torch.tensor(3), torch.tensor(4)],
    "grad-tensor": lambda: [1, torch.tensor(1.5, requires_grad=True) * 2],
    "large-int-tensor-in-fp32": lambda: [torch.tensor(2**25 + 1), torch.tensor(0.5)],
}


def _bits(tensor):
    return tensor.view(_BITS[tensor.dtype]) if tensor.dtype in _BITS else tensor


@pytest.mark.parametrize("name", list(CASES))
def test_the_values_have_the_dtype_and_bits_torch_tensor_gave(name):
    values = CASES[name]()
    old = torch.tensor(values, device="cpu")
    new = _metric_values(values, torch.device("cpu"))
    assert new.dtype == old.dtype and new.shape == old.shape
    assert not new.requires_grad
    assert torch.equal(_bits(new), _bits(old))


def test_device_scalars_are_not_read_on_the_host():
    """Meta tensors hold no data: any host read (item, a copy to the host) raises."""
    values = [torch.empty((), dtype=torch.int64, device="meta"), torch.empty((), device="meta"), 3]
    with pytest.raises(RuntimeError):
        values[1].item()  # positive control: a host read of a meta tensor raises
    out = _metric_values(values, torch.device("meta"))
    assert out.device.type == "meta" and out.dtype == torch.float32 and out.shape == (3,)


@pytest.mark.parametrize(
    "values", [[torch.tensor(2), torch.tensor([1.0])], [torch.tensor([2.0]), torch.tensor([1.0])]]
)
def test_a_non_scalar_metric_keeps_the_old_behaviour(values):
    try:
        old = torch.tensor(values)
    except Exception as error:  # noqa: BLE001 -- whatever torch.tensor raises, _metric_values raises too
        with pytest.raises(type(error)):
            _metric_values(values, torch.device("cpu"))
        return
    new = _metric_values(values, torch.device("cpu"))
    assert new.dtype == old.dtype and torch.equal(new, old)
