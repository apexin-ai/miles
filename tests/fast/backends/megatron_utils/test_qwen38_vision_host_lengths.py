"""CPU contracts for the vision tower's host cu_seqlens hook (one device sync per tower forward, not 3 x depth).

The hook must not change any value: outputs, input gradients and parameter gradients are bit-identical with and
without it, with and without gradient checkpointing. On CPU there is no sync to count, so a spy checks the mechanism:
every attention call (forward and recompute) sees the one host tensor the hook built per tower forward.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.vision_utils import get_vision_attention_seqlens

from miles_plugins.models.qwen3_8_next import vision, vision_host_lengths

_VISION = {
    "depth": 3,
    "hidden_size": 16,
    "intermediate_size": 24,
    "out_hidden_size": 16,
    "num_heads": 2,
    "patch_size": 2,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [],
}
# three entries of different sizes packed (the last one has two frames): 16 + 12 + 16 patches
_GRID = torch.tensor([[1, 4, 4], [1, 2, 6], [2, 4, 2]])


def _tower(dtype, *, attn="sdpa", seed=0):
    config = Qwen3VLVisionConfig(**_VISION)
    config._attn_implementation = attn
    torch.manual_seed(seed)
    return Qwen3VLVisionModel(config).to(dtype=dtype)


def _patches(dtype, seed=1):
    generator = torch.Generator().manual_seed(seed)
    tokens = int((_GRID[:, 0] * _GRID[:, 1] * _GRID[:, 2]).sum())
    width = 3 * _VISION["temporal_patch_size"] * _VISION["patch_size"] ** 2
    return torch.randn(tokens, width, generator=generator).to(dtype)


def _spy_attention(tower):
    seen = []
    for block in tower.blocks:
        block.attn.register_forward_pre_hook(lambda m, a, k: seen.append(k["cu_seqlens"]), with_kwargs=True)
    return seen


def _run(tower, dtype, recompute):
    tower.train()
    if recompute:
        tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pixels = _patches(dtype).requires_grad_(True)
    result = tower(pixels, grid_thw=_GRID, return_dict=True)
    weights = torch.randn(result.pooler_output.shape, generator=torch.Generator().manual_seed(2)).to(dtype)
    (result.pooler_output.float() * weights.float()).sum().backward()
    grads = {name: p.grad for name, p in tower.named_parameters()}
    return result.last_hidden_state, result.pooler_output, pixels.grad, grads


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hook_is_bit_exact_and_attention_sees_one_host_tensor(monkeypatch, dtype, recompute):
    built = []

    def recording(grid_thw):
        built.append(vision_host_lengths.get_vision_cu_seqlens(grid_thw.cpu()))
        return built[-1]

    monkeypatch.setattr(vision_host_lengths, "host_cu_seqlens", recording)
    plain = _tower(dtype)
    hooked = copy.deepcopy(plain)
    assert vision_host_lengths.install_host_cu_seqlens(hooked) is not None
    plain_seen, hooked_seen = _spy_attention(plain), _spy_attention(hooked)

    expected = _run(plain, dtype, recompute)
    assert not built
    got = _run(hooked, dtype, recompute)

    for want, have in zip(expected[:3], got[:3]):
        assert want.dtype == have.dtype and torch.equal(want, have)
    assert expected[3].keys() == got[3].keys()
    for name, want in expected[3].items():
        assert want is not None and torch.equal(want, got[3][name]), name

    calls = _VISION["depth"] * (2 if recompute else 1)  # the recompute runs every block's attention again
    assert len(built) == 1 and len(hooked_seen) == calls and len(plain_seen) == calls
    assert all(seen is built[0] for seen in hooked_seen)
    assert built[0].device.type == "cpu"
    assert not any(seen is built[0] for seen in plain_seen)
    for seen in plain_seen:
        assert seen.dtype == built[0].dtype and torch.equal(seen, built[0])


def test_host_cu_seqlens_matches_transformers():
    reference, _ = get_vision_attention_seqlens(_GRID, Qwen3VLVisionConfig(**_VISION))
    host = vision_host_lengths.host_cu_seqlens(_GRID)
    assert host.device.type == "cpu" and host.dtype == reference.dtype == torch.int32
    assert host.tolist() == reference.tolist() == [0, 16, 28, 36, 44]


def test_one_host_tensor_per_tower_forward(monkeypatch):
    built = []
    monkeypatch.setattr(
        vision_host_lengths,
        "host_cu_seqlens",
        lambda grid: built.append(vision_host_lengths.get_vision_cu_seqlens(grid.cpu())) or built[-1],
    )
    tower = _tower(torch.float32)
    vision_host_lengths.install_host_cu_seqlens(tower)
    seen = _spy_attention(tower)
    with torch.no_grad():
        tower(_patches(torch.float32), _GRID)  # grid_thw passed positionally
        tower(_patches(torch.float32, seed=3), grid_thw=_GRID)
    assert len(built) == 2
    depth = _VISION["depth"]
    assert all(s is built[0] for s in seen[:depth]) and all(s is built[1] for s in seen[depth:])


def test_flash_tower_is_not_hooked():
    flash = torch.nn.Module()
    flash.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    assert vision_host_lengths.install_host_cu_seqlens(flash) is None
    assert not flash._forward_pre_hooks
    # a hooked tower switched to flash later keeps the device cu_seqlens the varlen kernel needs
    kwargs = {"grid_thw": _GRID}
    assert vision_host_lengths._pre_hook(flash, (torch.zeros(1),), kwargs) is None
    assert "cu_seqlens" not in kwargs


def test_explicit_cu_seqlens_is_kept():
    tower = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    given = torch.tensor([0, 44], dtype=torch.int32)
    kwargs = {"grid_thw": _GRID, "cu_seqlens": given}
    assert vision_host_lengths._pre_hook(tower, (torch.zeros(1),), kwargs) is None
    assert kwargs["cu_seqlens"] is given


def test_wire_visual_installs_the_hook():
    class _Language(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre_process = True
            self.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
            self.embedding = SimpleNamespace(word_embeddings=torch.nn.Embedding(4, 16))

    model = _Language()
    config = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003}
    vision.wire_visual(model, config | {"vision_config": _VISION}, scope="full", recompute=True, freeze_language=False)
    hooks = list(model.vision_model._forward_pre_hooks.values())
    assert vision_host_lengths._pre_hook in hooks
