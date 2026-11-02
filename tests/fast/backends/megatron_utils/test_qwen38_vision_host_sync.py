"""The Qwen3.8 vision provider reads the device once per microbatch.

It copies the token ids, the mRoPE positions and the image grids to the host in one copy and runs every check and
span search there; the native tower gets its grid-derived inputs from the host grid (transformers' vision_utils
precompute keys), so it reads nothing back either, not even sdpa's per-block ``lengths.tolist()``. The outputs are
those of the code as of 8325e8bd (kept below), and the checks refuse the same inputs with the same errors.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import _CONFIG, _Language
from transformers import vision_utils
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from miles_plugins.models.qwen3_8_next import vision
from miles_plugins.models.qwen3_8_next.vision_inputs import (
    image_rope_positions,
    image_span_starts,
    packed_rope_positions,
    scatter_image_embeddings,
    text_rope_positions,
)

IDS = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003, "merge_size": 2}


# ---- the code as of 8325e8bd ----
def _old_packed_rope_positions(
    token_ids,
    boundaries,
    positions,
    grids,
    *,
    rows,
    image_token_id,
    vision_start_token_id,
    vision_end_token_id,
    merge_size,
):
    ids = torch.as_tensor(token_ids)
    total = ids.shape[0]
    bounds = [int(b) for b in boundaries]
    if ids.ndim != 1 or not bounds or bounds[0] != 0 or bounds[-1] != total:
        raise ValueError(f"cu_seqlens {bounds[:4]}... do not describe a packed sequence of {total} tokens")
    if any(b <= a for a, b in zip(bounds, bounds[1:], strict=False)):
        raise ValueError("cu_seqlens must be strictly increasing")
    segments = len(bounds) - 1
    if not 1 <= rows <= segments or segments - rows > 1:
        raise ValueError(f"{rows} rows do not fit cu_seqlens of {segments} segments (rows, then at most one pad)")
    end_of_rows = bounds[rows]
    out = torch.zeros((total, 3), dtype=torch.long, device=ids.device)
    if positions is None:
        if image_span_starts(ids[:end_of_rows], image_token_id):
            raise ValueError("Image tokens reached the model without mRoPE positions")
        for start, end in zip(bounds[:rows], bounds[1 : rows + 1], strict=True):
            out[start:end] = text_rope_positions(end - start, device=ids.device)
        return out
    positions = torch.as_tensor(positions).to(device=ids.device, dtype=torch.long)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"mRoPE positions must be [tokens, 3], got {tuple(positions.shape)}")
    if positions.shape[0] != end_of_rows:
        raise ValueError(
            f"Rows without mRoPE positions in a microbatch that has them: positions cover {positions.shape[0]} "
            f"of the {end_of_rows} tokens of its {rows} rows (every row must carry them)"
        )
    grids = torch.zeros((0, 3), dtype=torch.long) if grids is None else torch.as_tensor(grids)
    used = 0
    for start, end in zip(bounds[:rows], bounds[1 : rows + 1], strict=True):
        row = ids[start:end]
        count = len(image_span_starts(row, image_token_id))
        if count:
            expected = image_rope_positions(
                row,
                grids[used : used + count],
                image_token_id=image_token_id,
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
                merge_size=merge_size,
            )
        else:
            expected = text_rope_positions(end - start, device=ids.device)
        used += count
        if not torch.equal(positions[start:end], expected.to(ids.device)):
            raise ValueError(f"mRoPE positions of the row at [{start}, {end}) differ from its tokens and images")
        out[start:end] = positions[start:end]
    if used != grids.shape[0]:
        raise ValueError(f"{grids.shape[0]} image grids for {used} image spans in the microbatch")
    if image_span_starts(ids[end_of_rows:], image_token_id):
        raise ValueError("Image tokens in the trailing pad segment")
    return out


def _old_scatter_image_embeddings(
    decoder_input, token_positions, image_embeddings, *, tp_rank=0, sequence_parallel=False
):
    if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
        raise ValueError("Qwen image SFT requires micro-batch-size 1")
    if token_positions.numel() != image_embeddings.shape[0]:
        raise ValueError("Image embeddings and expanded placeholder counts differ")
    local = token_positions.to(decoder_input.device)
    if sequence_parallel:
        local = local - tp_rank * decoder_input.shape[0]
    selected = (local >= 0) & (local < decoder_input.shape[0])
    if not sequence_parallel and not selected.all():
        raise ValueError("An image token lies outside the decoder input")
    output = decoder_input.clone()
    output[local[selected], 0] = image_embeddings[selected].to(output.dtype)
    return output + image_embeddings.sum().to(output.dtype) * 0


# ---- inputs ----
def _image_row(prefix=2, grid=(1, 4, 4), suffix=2, base=100):
    tokens = grid[1] * grid[2] // 4
    ids = torch.tensor(
        list(range(base, base + prefix))
        + [9002]
        + [9001] * tokens
        + [9003]
        + list(range(base + 50, base + 50 + suffix))
    )
    grids = torch.tensor([grid])
    return ids, grids, image_rope_positions(ids, grids, **IDS)


def _text_row(n, base=200):
    return torch.arange(base, base + n), None, text_rope_positions(n).clone()


def _two_image_row():
    ids = torch.cat([_image_row()[0], _image_row(grid=(1, 2, 4), base=300)[0]])
    grids = torch.tensor([[1, 4, 4], [1, 2, 4]])
    return ids, grids, image_rope_positions(ids, grids, **IDS)


def _pack(rows, pad=0, with_positions=True):
    ids = torch.cat([r[0] for r in rows] + ([torch.zeros(pad, dtype=torch.long)] if pad else []))
    bounds = [0]
    for r in rows:
        bounds.append(bounds[-1] + len(r[0]))
    if pad:
        bounds.append(bounds[-1] + pad)
    grids = [r[1] for r in rows if r[1] is not None]
    grids = torch.cat(grids) if grids else None
    positions = torch.cat([r[2] for r in rows]) if with_positions else None
    return ids, tuple(bounds), positions, grids, len(rows)


def _packs():
    return {
        "image": _pack([_image_row()]),
        "image-text-pad": _pack([_image_row(), _text_row(5)], pad=3),
        "text-two-images-text": _pack(
            [_text_row(4), _image_row(grid=(1, 4, 6)), _image_row(grid=(1, 2, 2)), _text_row(1)]
        ),
        "two-images-in-a-row": _pack([_two_image_row()], pad=2),
        "text-only": _pack([_text_row(6), _text_row(3)], pad=2),
        "text-only-no-positions": _pack([_text_row(6), _text_row(3)], pad=2, with_positions=False),
    }


def _bad_inputs():
    """(name, ids, bounds, positions, grids, rows) cases the checks refuse."""
    ids, bounds, positions, grids, _ = _pack([_image_row(), _text_row(5)], pad=3)
    wrong = positions.clone()
    wrong[3, 1] += 1
    padded_image = torch.cat([ids[:-3], torch.tensor([9001, 0, 0])])
    return {
        "bounds-short": (ids, bounds[:-1], positions, grids, 2),
        "bounds-not-increasing": (ids, (0, 10, 10, 15, 18), positions, grids, 2),
        "too-many-rows": (ids, bounds, positions, grids, 4),
        "two-pads": (ids, bounds, positions, grids, 1),
        "wrong-position": (ids, bounds, wrong, grids, 2),
        "positions-short": (ids, bounds, positions[:-1], grids, 2),
        "positions-2d-wrong": (ids, bounds, positions[:, :2], grids, 2),
        "extra-grid": (ids, bounds, positions, torch.cat([grids, grids]), 2),
        "missing-grid": (ids, bounds, positions, None, 2),
        "odd-grid": (ids, bounds, positions, torch.tensor([[1, 4, 5]]), 2),
        "image-in-pad": (padded_image, bounds, positions, grids, 2),
        "image-without-positions": (ids, bounds, None, grids, 2),
    }


# ---- vision_inputs: the same outputs and errors on host copies ----
@pytest.mark.parametrize("name", list(_packs()))
def test_packed_positions_are_unchanged(name):
    ids, bounds, positions, grids, rows = _packs()[name]
    old = _old_packed_rope_positions(ids, bounds, positions, grids, rows=rows, **IDS)
    new = packed_rope_positions(ids, bounds, positions, grids, rows=rows, **IDS)
    assert new.dtype == old.dtype and torch.equal(new, old)
    if positions is not None:  # the result built from the device copy of the positions
        built = packed_rope_positions(
            ids, bounds, positions, grids, rows=rows, **IDS, device="cpu", device_positions=positions.clone()
        )
        assert torch.equal(built, old)


@pytest.mark.parametrize("name", list(_bad_inputs()))
def test_packed_positions_refuse_the_same_inputs_with_the_same_errors(name):
    ids, bounds, positions, grids, rows = _bad_inputs()[name]
    with pytest.raises(ValueError) as old:
        _old_packed_rope_positions(ids, bounds, positions, grids, rows=rows, **IDS)
    with pytest.raises(ValueError) as new:
        packed_rope_positions(ids, bounds, positions, grids, rows=rows, **IDS)
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("sequence_parallel,tp_rank", [(False, 0), (True, 0), (True, 1), (True, 2)])
def test_scatter_is_unchanged_with_host_positions(sequence_parallel, tp_rank):
    torch.manual_seed(0)
    decoder = torch.randn(8, 1, 4, requires_grad=True)
    positions = torch.tensor([1, 2, 9, 10, 17]) if sequence_parallel else torch.tensor([1, 2, 5, 6, 7])
    embeddings = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    kwargs = {"tp_rank": tp_rank, "sequence_parallel": sequence_parallel}
    old = _old_scatter_image_embeddings(decoder, positions, embeddings, **kwargs)
    old_grads = torch.autograd.grad(old.square().sum(), (decoder, embeddings))
    new = scatter_image_embeddings(decoder, positions, embeddings, **kwargs)
    new_grads = torch.autograd.grad(new.square().sum(), (decoder, embeddings))
    assert torch.equal(new, old)
    for got, want in zip(new_grads, old_grads, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "decoder,positions,embeddings",
    [
        (torch.zeros(8, 2, 4), torch.tensor([1]), torch.zeros(1, 4)),
        (torch.zeros(8, 1, 4), torch.tensor([1, 2]), torch.zeros(1, 4)),
        (torch.zeros(8, 1, 4), torch.tensor([1, 8]), torch.zeros(2, 4)),
        (torch.zeros(8, 1, 4), torch.tensor([-1, 2]), torch.zeros(2, 4)),
    ],
)
def test_scatter_refuses_the_same_inputs_with_the_same_errors(decoder, positions, embeddings):
    with pytest.raises(ValueError) as old:
        _old_scatter_image_embeddings(decoder, positions, embeddings)
    with pytest.raises(ValueError) as new:
        scatter_image_embeddings(decoder, positions, embeddings)
    assert str(new.value) == str(old.value)


# ---- the native tower: precomputed grid inputs ----
def _tower(*, recompute=False, device="cpu"):
    torch.manual_seed(0)
    config = Qwen3VLVisionConfig(**_CONFIG["vision_config"])
    config._attn_implementation = "sdpa"
    with torch.device(device):
        tower = Qwen3VLVisionModel(config).to(torch.float64)
    if recompute:
        tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tower


GRIDS = torch.tensor([[1, 4, 4], [1, 6, 2], [1, 2, 8], [1, 2, 2]])


def test_the_grid_inputs_are_what_the_tower_computes():
    tower = _tower()
    got = vision._grid_inputs(tower, GRIDS, torch.device("cpu"))
    indices, weights = vision_utils.get_vision_interpolation_indices_and_weights(
        GRIDS,
        num_grid_per_side=tower.num_grid_per_side,
        mode=tower.interpolation_mode,
        align_corners=tower.interpolation_align_corners,
        spatial_merge_size=tower.config.spatial_merge_size,
    )
    want = {
        "interp_indices": indices,
        "interp_weights": weights,
        "position_ids": vision_utils.get_vision_position_ids(GRIDS, tower.spatial_merge_size),
        "cu_seqlens": vision_utils.get_vision_cu_seqlens(GRIDS),
    }
    assert set(got) == set(want)
    for key, value in want.items():
        assert got[key].dtype == value.dtype and torch.equal(got[key], value), key
    assert got["cu_seqlens"].device.type == "cpu"


@pytest.mark.parametrize("recompute", [False, True])
def test_the_tower_gives_the_same_outputs_and_grads_with_the_grid_inputs(recompute):
    pixels = torch.randn(int(GRIDS.prod(-1).sum()), 24, dtype=torch.float64)
    results = []
    for precompute in (False, True):
        tower = _tower(recompute=recompute)
        tower.train()
        extra = vision._grid_inputs(tower, GRIDS, torch.device("cpu")) if precompute else {}
        out = tower(pixels, grid_thw=GRIDS, return_dict=True, **extra).pooler_output
        out.square().sum().backward()
        results.append((out.detach(), {n: p.grad for n, p in tower.named_parameters()}))
    (old, old_grads), (new, new_grads) = results
    assert torch.equal(new, old)
    for name, grad in old_grads.items():
        assert torch.equal(new_grads[name], grad), name


def test_the_tower_reads_nothing_back_with_the_grid_inputs():
    """On meta the tower runs only if nothing reads a device value: the grid is given only as a meta tensor."""
    tower = _tower(device="meta")
    grid = torch.empty(GRIDS.shape, dtype=GRIDS.dtype, device="meta")
    pixels = torch.empty(int(GRIDS.prod(-1).sum()), 24, dtype=torch.float64, device="meta")
    with pytest.raises((NotImplementedError, RuntimeError)):
        tower(pixels, grid_thw=grid, return_dict=True)  # positive control: without them it reads the grid
    out = tower(pixels, grid_thw=grid, return_dict=True, **vision._grid_inputs(tower, GRIDS, torch.device("meta")))
    assert out.pooler_output.shape == (int(GRIDS.prod(-1).sum()) // 4, _CONFIG["vision_config"]["out_hidden_size"])


# ---- the provider forward ----
@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


def _wired(device="cpu", scope="full"):
    torch.manual_seed(0)
    language = _Language()
    language.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
    language.to(device)
    vision.wire_visual(language, _CONFIG, scope=scope, recompute=False, freeze_language=False, data="mixed")
    return language


def _forward_inputs(name):
    ids, bounds, positions, grids, rows = _packs()[name]
    packed = SimpleNamespace(cu_seqlens_host=bounds, sample_count=rows)
    inputs = {} if positions is None else {"qwen38_position_ids": positions}
    if grids is not None:
        inputs |= {"pixel_values": torch.randn(int(grids.prod(-1).sum()), 24), "image_grid_thw": grids}
    return ids, packed, inputs


@pytest.mark.parametrize("name", ["image", "image-text-pad", "text-two-images-text", "text-only"])
def test_the_forward_gives_what_the_old_code_gave(tp_rank_zero, name):
    ids, packed, inputs = _forward_inputs(name)
    model = _wired()
    model(input_ids=ids[None], packed_seq_params=packed, **inputs)
    call = model.calls[-1]
    rows = packed.sample_count
    old_positions = _old_packed_rope_positions(
        ids, packed.cu_seqlens_host, inputs.get("qwen38_position_ids"), inputs.get("image_grid_thw"), rows=rows, **IDS
    )
    assert torch.equal(call["position_ids"], old_positions.T.unsqueeze(1).contiguous())
    if "pixel_values" not in inputs:
        assert call["decoder_input"] is None
        return
    pooled = model.vision_model(
        inputs["pixel_values"], grid_thw=inputs["image_grid_thw"], return_dict=True
    ).pooler_output
    old_decoder = _old_scatter_image_embeddings(
        model.embedding(input_ids=ids[None]), (ids == 9001).nonzero().flatten(), pooled
    )
    assert torch.equal(call["decoder_input"], old_decoder)


@pytest.mark.parametrize(
    "name", ["image", "image-text-pad", "text-two-images-text", "text-only", "text-only-no-positions"]
)
def test_the_forward_reads_the_device_once(tp_rank_zero, monkeypatch, name):
    """Everything on meta (no data): the forward runs only if it reads the device through _host_copies alone."""
    ids, packed, inputs = _forward_inputs(name)
    meta = {key: torch.empty(value.shape, dtype=value.dtype, device="meta") for key, value in inputs.items()}
    meta_ids = torch.empty((1, ids.shape[0]), dtype=ids.dtype, device="meta")
    with pytest.raises((NotImplementedError, RuntimeError)):
        meta_ids.cpu()  # positive control: meta holds no data
    copies = []

    def host_copies(token_ids, positions, grids):
        """Stands in for the one device-to-host copy, giving the host values of the meta inputs."""
        copies.append(1)
        assert token_ids.device.type == "meta" and token_ids.shape == ids.shape
        assert positions is meta.get("qwen38_position_ids") and grids is meta.get("image_grid_thw")
        return ids, inputs.get("qwen38_position_ids"), inputs.get("image_grid_thw")

    monkeypatch.setattr(vision, "_host_copies", host_copies)
    model = _wired(device="meta")
    out = model(input_ids=meta_ids, packed_seq_params=packed, **meta)
    assert out.device.type == "meta"
    assert copies == [1]
    call = model.calls[-1]
    assert call["position_ids"].shape == (3, 1, ids.shape[0]) and call["position_ids"].device.type == "meta"
    assert (call["decoder_input"] is None) == ("pixel_values" not in inputs)


def test_host_copies_share_one_copy():
    ids, positions, grids = torch.arange(6), torch.arange(18).reshape(6, 3), torch.tensor([[1, 2, 2]])
    assert all(torch.equal(a, b) for a, b in zip(vision._host_copies(ids, positions, grids), (ids, positions, grids)))
    assert vision._host_copies(ids, None, None)[1:] == (None, None)
