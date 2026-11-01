"""THD packing for the Qwen3.8 vision provider: several rows in one microbatch.

miles packs a microbatch's rows back to back (cu_seqlens, CP1) and appends one pad segment of token 0. Each row
must get the positions it would get alone (text: the token index on every axis, from 0; image: native mRoPE), and
the image embeddings must land on that row's pad tokens.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import _CONFIG, _forbid_vision_forward, _model

from miles_plugins.models.qwen3_8_next.vision_inputs import (
    image_rope_positions,
    packed_rope_positions,
    text_rope_positions,
)

IDS = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003, "merge_size": 2}


def _text(length, first=100):
    return torch.arange(first, first + length)


def _image_row(grid=(1, 4, 4), before=2, after=2):
    count = grid[0] * grid[1] * grid[2] // 4
    ids = torch.cat(
        (_text(before), torch.tensor([9002]), torch.full((count,), 9001), torch.tensor([9003]), _text(after, 200))
    )
    g = torch.tensor([grid])
    return ids, g, image_rope_positions(ids, g, **IDS)


def _pack(rows, pad=0):
    ids = torch.cat([r for r in rows] + ([torch.zeros(pad, dtype=torch.long)] if pad else []))
    bounds = [0]
    for r in rows:
        bounds.append(bounds[-1] + len(r))
    if pad:
        bounds.append(bounds[-1] + pad)
    return ids, bounds


# ── packed_rope_positions ────────────────────────────────────────────────────────────────────────────────


def test_each_row_restarts_and_keeps_its_image_positions_and_the_pad_gets_zero():
    text_a, (image, grid, image_pos), text_b = _text(5), _image_row(), _text(3, 300)
    ids, bounds = _pack([text_a, image, text_b], pad=4)
    positions = torch.cat((text_rope_positions(5), image_pos, text_rope_positions(3)))
    out = packed_rope_positions(ids, bounds, positions, grid, **IDS)
    assert torch.equal(out[:5], text_rope_positions(5))
    assert torch.equal(out[5 : 5 + len(image)], image_pos)
    assert torch.equal(out[5 + len(image) : bounds[3]], text_rope_positions(3))
    assert torch.equal(out[bounds[3] :], torch.zeros(4, 3, dtype=torch.long))


def test_without_positions_text_rows_restart_at_zero():
    ids, bounds = _pack([_text(4), _text(1, 50), _text(3, 60)], pad=2)
    out = packed_rope_positions(ids, bounds, None, None, **IDS)
    for start, end in zip(bounds, bounds[1:]):
        assert torch.equal(out[start:end], text_rope_positions(end - start))


@pytest.mark.parametrize("lengths", [[1], [3], [1, 3, 1]])
def test_rows_of_one_and_three_tokens(lengths):
    rows = [_text(n, 10 * i + 100) for i, n in enumerate(lengths)]
    ids, bounds = _pack(rows)
    positions = torch.cat([text_rope_positions(n) for n in lengths])
    assert torch.equal(packed_rope_positions(ids, bounds, positions, None, **IDS), positions)


def test_images_in_two_rows_consume_the_grids_in_order():
    (a, ga, pa), (b, gb, pb) = _image_row((1, 4, 4)), _image_row((1, 2, 4), before=3)
    ids, bounds = _pack([a, _text(2), b])
    positions = torch.cat((pa, text_rope_positions(2), pb))
    out = packed_rope_positions(ids, bounds, positions, torch.cat((ga, gb)), **IDS)
    assert torch.equal(out, positions)
    # grids in the wrong order no longer match the rows' pad runs
    with pytest.raises(ValueError, match="invalid|differ"):
        packed_rope_positions(ids, bounds, positions, torch.cat((gb, ga)), **IDS)


def test_a_real_row_without_positions_is_refused():
    """The last row came without positions: it is not miles' pad (not all token 0), so the pack is refused."""
    image, grid, image_pos = _image_row()
    ids, bounds = _pack([image, _text(3)])
    with pytest.raises(ValueError, match="Rows without mRoPE positions"):
        packed_rope_positions(ids, bounds, image_pos, grid, **IDS)


def test_positions_must_end_at_a_row_boundary():
    ids, bounds = _pack([_text(4), _text(4, 50)])
    with pytest.raises(ValueError, match="row boundary"):
        packed_rope_positions(ids, bounds, text_rope_positions(6), None, **IDS)


def test_positions_of_another_row_are_refused():
    """Same lengths, swapped: a text row given image positions, an image row given text positions."""
    image, grid, image_pos = _image_row()
    text = _text(len(image))
    ids, bounds = _pack([text, image])
    with pytest.raises(ValueError, match="differ"):
        packed_rope_positions(ids, bounds, torch.cat((image_pos, text_rope_positions(len(image)))), grid, **IDS)


def test_unused_grids_and_image_tokens_without_positions_are_refused():
    image, grid, image_pos = _image_row()
    ids, bounds = _pack([image])
    with pytest.raises(ValueError, match="2 image grids for 1 image spans"):
        packed_rope_positions(ids, bounds, image_pos, torch.cat((grid, grid)), **IDS)
    with pytest.raises(ValueError, match="without mRoPE positions"):
        packed_rope_positions(ids, bounds, None, None, **IDS)


def test_an_image_cut_by_a_row_boundary_is_refused():
    image, grid, _ = _image_row()
    cut = 5  # inside the pad run
    ids, bounds = image, [0, cut, len(image)]
    positions = torch.cat((text_rope_positions(cut), text_rope_positions(len(image) - cut)))
    with pytest.raises(ValueError):
        packed_rope_positions(ids, bounds, positions, grid, **IDS)


# ── the model wrapper ───────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


def _packed(bounds):
    return SimpleNamespace(cu_seqlens_host=tuple(bounds))


def _row_inputs(ids, grid, positions, pixels):
    return {"pixel_values": pixels, "image_grid_thw": grid, "qwen38_position_ids": positions}


def test_a_packed_microbatch_equals_its_rows_run_alone(tp_rank_zero):
    """position_ids and the decoder input (image embeddings on the image row's pads) of [text, image, text] + pad
    equal those of each row alone."""
    model = _model(scope="full", data="mixed")
    text_a, (image, grid, image_pos), text_b = _text(5), _image_row(), _text(3, 300)
    pixels = torch.randn(16, 24)
    ids, bounds = _pack([text_a, image, text_b], pad=4)
    positions = torch.cat((text_rope_positions(5), image_pos, text_rope_positions(3)))

    model(input_ids=ids[None], packed_seq_params=_packed(bounds), **_row_inputs(ids, grid, positions, pixels))
    packed = model.calls[-1]

    model(input_ids=image[None], **_row_inputs(image, grid, image_pos, pixels))
    alone = model.calls[-1]
    model(input_ids=text_a[None])
    alone_text = model.calls[-1]

    assert torch.equal(packed["position_ids"][:, 0].T[: bounds[3]], positions)
    assert torch.equal(packed["position_ids"][:, 0].T[bounds[3] :], torch.zeros(4, 3, dtype=torch.long))
    rows = slice(bounds[1], bounds[2])
    assert torch.equal(packed["decoder_input"][rows], alone["decoder_input"])
    assert torch.equal(packed["position_ids"][:, 0, :5], alone_text["position_ids"][:, 0])
    # text rows of the pack get their plain token embeddings
    embedded = model.embedding(input_ids=ids[None])
    assert torch.equal(packed["decoder_input"][: bounds[1]], embedded[: bounds[1]])
    assert torch.equal(packed["decoder_input"][bounds[2] :], embedded[bounds[2] :])


def test_a_packed_text_microbatch_skips_the_tower_and_still_reports_zero_grads(monkeypatch):
    model = _model(scope="full", data="mixed")
    _forbid_vision_forward(model, monkeypatch)
    ids, bounds = _pack([_text(4), _text(6, 50)], pad=2)
    positions = torch.cat((text_rope_positions(4), text_rope_positions(6)))
    model(input_ids=ids[None], packed_seq_params=_packed(bounds), qwen38_position_ids=positions).backward()
    assert model.calls[-1]["decoder_input"] is None
    assert torch.equal(model.calls[-1]["position_ids"][:, 0].T[:10], positions)
    assert all(p.grad is not None and not p.grad.any() for p in model.vision_model.parameters())


def test_image_tokens_and_patches_must_come_together(tp_rank_zero):
    model = _model(scope="full", data="mixed")
    image, grid, image_pos = _image_row()
    with pytest.raises(ValueError, match="must come together"):
        model(input_ids=image[None], qwen38_position_ids=image_pos)
    with pytest.raises(ValueError, match="must come together"):
        model(input_ids=_text(6)[None], **_row_inputs(None, grid, text_rope_positions(6), torch.randn(16, 24)))


def test_image_mode_refuses_a_packed_text_microbatch():
    model = _model(scope="full", data="image")
    ids, bounds = _pack([_text(4), _text(3, 50)])
    with pytest.raises(ValueError, match="requires genuine image patches"):
        model(input_ids=ids[None], packed_seq_params=_packed(bounds))


def test_cu_seqlens_must_cover_the_microbatch():
    model = _model(scope="none", data="mixed")
    ids, bounds = _pack([_text(4), _text(3, 50)])
    with pytest.raises(ValueError, match="CP must be 1"):
        model(input_ids=ids[None], packed_seq_params=_packed([0, 4]))
    assert _CONFIG["image_token_id"] == IDS["image_token_id"]
