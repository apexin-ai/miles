"""Which segments of a packed microbatch are rows (CRV-fix-vl-packing24).

1. The messages source (vision_sft) gives a text row its text positions, as the pre-tokenized source does, so a
   packed microbatch mixing its text and image rows works in either order.
2. Rows and miles' trailing pad are told apart by the real row count get_batch records (sample_count), not by
   token values: a real last row of token 0 is a row, and a row without positions is refused.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import (
    _CONFIG,
    _TEXT_MESSAGES,
    _model,
    _Processor,
)

from miles_plugins.models.qwen3_8_next import vision_sft
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, text_rope_positions

IDS = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003, "merge_size": 2}


@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


def _image_row():
    ids = torch.tensor([100, 101, 9002, 9001, 9001, 9001, 9001, 9003, 102, 103])
    grid = torch.tensor([[1, 4, 4]])
    positions = image_rope_positions(ids, grid, **IDS)
    return ids, {"pixel_values": torch.randn(16, 24), "image_grid_thw": grid, "qwen38_position_ids": positions}


def _concat(rows):
    """What miles' get_batch does with the rows' multimodal_train_inputs: concatenate each key in row order,
    skipping rows that have none."""
    present = [inputs for _, inputs in rows if inputs is not None]
    keys = {key for inputs in present for key in inputs}
    return {key: torch.cat([inputs[key] for inputs in present if key in inputs]) for key in keys}


def _pack(rows, pad=0):
    ids = torch.cat([row_ids for row_ids, _ in rows] + ([torch.zeros(pad, dtype=torch.long)] if pad else []))
    bounds = [0]
    for row_ids, _ in rows:
        bounds.append(bounds[-1] + len(row_ids))
    if pad:
        bounds.append(bounds[-1] + pad)
    return ids, SimpleNamespace(cu_seqlens_host=tuple(bounds), sample_count=len(rows))


def test_the_messages_source_gives_a_text_row_its_text_positions():
    tokens, mask, inputs = vision_sft.encode_sft_sample(_Processor(), _CONFIG, _TEXT_MESSAGES, [], allow_text=True)
    assert set(inputs) == {"qwen38_position_ids"}
    assert torch.equal(inputs["qwen38_position_ids"], text_rope_positions(len(tokens)))


@pytest.mark.parametrize("order", ["text-image", "image-text"])
def test_a_packed_microbatch_of_messages_source_rows_works_in_either_order(tp_rank_zero, order):
    tokens, _, text_inputs = vision_sft.encode_sft_sample(_Processor(), _CONFIG, _TEXT_MESSAGES, [], allow_text=True)
    text = (torch.tensor(tokens), text_inputs)
    image = _image_row()
    rows = [text, image] if order == "text-image" else [image, text]
    ids, packed = _pack(rows, pad=3)
    model = _model(scope="full", data="mixed")
    model(input_ids=ids[None], packed_seq_params=packed, **_concat(rows))
    got = model.calls[-1]["position_ids"][:, 0].T
    start = 0
    for row_ids, inputs in rows:
        assert torch.equal(got[start : start + len(row_ids)], inputs["qwen38_position_ids"])
        start += len(row_ids)
    assert torch.equal(got[start:], torch.zeros(3, 3, dtype=torch.long))


def test_a_real_last_row_of_token_zero_without_positions_is_refused(tp_rank_zero):
    """CX1's case: image row 10 tokens + a real text row of 118 x token 0, cu [0, 10, 128], no pad. Only the image
    row carries positions; the text row must not be taken for pad."""
    image = _image_row()
    zeros = (torch.zeros(118, dtype=torch.long), None)
    ids, packed = _pack([image, zeros])
    model = _model(scope="full", data="mixed")
    with pytest.raises(ValueError, match="Rows without mRoPE positions"):
        model(input_ids=ids[None], packed_seq_params=packed, **_concat([image, zeros]))


def test_a_real_last_row_of_token_zero_with_positions_gets_them(tp_rank_zero):
    image = _image_row()
    zeros = (torch.zeros(118, dtype=torch.long), {"qwen38_position_ids": text_rope_positions(118).clone()})
    ids, packed = _pack([image, zeros])
    model = _model(scope="full", data="mixed")
    model(input_ids=ids[None], packed_seq_params=packed, **_concat([image, zeros]))
    got = model.calls[-1]["position_ids"][:, 0].T
    assert torch.equal(got[10:], text_rope_positions(118))


def test_the_trailing_pad_is_known_from_the_row_count(tp_rank_zero):
    """Same tokens, two readings: with sample_count 1 the zero segment is pad (gets 0); with 2 it is a row."""
    image = _image_row()
    ids, packed = _pack([image], pad=118)
    model = _model(scope="full", data="mixed")
    model(input_ids=ids[None], packed_seq_params=packed, **_concat([image]))
    assert torch.equal(model.calls[-1]["position_ids"][:, 0].T[10:], torch.zeros(118, 3, dtype=torch.long))


def test_a_pack_without_a_row_count_is_refused(tp_rank_zero):
    image = _image_row()
    ids, packed = _pack([image], pad=6)
    model = _model(scope="full", data="mixed")
    with pytest.raises(ValueError, match="sample_count"):
        model(
            input_ids=ids[None],
            packed_seq_params=SimpleNamespace(cu_seqlens_host=packed.cu_seqlens_host),
            **_concat([image])
        )
