"""The QSA indexer's RoPE table when MCore hands mRoPE layers raw freqs (rope fusion on).

With rope fusion, MCore's GPT model gives attention the raw per-axis mRoPE freqs ``[3, 1, T, rotary_dim / 2]``
for its fused kernel. The indexer applies RoPE itself from a materialized ``[T, 1, 1, rotary_dim]`` table, so it
converts the raw freqs with MCore's own ``mrope_freqs_to_rotary_emb``. Qwen3.8 settings: head_dim 256, partial
rotary 0.25 (rotary_dim 64), theta 1e7, interleaved mRoPE [11, 11, 10].
"""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import apply_indexer_rope, indexer_rope_table
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, packed_rope_positions

rope = pytest.importorskip("megatron.core.models.common.embeddings.rotary_pos_embedding")
pytest.importorskip("megatron.core.fusions.fused_mrope")

CONFIG = SimpleNamespace(mrope_section=[11, 11, 10], mrope_interleaved=True, rotary_interleaved=False)
IDS = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003, "merge_size": 2}


@pytest.fixture
def mrope(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    return rope.MultimodalRotaryEmbedding(256, 0.25, rotary_base=10_000_000, interleaved_mrope=True, cp_group=None)


def _packed_positions():
    """[text 7, image (grid 1x4x4, 4 pads), text 3] as THD rows: text restarts, image has native mRoPE."""
    image = torch.tensor([100, 101, 9002, 9001, 9001, 9001, 9001, 9003, 102])
    grid = torch.tensor([[1, 4, 4]])
    ids = torch.cat((torch.arange(200, 207), image, torch.arange(300, 303)))
    bounds = [0, 7, 16, 19]
    positions = torch.cat(
        (
            torch.arange(7)[:, None].expand(-1, 3),
            image_rope_positions(image, grid, **IDS),
            torch.arange(3)[:, None].expand(-1, 3),
        )
    )
    return packed_rope_positions(ids, bounds, positions, grid, rows=3, **IDS)


@pytest.mark.parametrize("kind", ["text", "packed-with-image"])
def test_raw_freqs_become_the_materialized_table(mrope, kind):
    positions = torch.arange(16)[:, None].expand(-1, 3) if kind == "text" else _packed_positions()
    position_ids = positions.T.unsqueeze(1).contiguous()  # [3, 1, T], what the vision provider passes
    materialized = mrope(position_ids, CONFIG.mrope_section, packed_seq=True)
    raw = mrope(position_ids, CONFIG.mrope_section, return_raw_freqs=True, packed_seq=True)
    assert raw.shape == (3, 1, positions.shape[0], 32)
    table = indexer_rope_table(raw, CONFIG)
    assert table.shape == materialized.shape == (positions.shape[0], 1, 1, 64)
    assert torch.equal(table, materialized)


def test_the_indexer_rotates_the_same_with_raw_or_materialized_input(mrope):
    position_ids = _packed_positions().T.unsqueeze(1).contiguous()
    materialized = mrope(position_ids, CONFIG.mrope_section, packed_seq=True)
    raw = mrope(position_ids, CONFIG.mrope_section, return_raw_freqs=True, packed_seq=True)
    torch.manual_seed(0)
    x = torch.randn(position_ids.shape[-1], 2, 128, dtype=torch.bfloat16)  # 64 rotary dims + 64 left as they are
    rows = torch.arange(position_ids.shape[-1])  # the indexer reads a per-token table by pack index
    a = apply_indexer_rope(x, indexer_rope_table(raw, CONFIG), rows)
    b = apply_indexer_rope(x, materialized, rows)
    assert torch.equal(a, b)
    assert torch.equal(a[..., 64:], x[..., 64:])  # dims past rotary_dim are untouched


@pytest.mark.parametrize("length", [1, 3, 5])
def test_a_materialized_table_passes_through_unchanged(mrope, length):
    """Including a 3-token table, [3, 1, 1, 64], which has the leading shape of raw freqs."""
    table = mrope(torch.arange(length)[None, None].expand(3, 1, -1), CONFIG.mrope_section)
    assert table.shape == (length, 1, 1, 64)
    assert indexer_rope_table(table, CONFIG) is table


@pytest.mark.parametrize("length", [1, 3])
def test_raw_freqs_of_one_and_three_tokens_are_converted(mrope, length):
    position_ids = torch.arange(length)[None, None].expand(3, 1, -1)
    raw = mrope(position_ids, CONFIG.mrope_section, return_raw_freqs=True)
    assert torch.equal(indexer_rope_table(raw, CONFIG), mrope(position_ids, CONFIG.mrope_section))


def test_raw_freqs_with_another_section_are_refused(mrope):
    raw = mrope(torch.arange(5)[None, None].expand(3, 1, -1), CONFIG.mrope_section, return_raw_freqs=True)
    with pytest.raises(ValueError, match="needs a \\[T, 1, 1, rotary_dim\\] RoPE table"):
        indexer_rope_table(raw, SimpleNamespace(mrope_section=[8, 8, 8], mrope_interleaved=True))
