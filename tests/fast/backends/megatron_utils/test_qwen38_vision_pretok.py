"""Image inputs of pre-tokenized Qwen3.8 rows (miles_plugins.models.qwen3_8_next.vision_pretok).

The rules run with a stand-in processor. ``test_real_row_matches_the_image_sft_encoder`` runs the real processor
on a VISION-smoke row when ``Q38_VL_HF`` (a Qwen3.8 HF directory) and ``Q38_VL_ROWS`` (a JSONL of
``{"images", "messages"}`` rows) are set, and compares every tensor with image SFT's own encoder.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from miles_plugins.models.qwen3_8_next import vision_pretok
from miles_plugins.models.qwen3_8_next.vision_inputs import text_rope_positions

CONFIG = {
    "model_type": "qwen4_exp",
    "image_token_id": 9,
    "vision_start_token_id": 8,
    "vision_end_token_id": 10,
    "vision_config": {"spatial_merge_size": 2},
}


@pytest.fixture
def fake(monkeypatch):
    """Grids per image path; image_tensors returns them with dummy patches."""
    grids = {}
    monkeypatch.setattr(vision_pretok, "processor_and_config", lambda checkpoint: (object(), CONFIG))
    monkeypatch.setattr(
        vision_pretok, "load_images", lambda entries, processor: [vision_pretok._entry_path(e)[0] for e in entries]
    )

    def image_tensors(images, processor):
        g = torch.tensor([grids[i] for i in images])
        return torch.zeros(int(g.prod(-1).sum()), 4), g

    monkeypatch.setattr(vision_pretok, "image_tensors", image_tensors)
    return grids


def _row(*runs, text=(1, 2, 3)):
    """ids: text, then per run [start, pad x n, end, 5]; labels: -100 except on the last text token."""
    ids = list(text)
    for n in runs:
        ids += [8] + [9] * n + [10, 5]
    ids += [7]
    labels = [-100] * (len(ids) - 1) + [7]
    return ids, labels


def test_a_text_row_has_no_image_inputs_and_keeps_its_ids(fake):
    ids, labels = [1, 2, 3, 7], [-100, -100, -100, 7]
    before = list(ids)
    assert vision_pretok.train_inputs_for_row(ids, labels, [], "hf") is None
    assert ids == before


def test_an_image_row_gets_patches_grids_and_mrope_positions(fake):
    fake["a.png"] = (1, 4, 4)  # 16 patches / merge^2 = 4 pads
    fake["b.png"] = (1, 2, 4)  # 2 pads
    ids, labels = _row(4, 2)
    before = list(ids)
    inputs = vision_pretok.train_inputs_for_row(ids, labels, ["a.png", "b.png"], "hf")
    assert ids == before
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert inputs["pixel_values"].shape[0] == 16 + 8
    positions = inputs["qwen38_position_ids"]
    assert positions.shape == (len(ids), 3) and positions.dtype == torch.long
    # text before the first image is plain text positions on all three axes
    assert torch.equal(positions[:4], text_rope_positions(4))


@pytest.mark.parametrize(
    ("runs", "images", "message"),
    [
        ((3,), ["a.png"], "pad counts"),  # one pad short of the grid's 4
        ((5,), ["a.png"], "pad counts"),  # one pad too many
        ((4,), ["a.png", "a.png"], "2 images"),  # fewer pad runs than images
        ((4, 4), ["a.png"], "for 1 images"),  # more pad runs than images
        ((4,), [], "without meta images"),  # pads in a row that has no images
    ],
    ids=["short", "long", "missing-run", "extra-run", "pads-without-images"],
)
def test_pad_runs_must_match_the_images(fake, runs, images, message):
    fake["a.png"] = (1, 4, 4)
    ids, labels = _row(*runs)
    with pytest.raises(ValueError, match=message):
        vision_pretok.train_inputs_for_row(ids, labels, images, "hf")


def test_two_images_without_a_boundary_between_their_pads_fail(fake):
    fake["a.png"] = (1, 4, 4)
    ids = [1, 8] + [9] * 8 + [10, 7]
    with pytest.raises(ValueError, match="1 image pad runs for 2 images"):
        vision_pretok.train_inputs_for_row(ids, [-100] * (len(ids) - 1) + [7], ["a.png", "a.png"], "hf")


def test_pads_outside_vision_markers_fail(fake):
    fake["a.png"] = (1, 4, 4)
    ids = [1, 2] + [9] * 4 + [10, 7]
    with pytest.raises(ValueError, match="not enclosed"):
        vision_pretok.train_inputs_for_row(ids, [-100] * (len(ids) - 1) + [7], ["a.png"], "hf")


def test_a_label_on_an_image_pad_fails(fake):
    fake["a.png"] = (1, 4, 4)
    ids, labels = _row(4)
    labels[5] = 9
    with pytest.raises(ValueError, match="must not receive SFT labels"):
        vision_pretok.train_inputs_for_row(ids, labels, ["a.png"], "hf")


def test_an_image_whose_bytes_changed_fails_before_it_is_used(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    image.write_bytes(b"new bytes")
    called = []
    monkeypatch.setattr(vision_pretok, "process_vision_info", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="sha256"):
        vision_pretok.load_images([{"path": str(image), "sha256": hashlib.sha256(b"old bytes").hexdigest()}], None)
    assert not called


def _real_rows():
    hf, rows = os.environ.get("Q38_VL_HF"), os.environ.get("Q38_VL_ROWS")
    if not hf or not rows or not Path(rows).is_file():
        return None, []
    with open(rows) as f:
        return hf, [json.loads(line) for line, _ in zip(f, range(4))]


@pytest.mark.parametrize("index", range(4))
def test_real_row_matches_the_image_sft_encoder(index):
    """Same tokens in, the same pixel_values / grids / mRoPE positions out as encode_sft_sample."""
    hf, rows = _real_rows()
    if hf is None or index >= len(rows):
        pytest.skip("set Q38_VL_HF and Q38_VL_ROWS to run against a real checkpoint and VISION-smoke rows")
    from miles_plugins.models.qwen3_8_next.vision_sft import encode_sft_sample

    row = rows[index]
    processor, config = vision_pretok.processor_and_config(hf)
    entries = [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in row["images"]]
    images = vision_pretok.load_images(entries, processor)
    tokens, mask, reference = encode_sft_sample(processor, config, row["messages"], images)
    labels = [t if m else -100 for t, m in zip(tokens, mask)]
    labels[0] = -100

    inputs = vision_pretok.train_inputs_for_row(tokens, labels, entries, hf)

    assert set(inputs) == set(reference)
    for key in reference:
        assert torch.equal(inputs[key], reference[key]), key
    assert vision_pretok.image_pad_counts(entries, hf) == [
        end - start for start, end in vision_pretok.image_spans(tokens, config)
    ]


@pytest.mark.parametrize("length", [1, 7, 4096])
def test_text_positions_give_the_same_rope_as_plain_text_rope(monkeypatch, length):
    """A text row in mixed mode gets mRoPE with T = H = W = token index (text_rope_positions). With the Qwen3.8
    settings (head_dim 256, partial rotary 0.25, theta 1e7, interleaved mRoPE [11, 11, 10]) its rotary embedding
    equals the 1-D RoPE of the vision-off path element for element."""
    rope = pytest.importorskip("megatron.core.models.common.embeddings.rotary_pos_embedding")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    plain = rope.RotaryEmbedding(256, 0.25, rotary_base=10_000_000, use_cpu_initialization=True, cp_group=None)
    mrope = rope.MultimodalRotaryEmbedding(256, 0.25, rotary_base=10_000_000, interleaved_mrope=True, cp_group=None)
    positions = text_rope_positions(length).T.unsqueeze(1).contiguous()  # what vision.py passes: [3, 1, L]
    expected = plain(length)
    actual = mrope(positions, [11, 11, 10])
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
