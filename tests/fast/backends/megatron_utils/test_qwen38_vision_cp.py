"""CPU contracts: the Qwen3.8 vision provider under allgather context parallelism equals CP1.

Under ``--allgather-cp`` miles' get_batch concatenates the microbatch's rows, pads the whole stream to a multiple of
``cp_size * pad`` and gives CP rank ``r`` the contiguous slice ``[r * T / cp, (r + 1) * T / cp)``; cu_seqlens, the
image patches, the grids and the rows' mRoPE positions stay whole-microbatch. Every CP rank runs the vision tower on
every image of the microbatch and writes into its decoder input only the image tokens of its own slice, so an image
that straddles the boundary is split between the ranks. The language model gets the whole stream's mRoPE positions:
the QSA indexer scores the gathered sequence by token index, and the attention layer slices its own rows for the
query / key RoPE (ops/attention.py).

What must hold, for text rows, image rows, an image across the CP boundary, several images, a text-only microbatch and
get_batch's pad segment:
  - the CP ranks' decoder inputs, concatenated, equal the CP1 decoder input bitwise; every rank passes the CP1
    positions;
  - the vision parameters' gradients summed over the CP ranks (what Megatron DDP's reduce over the DP x CP group
    adds) equal the CP1 gradients;
  - the image counts of --qwen38-vision-lazy and of the vision audit equal CP1's (each image counted once, not once
    per CP rank);
  - the provider accepts CP > 1 only with --allgather-cp.

CP is simulated in one process: parallel_state's CP size / rank are patched, and the all-gather of the token ids
returns the whole stream (the test asserts it is handed this rank's slice).
"""

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_packing import IDS, _image_row, _packed, _text
from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import _HIDDEN, _forbid_vision_forward, _model

from miles_plugins.models.qwen3_8_next import vision, vision_audit, vision_lazy
from miles_plugins.models.qwen3_8_next.vision_inputs import scatter_image_embeddings, text_rope_positions


def _allgather_pack(rows, cp_size, pad_multiple=4):
    """get_batch's allgather-CP stream: rows back to back, one pad segment to a multiple of cp_size * pad_multiple."""
    ids = torch.cat([r[0] for r in rows])
    bounds = [0]
    for r in rows:
        bounds.append(bounds[-1] + len(r[0]))
    multiple = cp_size * pad_multiple
    pad = (multiple - ids.numel() % multiple) % multiple
    if pad:
        ids = torch.cat((ids, torch.zeros(pad, dtype=torch.long)))
        bounds.append(bounds[-1] + pad)
    grids = [r[1] for r in rows if r[1] is not None]
    positions = torch.cat([r[2] for r in rows])
    return ids, bounds, len(rows), (torch.cat(grids) if grids else None), positions


def _text_row(length, first=100):
    return _text(length, first), None, text_rope_positions(length)


def _image(grid=(1, 4, 4), before=2, after=2):
    return _image_row(grid, before=before, after=after)


# rows of each case; the comments give the stream at CP2 (the CP boundary is T / 2)
CASES = {
    # 16 tokens, no pad
    "text-only": [_text_row(7), _text_row(9, 50)],
    # T = 16: image tokens [3, 7), all on rank 0
    "image-in-first-slice": [_image(), _text_row(6)],
    # T = 24: image tokens [13, 17), all on rank 1
    "image-in-last-slice": [_text_row(10), _image()],
    # T = 24: image tokens [6, 14) across 12 (and across 8 at CP4)
    "image-straddles-boundary": [_text_row(3), _image((1, 4, 8), before=2, after=3), _text_row(2)],
    # T = 48: the first image's tokens [10, 26) across 24, a second image at [31, 35)
    "two-images-and-text": [_text_row(5), _image((1, 8, 8), before=4, after=1), _image(), _text_row(3)],
    # T = 16: an image row of 9 tokens (image tokens [2, 6)) and get_batch's pad segment of 7
    "image-then-pad": [_image((1, 4, 4), before=1, after=2)],
}


@pytest.fixture
def cp_world(monkeypatch):
    """Patch the CP size / rank and the token-id gather; returns a setter for (size, rank, whole stream)."""
    from megatron.core import parallel_state

    state = {"size": 1, "rank": 0, "whole": None}
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda check_initialized=True: object())
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: state["size"])
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: state["rank"])
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)

    def gather(local):
        whole = state["whole"]
        local_len = whole.numel() // state["size"]
        assert torch.equal(local, whole[state["rank"] * local_len : (state["rank"] + 1) * local_len])
        return whole.to(local.dtype)

    monkeypatch.setattr(vision, "gather_sequence_no_grad", gather)

    def set_cp(size, rank, whole):
        state.update(size=size, rank=rank, whole=whole)

    return set_cp


def _run(model, ids, bounds, rows, grid, positions, pixels, weight):
    inputs = {"qwen38_position_ids": positions}
    if grid is not None:
        inputs |= {"pixel_values": pixels, "image_grid_thw": grid}
    model(input_ids=ids[None], packed_seq_params=_packed(bounds, rows), **inputs)
    call = model.calls[-1]
    decoder = call["decoder_input"]
    # a loss that sums over tokens, so the CP ranks' losses add up to the CP1 loss
    (decoder * weight[: decoder.shape[0]]).sum().backward()
    return call


def _grads(model):
    return {name: p.grad.clone() for name, p in model.vision_model.named_parameters() if p.grad is not None}


def _zero_grads(model):
    for p in model.parameters():
        p.grad = None


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("case", list(CASES))
def test_cp_slices_concatenate_to_the_cp1_decoder_input_and_grads_sum_to_cp1(cp_world, case, cp_size):
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    ids, bounds, rows, grid, positions = _allgather_pack(CASES[case], cp_size)
    total = ids.numel()
    assert total % cp_size == 0
    torch.manual_seed(5)
    pixels = None if grid is None else torch.randn(int(grid.prod(-1).sum()), 24)
    weight = torch.randn(total, 1, _HIDDEN)

    cp_world(1, 0, ids)
    whole = _run(model, ids, bounds, rows, grid, positions, pixels, weight)
    whole_grads, whole_embedding_grad = _grads(model), model.embedding.word_embeddings.weight.grad.clone()
    _zero_grads(model)

    local = total // cp_size
    pieces = []
    for rank in range(cp_size):
        cp_world(cp_size, rank, ids)
        lo, hi = rank * local, (rank + 1) * local
        call = _run(model, ids[lo:hi], bounds, rows, grid, positions, pixels, weight[lo:hi])
        # every rank passes the whole stream's positions; the attention layer slices its rows
        assert torch.equal(call["position_ids"], whole["position_ids"])
        pieces.append(call["decoder_input"].detach())
    assert torch.equal(torch.cat(pieces), whole["decoder_input"].detach())

    summed = _grads(model)
    assert summed.keys() == whole_grads.keys() and summed
    for name, grad in whole_grads.items():
        torch.testing.assert_close(summed[name], grad, rtol=1e-5, atol=1e-6, msg=name)
    torch.testing.assert_close(model.embedding.word_embeddings.weight.grad, whole_embedding_grad, rtol=1e-5, atol=1e-6)
    if grid is not None:
        assert any(g.any() for g in whole_grads.values())


def test_the_straddling_case_really_splits_an_image_over_two_ranks():
    """Guard of the test data: the boundary case has image tokens on both sides of T / 2."""
    ids, *_ = _allgather_pack(CASES["image-straddles-boundary"], 2)
    half = ids.numel() // 2
    image = IDS["image_token_id"]
    assert (ids[:half] == image).any() and (ids[half:] == image).any()
    ids, *_ = _allgather_pack(CASES["two-images-and-text"], 2)
    half = ids.numel() // 2
    assert (ids[half - 1] == image) and (ids[half] == image)
    _, bounds, rows, *_ = _allgather_pack(CASES["image-then-pad"], 2)
    assert len(bounds) - 1 == rows + 1  # a pad segment after the rows


def test_a_text_microbatch_under_cp_skips_the_tower_on_every_rank_and_reports_zero_grads(cp_world, monkeypatch):
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    _forbid_vision_forward(model, monkeypatch)
    ids, bounds, rows, grid, positions = _allgather_pack(CASES["text-only"] + [_text_row(3, 70)], 2)
    local = ids.numel() // 2
    for rank in range(2):
        cp_world(2, rank, ids)
        model(
            input_ids=ids[rank * local : (rank + 1) * local][None],
            packed_seq_params=_packed(bounds, rows),
            qwen38_position_ids=positions,
        ).backward()
        assert model.calls[-1]["position_ids"].shape == (3, 1, ids.numel())
    assert all(p.grad is not None and not p.grad.any() for p in model.vision_model.parameters())


def test_an_image_only_on_the_other_rank_still_reaches_every_tower_parameter(cp_world):
    """Rank 0 holds no image token, yet its tower ran and every trainable parameter reports a (zero) gradient: the
    DDP buckets of the two CP ranks complete together."""
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    ids, bounds, rows, grid, positions = _allgather_pack(CASES["image-in-last-slice"], 2)
    local = ids.numel() // 2
    assert not (ids[:local] == IDS["image_token_id"]).any()
    cp_world(2, 0, ids)
    pixels = torch.randn(16, 24)
    _run(model, ids[:local], bounds, rows, grid, positions, pixels, torch.randn(local, 1, _HIDDEN))
    assert all(p.grad is not None and not p.grad.any() for p in model.vision_model.parameters())


def test_cp1_ignores_the_allgather_flag(cp_world):
    """At CP1 the wrapper never gathers: the flag alone changes nothing."""
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    ids, bounds, rows, grid, positions = _allgather_pack(CASES["image-in-first-slice"], 2)
    cp_world(1, 0, torch.zeros(1, dtype=torch.long))  # a gather would fail its own slice check
    model(
        input_ids=ids[None],
        packed_seq_params=_packed(bounds, rows),
        qwen38_position_ids=positions,
        pixel_values=torch.randn(16, 24),
        image_grid_thw=grid,
    )


def test_under_cp_the_microbatch_must_be_packed(cp_world):
    model = _model(scope="none", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    ids = _text(8)
    cp_world(2, 0, ids)
    with pytest.raises(ValueError, match="packed"):
        model(input_ids=ids[:4][None])


def test_under_cp_cu_seqlens_must_cover_the_whole_stream(cp_world):
    model = _model(scope="none", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    ids, bounds, rows, _, positions = _allgather_pack(CASES["text-only"], 2)
    cp_world(2, 0, ids)
    with pytest.raises(ValueError, match="whole"):
        model(input_ids=ids[:8][None], packed_seq_params=_packed([0, 8], 1), qwen38_position_ids=positions[:8])


# ── scatter_image_embeddings with a CP window, with and without sequence parallelism ──────────────────────────


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_scatter_with_a_cp_window_and_sp_equals_the_whole_scatter(cp_size, tp_size):
    torch.manual_seed(0)
    total = 32
    decoder = torch.randn(total, 1, 8)
    positions = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 30])  # runs across 8, 16 and 24
    embeddings = torch.randn(len(positions), 8)
    whole = scatter_image_embeddings(decoder, positions, embeddings)
    local, shard = total // cp_size, total // cp_size // tp_size
    pieces = []
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            start = cp_rank * local + tp_rank * shard
            pieces.append(
                scatter_image_embeddings(
                    decoder[start : start + shard],
                    positions,
                    embeddings,
                    tp_rank=tp_rank,
                    sequence_parallel=tp_size > 1,
                    window_start=cp_rank * local,
                )
            )
    assert torch.equal(torch.cat(pieces), whole)


def test_scatter_without_a_window_still_refuses_tokens_outside_the_input():
    with pytest.raises(ValueError, match="outside the decoder input"):
        scatter_image_embeddings(torch.zeros(4, 1, 2), torch.tensor([5]), torch.zeros(1, 2))


# ── image counts: lazy vision and the audit count each image once ─────────────────────────────────────────


@pytest.mark.parametrize("module", [vision_lazy, vision_audit], ids=["lazy", "audit"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_images_are_counted_once_per_dp_replica_not_once_per_cp_rank(monkeypatch, module, cp_size):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda check_initialized=True: object())
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: cp_size)
    counted = 0
    for cp_rank in range(cp_size):
        for tp_rank in range(2):
            monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda r=cp_rank: r)
            monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda r=tp_rank: r)
            counted += module._counts_images_here()
    assert counted == 1


def test_lazy_and_audit_hooks_see_the_whole_microbatch_on_every_cp_rank(cp_world):
    """The towers' forward pre-hooks count grid rows; every CP rank's tower runs on all images of the microbatch."""
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    seen = []
    model.vision_model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(int(kwargs["grid_thw"].shape[0])), with_kwargs=True
    )
    ids, bounds, rows, grid, positions = _allgather_pack(CASES["two-images-and-text"], 2)
    local = ids.numel() // 2
    pixels = torch.randn(int(grid.prod(-1).sum()), 24)
    for rank in range(2):
        cp_world(2, rank, ids)
        piece = ids[rank * local : (rank + 1) * local]
        _run(model, piece, bounds, rows, grid, positions, pixels, torch.zeros(local, 1, _HIDDEN))
    assert seen == [2, 2]


# ── the vision gradients are reduced over DP x CP ─────────────────────────────────────────────────────────


def test_vision_parameters_are_dense_and_megatron_ddp_reduces_dense_grads_over_dp_x_cp():
    """Each CP rank's tower gradient holds only its own image tokens' share; the CP1 gradient is their sum. Megatron
    DDP gives a parameter without ``allreduce=False`` (not an expert) to the dense buffers, which it reduces over the
    intra DP x CP group, as it does for the word embeddings. Pinned on Megatron's source: a change there must be
    looked at again."""
    import inspect

    from megatron.core.distributed import param_and_grad_buffer
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

    model = _model(scope="full", data="mixed")
    assert all(getattr(p, "allreduce", True) for p in model.vision_model.parameters())
    grouping = inspect.getsource(param_and_grad_buffer.group_params_for_buffers)
    assert "is_expert_parallel = not getattr(param, 'allreduce', True)" in grouping
    ddp = inspect.getsource(DistributedDataParallel.__init__)
    dense = ddp.index("data_parallel_group = self.intra_dp_cp_group")
    assert ddp.rindex("if buffer_key.is_expert_parallel:", 0, dense) > 0
