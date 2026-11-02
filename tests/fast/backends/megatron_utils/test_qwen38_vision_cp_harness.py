"""Independent CPU harness: the Qwen3.8 vision path under allgather-CP2 against CP1 (GBSMIG V2, 2026-11-02).

Written apart from the vision-CP implementation and against stable entry points only:

* the batch comes from miles' real ``get_batch``: CP1 with ``allgather_cp=False``; CP2 with ``allgather_cp=True``,
  where each CP rank gets one contiguous slice of the globally padded packed stream, ``cu_seqlens`` stays global
  and the multimodal inputs (patches, grids, ``qwen38_position_ids``) are those of the whole microbatch;
* the model is ``vision.wire_visual`` on a stand-in language model, called as ``forward_step`` calls it.

The two CP ranks are two gloo processes, so every process-group call the implementation makes is real. Megatron's
``parallel_state`` CP / TP getters, the miles ``ParallelState`` and any copy of those getters bound into a
``miles_plugins.models.qwen3_8_next`` module are pointed at the gloo groups.

Claims, per packed layout (``CASES``):

* decoder input: the CP ranks' pieces, concatenated, equal the CP1 decoder input bitwise (fp64; each rank runs the
  same tower on the same images, so no tolerance);
* mRoPE positions: each CP rank hands the language model either its ``[3, T/2]`` slice of the CP1 ``[3, T]``
  positions or the whole CP1 positions (both ranks the same form), exactly;
* gradients: the CP ranks' losses sum to the CP1 loss and, for every parameter (vision tower, word embeddings, head),
  the per-rank gradients sum to the CP1 gradient (fp64; rtol 1e-10, atol 1e-8 x the largest |CP1 gradient| of
  that parameter): vision parameters are replicated over CP and DDP sums their gradients over DP x CP. The atol
  is rounding, measured without any CP code: one tower backward against two backwards on the two halves of the
  same upstream gradient differ by up to 2.1e-10 x max|grad| when an image straddles the split (0 otherwise). A
  missing or doubled contribution is of the order of the gradient itself;
* every trainable vision parameter gets a gradient on every CP rank (zero on a text microbatch), as on CP1:
  with --overlap-grad-reduce a parameter that never reports keeps its DDP bucket from being reduced;
* the lazy-vision and audit image counters (the tower's forward pre-hooks) count on each CP rank the images CP1
  counts: every CP rank runs the tower on all images of the microbatch.

CP1 itself is checked against an oracle built from the embedding and the tower directly, so a CP change that moves
the CP1 path shows up here too.

Until the vision-CP branch is used, ``_VISION_CP_IMPLEMENTED`` is False and the CP2 claims are strict xfails.
"""

import json
import os
import tempfile
import time
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.backends.megatron_utils.test_vision_provider_contracts import base_model  # noqa: F401  (fixture)

_VISION_CP_IMPLEMENTED = False
_CP2 = pytest.mark.xfail(not _VISION_CP_IMPLEMENTED, strict=True, reason="vision CP not implemented")

IMG, VSTART, VEND = 9001, 9002, 9003
_HIDDEN = 16
_VOCAB = 10000
_PATCH_DIM = 3 * 2 * 2 * 2  # in_channels x temporal_patch_size x patch_size^2
_PAD_MULTIPLIER = 8  # TP1: get_batch pads CP1 to 8 tokens, allgather-CP2 the global stream to 16
_CONFIG = {
    "image_token_id": IMG,
    "vision_start_token_id": VSTART,
    "vision_end_token_id": VEND,
    "vision_config": {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "out_hidden_size": _HIDDEN,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [],
    },
}
GRAD_RTOL, GRAD_ATOL_SCALE = 1e-10, 1e-8
_JOIN_TIMEOUT_S = 240
_GLOO_TIMEOUT_S = 60

# A row is a list of segments: an int is a run of text tokens, a (t, h, w) tuple an image (its vision-start token,
# h/2 * w/2 image pad tokens, its vision-end token). The bool says whether the row carries qwen38_position_ids
# (image rows always do; the messages source gives text rows theirs too). Every layout's CP1 and CP2 padded totals
# are equal (checked), so the two streams line up token for token. Comments give global token indices; the CP split
# is at T/2.
CASES = {
    # 20 + 27 = 47 tokens + 1 pad; split at 24 inside row 2; no multimodal inputs at all
    "text_only": [([20], False), ([27], False)],
    # the same with text positions (messages source)
    "text_only_with_positions": [([20], True), ([27], True)],
    # 32 tokens, image pads 7..10 on rank 0; rank 1 holds text only
    "one_image_row": [([6, (1, 4, 4), 20], True)],
    # 16 + 16; row 1 image pads 4..7 (rank 0), row 2 image pads 19..26 (rank 1)
    "two_image_rows": [([3, (1, 4, 4), 7], True), ([2, (1, 4, 8), 4], True)],
    # 32 tokens, image pads 12..19 straddle the split at 16
    "image_straddles_split": [([11, (1, 4, 8), 11], True)],
    # 32 tokens, two images in one row: pads 6..9 (rank 0) and 15..22 (straddles 16)
    "two_images_in_a_row_one_straddles": [([5, (1, 4, 4), 3, (1, 4, 8), 8], True)],
    # 32 tokens, vision-start at 15 (last of rank 0), image pads 16..19 (first of rank 1)
    "image_starts_at_split": [([15, (1, 4, 4), 11], True)],
    # 32 tokens, image pads 8..15 (last of rank 0), vision-end at 16 (first of rank 1)
    "image_ends_at_split": [([7, (1, 4, 8), 15], True)],
    # 20 + 5 = 25 tokens + 7 pad (one trailing pad segment, on rank 1)
    "pad_segment_at_end": [([4, (1, 4, 4), 10], True), ([5], True)],
    # 12 + 52 = 64; the image is on rank 0, the long text row covers the split and all of rank 1
    "image_and_long_text_row": [([2, (1, 4, 4), 4], True), ([52], True)],
    # 18 + 14 = 32; rank 0 holds text only, the image pads 22..25 are on rank 1
    "image_row_in_second_half": [([18], True), ([3, (1, 4, 4), 5], True)],
}


# ---------------------------------------------------------------------------------------------------------- inputs


def _row(segments, with_positions, seed):
    from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, text_rope_positions

    ids, grids, text_id = [], [], 100
    for segment in segments:
        if isinstance(segment, int):
            ids += range(text_id, text_id + segment)
            text_id += segment
        else:
            _, h, w = segment
            ids += [VSTART] + [IMG] * ((h // 2) * (w // 2)) + [VEND]
            grids.append(list(segment))
    ids = torch.tensor(ids, dtype=torch.long)
    if grids:
        grid = torch.tensor(grids, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        patches = int(grid.prod(dim=1).sum())
        return ids, {
            "pixel_values": torch.randn(patches, _PATCH_DIM, generator=generator),
            "image_grid_thw": grid,
            "qwen38_position_ids": image_rope_positions(
                ids, grid, image_token_id=IMG, vision_start_token_id=VSTART, vision_end_token_id=VEND, merge_size=2
            ),
        }
    if with_positions:
        return ids, {"qwen38_position_ids": text_rope_positions(len(ids)).clone()}
    return ids, None


def _rows(case):
    return [_row(segments, with_positions, seed=1000 + i) for i, (segments, with_positions) in enumerate(CASES[case])]


class _Iterator:
    """What get_batch reads from miles' DataIterator for one microbatch."""

    rollout_data = {}

    def __init__(self, rows):
        lengths = [len(ids) for ids, _ in rows]
        responses = [max(1, n // 2) for n in lengths]
        self.batch = {
            "tokens": [ids.clone() for ids, _ in rows],
            "loss_masks": [torch.ones(r, dtype=torch.int) for r in responses],
            "multimodal_train_inputs": [inputs for _, inputs in rows],
            "total_lengths": lengths,
            "response_lengths": responses,
            "max_seq_lens": [max(lengths)] * len(rows),
            "adapter_slots": None,
        }

    def get_next(self, keys):
        return {key: self.batch.get(key) for key in keys}


def _get_batch(rows, *, allgather_cp):
    from miles.backends.megatron_utils.parallel import get_packed_seq_params
    from miles.backends.training_utils.data import rollout

    batch = rollout.get_batch(
        _Iterator(rows),
        ["tokens", "loss_masks", "multimodal_train_inputs", "total_lengths", "response_lengths", "max_seq_lens"],
        _PAD_MULTIPLIER,
        "thd",
        allgather_cp=allgather_cp,
    )
    packed = get_packed_seq_params(batch, SimpleNamespace(qkv_format="thd"))
    return batch, packed


# ----------------------------------------------------------------------------------------------------------- model


def _language_loss(head, hidden):
    return torch.tanh(head(hidden)).square().sum()  # a per-token sum: the CP ranks' losses add up to CP1's


class _Embedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(_VOCAB, _HIDDEN, dtype=torch.float64)

    def forward(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids).transpose(0, 1)  # Megatron layout [s, b, h]


class _Language(torch.nn.Module):
    """Stands in for the Megatron GPT model: records what the vision wrapper hands it and returns a loss."""

    def __init__(self, cp_size) -> None:
        super().__init__()
        self.pre_process = True
        self.post_process = True
        self.config = SimpleNamespace(
            params_dtype=torch.float64,
            sequence_parallel=False,
            context_parallel_size=cp_size,
            qwen3_8_next_allgather_cp=cp_size > 1,
        )
        self.embedding = _Embedding()
        self.head = torch.nn.Linear(_HIDDEN, _HIDDEN, dtype=torch.float64)
        self.calls = []

    def forward(self, input_ids=None, position_ids=None, decoder_input=None, **kwargs):
        hidden = decoder_input if decoder_input is not None else self.embedding(input_ids)
        self.calls.append({"decoder_input": hidden.detach().clone(), "position_ids": position_ids})
        return _language_loss(self.head, hidden)


def _model(cp_size):
    from miles_plugins.models.qwen3_8_next import vision, vision_audit, vision_lazy

    torch.manual_seed(0)
    model = _Language(cp_size)
    vision.wire_visual(model, _CONFIG, scope="full", recompute=False, freeze_language=False, data="mixed")
    tower = model.vision_model
    tower._qwen38_lazy_images = 0
    tower._qwen38_audit_images = 0
    tower.register_forward_pre_hook(vision_lazy._count_images, with_kwargs=True)
    tower.register_forward_pre_hook(vision_audit._count_images, with_kwargs=True)
    return model


def _run(model, batch, packed):
    """forward_step's model call, then backward; what the language model saw, the loss and the gradients."""
    mm = batch["multimodal_train_inputs"] or {}
    loss = model(
        input_ids=batch["tokens"],
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=packed,
        loss_mask=batch["input_loss_masks"],
        **mm,
    )
    loss.backward()
    call = model.calls[-1]
    positions = call["position_ids"]
    return {
        "error": None,
        "total": int(batch["cu_seqlens_host"][-1]),
        "tokens": batch["tokens"][0].clone(),
        "decoder_input": call["decoder_input"][:, 0],
        "position_ids": None if positions is None else positions.detach()[:, 0].clone(),
        "loss": loss.detach().clone(),
        "grads": {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        },
        "lazy_images": model.vision_model._qwen38_lazy_images,
        "audit_images": model.vision_model._qwen38_audit_images,
    }


# ------------------------------------------------------------------------------------------------- CP1 (in-process)


def _cp1_results():
    import megatron.core.parallel_state as mpu

    from miles.backends.training_utils import parallel
    from miles.backends.training_utils.data import rollout

    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(size=1))
    results = {}
    with pytest.MonkeyPatch.context() as m:
        m.setattr(parallel, "_parallel_state", state)
        m.setattr(rollout.current_platform, "current_device", lambda: torch.device("cpu"))
        m.setattr(mpu, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
        for case in CASES:
            rows = _rows(case)
            batch, packed = _get_batch(rows, allgather_cp=False)
            results[case] = _run(_model(cp_size=1), batch, packed)
            results[case]["oracle"] = _oracle(rows, results[case]["total"])
    return results


def _oracle(rows, total):
    """CP1 from first principles: word embeddings with the tower's outputs on the image pad tokens; each row's own
    positions (text rows without any: the token index), 0 on the trailing pad."""
    from miles_plugins.models.qwen3_8_next.vision_inputs import text_rope_positions

    model = _model(cp_size=1)
    ids = torch.cat([r for r, _ in rows])
    pad = total - len(ids)
    ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long)])
    hidden = model.embedding(ids[None])
    inputs = [x for _, x in rows if x is not None and "pixel_values" in x]
    if inputs:
        out = model.vision_model(
            torch.cat([x["pixel_values"] for x in inputs]).to(torch.float64),
            grid_thw=torch.cat([x["image_grid_thw"] for x in inputs]),
            return_dict=True,
        ).pooler_output
        hidden = hidden.clone()
        hidden[(ids == IMG).nonzero().flatten(), 0] = out
    loss = _language_loss(model.head, hidden)
    loss.backward()
    positions = torch.cat(
        [
            (x["qwen38_position_ids"] if x is not None else text_rope_positions(len(r)))
            for r, x in rows
        ]
        + [torch.zeros(pad, 3, dtype=torch.long)]
    ).T
    grads = {
        name: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    return {"decoder_input": hidden.detach()[:, 0], "position_ids": positions, "loss": loss.detach(), "grads": grads}


# ----------------------------------------------------------------------------------------------- CP2 (two ranks)


def _point_cp_at(rank, size, cp_group, tp_group):
    """Megatron's and miles' parallel state, and every bound copy in the qwen3_8_next modules, for this CP rank."""
    import sys

    import megatron.core.parallel_state as mpu

    from miles.backends.training_utils.data import rollout
    from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=size, group=cp_group)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=cp,
            cp=cp,
            tp=GroupInfo(rank=0, size=1, group=tp_group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )
    rollout.current_platform.current_device = lambda: torch.device("cpu")
    getters = {
        "get_context_parallel_world_size": lambda *a, **k: size,
        "get_context_parallel_rank": lambda *a, **k: rank,
        "get_context_parallel_group": lambda *a, **k: cp_group,
        "get_context_parallel_global_ranks": lambda *a, **k: list(range(size)),
        "get_tensor_model_parallel_world_size": lambda *a, **k: 1,
        "get_tensor_model_parallel_rank": lambda *a, **k: 0,
        "get_tensor_model_parallel_group": lambda *a, **k: tp_group,
    }
    for name, fn in getters.items():
        setattr(mpu, name, fn)
    bound = {"cp_world_size": lambda: size, "cp_rank_of": lambda: rank, **getters}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("miles_plugins.models.qwen3_8_next"):
            continue
        for name, fn in bound.items():
            if callable(getattr(module, name, None)):
                setattr(module, name, fn)


def _cp2_worker(rank, world_size, port, out_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=_GLOO_TIMEOUT_S)
    )
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
    # import every module the vision path uses before rebinding, so bound copies are found
    import miles_plugins.models.qwen3_8_next.ops.cp  # noqa: F401
    import miles_plugins.models.qwen3_8_next.vision  # noqa: F401
    import miles_plugins.models.qwen3_8_next.vision_audit  # noqa: F401
    import miles_plugins.models.qwen3_8_next.vision_lazy  # noqa: F401

    _point_cp_at(rank, world_size, dist.group.WORLD, tp_group)
    results = {}
    for case in CASES:
        try:
            batch, packed = _get_batch(_rows(case), allgather_cp=True)
            results[case] = _run(_model(cp_size=world_size), batch, packed)
        except Exception as error:  # recorded per case; the tests report it
            results[case] = {"error": f"{type(error).__name__}: {error}\n" + traceback.format_exc(limit=6)}
    torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _cp2_results():
    from tests.fast.dist_utils import find_free_port

    with tempfile.TemporaryDirectory() as out_dir:
        context = mp.start_processes(
            _cp2_worker,
            args=(2, find_free_port(), out_dir),
            nprocs=2,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + _JOIN_TIMEOUT_S
        try:
            # join() returns after each process exit, False while any is still running
            while not context.join(timeout=max(1.0, deadline - time.monotonic())):
                if time.monotonic() > deadline:
                    for process in context.processes:
                        process.kill()
                    return {"infra_error": f"CP2 workers did not finish in {_JOIN_TIMEOUT_S} s"}
        except Exception as error:
            return {"infra_error": f"{type(error).__name__}: {error}"}
        return {rank: torch.load(os.path.join(out_dir, f"rank{rank}.pt")) for rank in (0, 1)}


@pytest.fixture(scope="module")
def results():
    return {"cp1": _cp1_results(), "cp2": _cp2_results()}


def _cp2(results, case):
    cp2 = results["cp2"]
    assert "infra_error" not in cp2, cp2.get("infra_error")
    ranks = [cp2[0][case], cp2[1][case]]
    for rank, result in enumerate(ranks):
        if result["error"] is not None:
            raise AssertionError(f"CP2 rank {rank} failed:\n{result['error']}")
    cp1 = results["cp1"][case]
    assert all(r["total"] == cp1["total"] for r in ranks), "layout guard: CP1 and CP2 padded totals differ"
    return cp1, ranks


def _first_diff(a, b):
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    where = (a != b).nonzero()
    return f"{len(where)} mismatches, first at {where[0].tolist()}: {a[tuple(where[0])]} != {b[tuple(where[0])]}"


# ------------------------------------------------------------------------------------------------------------ tests


def test_cp2_workers_ran(results):
    """The harness itself: both CP ranks finished every case (a case may fail; that is recorded, not raised)."""
    cp2 = results["cp2"]
    assert "infra_error" not in cp2, cp2.get("infra_error")
    assert set(cp2[0]) == set(cp2[1]) == set(CASES)


@pytest.mark.parametrize("case", CASES)
def test_cp1_matches_the_oracle(results, case):
    cp1 = results["cp1"][case]
    oracle = cp1["oracle"]
    assert torch.equal(cp1["decoder_input"], oracle["decoder_input"]), _first_diff(
        cp1["decoder_input"], oracle["decoder_input"]
    )
    assert torch.equal(cp1["position_ids"], oracle["position_ids"]), _first_diff(
        cp1["position_ids"], oracle["position_ids"]
    )
    torch.testing.assert_close(cp1["loss"], oracle["loss"], rtol=1e-12, atol=0)
    for name, grad in cp1["grads"].items():
        assert grad is not None, f"CP1: {name} reported no gradient"
        torch.testing.assert_close(grad, oracle["grads"][name], rtol=1e-12, atol=1e-14, msg=lambda m: f"{name}: {m}")
    images = sum(len(x["image_grid_thw"]) for _, x in _rows(case) if x is not None and "image_grid_thw" in x)
    assert cp1["lazy_images"] == cp1["audit_images"] == images


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_tokens_are_the_contiguous_halves(results, case):
    """Guards the input contract the other CP2 claims rest on (get_batch's allgather-CP slice)."""
    cp1, ranks = _cp2(results, case)
    assert torch.equal(torch.cat([r["tokens"] for r in ranks]), cp1["tokens"])


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_decoder_input_slices_concatenate_to_cp1(results, case):
    cp1, ranks = _cp2(results, case)
    half = cp1["total"] // 2
    for rank, result in enumerate(ranks):
        assert result["decoder_input"].shape == (half, _HIDDEN), f"rank {rank}: {tuple(result['decoder_input'].shape)}"
    got = torch.cat([r["decoder_input"] for r in ranks])
    assert torch.equal(got, cp1["decoder_input"]), _first_diff(got, cp1["decoder_input"])


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_positions_match_cp1(results, case):
    """Each CP rank hands the language model either its own [lo, hi) slice of the CP1 positions or the whole CP1
    positions (when the language side slices them itself, e.g. for an indexer that reads the whole sequence), and
    both ranks do the same. Which of the two the language side expects is not observable here."""
    cp1, ranks = _cp2(results, case)
    want, half = cp1["position_ids"], cp1["total"] // 2
    forms = []
    for rank, result in enumerate(ranks):
        got = result["position_ids"]
        assert got is not None, f"rank {rank}: no position_ids reached the language model"
        if got.shape == (3, cp1["total"]):
            assert torch.equal(got, want), f"rank {rank} (whole): {_first_diff(got, want)}"
            forms.append("whole")
        else:
            piece = want[:, rank * half : (rank + 1) * half]
            assert torch.equal(got, piece), f"rank {rank} (slice [{rank * half}, {(rank + 1) * half})): " + (
                _first_diff(got, piece)
            )
            forms.append("slice")
    assert forms[0] == forms[1], f"the CP ranks use different position forms: {forms}"


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_losses_and_gradients_sum_to_cp1(results, case):
    cp1, ranks = _cp2(results, case)
    torch.testing.assert_close(ranks[0]["loss"] + ranks[1]["loss"], cp1["loss"], rtol=GRAD_RTOL, atol=0)
    assert set(ranks[0]["grads"]) == set(ranks[1]["grads"]) == set(cp1["grads"])
    for name, want in cp1["grads"].items():
        parts = [r["grads"][name] for r in ranks]
        got = sum(torch.zeros_like(want) if g is None else g for g in parts)
        atol = GRAD_ATOL_SCALE * float(want.abs().max())
        torch.testing.assert_close(got, want, rtol=GRAD_RTOL, atol=atol, msg=lambda m: f"{name} (atol {atol:.1e}): {m}")


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_every_rank_reports_every_vision_gradient(results, case):
    _, ranks = _cp2(results, case)
    for rank, result in enumerate(ranks):
        missing = [n for n, g in result["grads"].items() if n.startswith("vision_model.") and g is None]
        assert not missing, f"rank {rank}: {len(missing)} vision parameters without a gradient, e.g. {missing[:3]}"


@_CP2
@pytest.mark.parametrize("case", CASES)
def test_cp2_image_counters_see_the_cp1_images_on_every_rank(results, case):
    cp1, ranks = _cp2(results, case)
    for rank, result in enumerate(ranks):
        got = (result["lazy_images"], result["audit_images"])
        assert got == (cp1["lazy_images"], cp1["audit_images"]), f"rank {rank}: {got}"


# ------------------------------------------------------------------------------------------- provider guard (CP)


def _provider_args(tmp_path, **updates):
    from tests.fast.backends.megatron_utils.test_vision_provider_contracts import _trainer_config

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return _trainer_config(
        "miles_plugins.models.qwen3_8_next.vision.get_qwen38_vision_model_provider",
        "--qkv-format",
        "thd",
        "--position-embedding-type",
        "mrope",
        "--mrope-section",
        "1",
        "1",
        "2",
        "--no-rope-fusion",
        "--qwen38-vision-trainable",
        "full",
        hf_checkpoint=str(tmp_path),
        **updates,
    )


@pytest.fixture
def provider_stubs(monkeypatch, base_model):
    """The real get_model_provider_func over a stub GPTModel (test_vision_provider_contracts' fixture)."""
    from miles_plugins.models.qwen3_8_next import vision

    monkeypatch.setattr(vision, "wire_visual", lambda *a, **k: None)
    return base_model


@_CP2
def test_provider_accepts_allgather_cp2(tmp_path, provider_stubs):
    from miles.backends.megatron_utils import model_provider

    args = _provider_args(tmp_path, context_parallel_size=2, allgather_cp=True)
    model_provider.get_model_provider_func(args)(pre_process=True, post_process=True)
    assert len(provider_stubs) == 1


def test_provider_refuses_zigzag_cp2(tmp_path, provider_stubs):
    """Only the contiguous allgather-CP slice has the affine local-to-global map the vision path needs."""
    from miles.backends.megatron_utils import model_provider

    args = _provider_args(tmp_path, context_parallel_size=2, allgather_cp=False)
    with pytest.raises(ValueError):
        model_provider.get_model_provider_func(args)(pre_process=True, post_process=True)
    assert provider_stubs == []
