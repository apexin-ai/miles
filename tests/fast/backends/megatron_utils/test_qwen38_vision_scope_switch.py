"""CPU contracts for the projector -> full vision scope switch on resume (vision_scope_switch).

Megatron's real layout function (DistributedOptimizer._compute_per_buffer_param_layout) and its real per-DP-rank range
builder (DistributedOptimizer._build_model_gbuf_range) run here on the tiny wired model; a dp_reshardable checkpoint
is simulated by laying each old parameter's values into flat old-bucket tensors the way Megatron saves them.
The DistributedOptimizer load itself needs CUDA and is covered by the 4-layer GPU test (req/VLF7-2).
"""

from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_param_groups import _model, _Wrapped

from miles_plugins.models.qwen3_8_next import vision_scope_switch as switch

_DDP = SimpleNamespace(pad_buckets_for_high_nccl_busbw=False)


def _ddp_params(model):
    """What Megatron DDP hands the layout: trainable params in named_parameters order."""
    return [p for _, p in _Wrapped(model).named_parameters() if p.requires_grad]


def _names(model):
    return {p: n for n, p in _Wrapped(model).named_parameters()}


def _layout(params, dp, bucket_size=None):
    return switch.compute_layout(params, bucket_size, dp, _DDP)


def _by_name(layout, names):
    return {names[p]: v for p, v in layout.param_index_map.items()}


@pytest.mark.parametrize("dp", [1, 2, 8])
@pytest.mark.parametrize("bucket_size", [None, 3000])
def test_the_old_layout_is_exactly_what_a_real_projector_run_laid_out(dp, bucket_size):
    full, projector = _model("full", seed=3), _model("projector", seed=3)
    full_params = _ddp_params(full)
    running = _layout(full_params, dp, bucket_size)
    new = switch.new_trainable_parameters([full], "projector")
    assert new and all(".merger." not in n and n.startswith("vision_model.") for n in new.values())
    old = switch.old_layout_for(
        full_params,
        new,
        bucket_sizes=[None, 3000],
        data_parallel_world_size=dp,
        ddp_config=_DDP,
        running=(running.param_index_map, running.per_bucket_numel_unpadded),
    )
    real = _layout(_ddp_params(projector), dp, bucket_size)
    assert _by_name(old, _names(full)) == _by_name(real, _names(projector))
    assert old.bucket_indices == real.bucket_indices
    assert old.per_bucket_numel_unpadded == real.per_bucket_numel_unpadded


def test_cross_check_the_vit_is_one_block_after_the_merger_and_the_language_shifts_by_its_size():
    full = _model("full")
    names = _names(full)
    layout = _layout(_ddp_params(full), 1)
    order = sorted(layout.param_index_map, key=lambda p: layout.param_index_map[p][0])
    kinds = ["merger" if ".merger." in names[p] else "vit" if "vision_model." in names[p] else "lang" for p in order]
    first_vit, last_vit = kinds.index("vit"), len(kinds) - 1 - kinds[::-1].index("vit")
    assert set(kinds[:first_vit]) == {"merger"} and set(kinds[first_vit : last_vit + 1]) == {"vit"}
    assert set(kinds[last_vit + 1 :]) == {"lang"}
    new = switch.new_trainable_parameters([full], "projector")
    old = switch.old_layout_for(
        _ddp_params(full),
        new,
        bucket_sizes=[None],
        data_parallel_world_size=1,
        ddp_config=_DDP,
        running=(layout.param_index_map, layout.per_bucket_numel_unpadded),
    )
    shifts = {
        layout.param_index_map[p][0] - old.param_index_map[p][0]
        for p in order
        if names[p] not in new.values() and "vision_model." not in names[p]
    }
    vit_start = layout.param_index_map[order[first_vit]][0]
    lang_start = layout.param_index_map[order[last_vit + 1]][0]
    assert shifts == {lang_start - vit_start}


def test_the_old_layout_is_refused_when_the_running_buffer_is_not_reproduced():
    full = _model("full")
    params = _ddp_params(full)
    running = _layout(params, 1, 3000)
    with pytest.raises(ValueError, match="does not reproduce the running DDP buffer layout"):
        switch.old_layout_for(
            params,
            switch.new_trainable_parameters([full], "projector"),
            bucket_sizes=[None],
            data_parallel_world_size=1,
            ddp_config=_DDP,
            running=(running.param_index_map, running.per_bucket_numel_unpadded),
        )


def _rank_param_maps(layout, dp, rank):
    """Megatron's own per-rank ownership: DistributedOptimizer._build_model_gbuf_range on a buffer with this layout."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    buffer = SimpleNamespace(
        data_parallel_group=SimpleNamespace(rank=lambda: rank, size=lambda: dp),
        buckets=[
            SimpleNamespace(grad_data=torch.empty(end - start), offset=start) for start, end in layout.bucket_indices
        ],
        param_index_map=layout.param_index_map,
    )
    return [
        DistributedOptimizer._build_model_gbuf_range(buffer, b)["param_map"] for b in range(len(layout.bucket_indices))
    ]


def _saved_old_buckets(old, values):
    """A dp_reshardable checkpoint's flat bucket tensors (unpadded size); NaN where no parameter is."""
    buckets = [torch.full((n,), float("nan"), dtype=torch.float64) for n in old.per_bucket_numel_unpadded]
    for p, (start, end, b) in old.param_index_map.items():
        base = old.bucket_indices[b][0]
        buckets[b][start - base : end - base] = values[p]
    return buckets


@pytest.mark.parametrize("dp", [1, 2, 4])
@pytest.mark.parametrize("bucket_size", [None, 3000])
def test_every_slice_a_rank_owns_after_the_switch_is_read_from_where_the_projector_run_saved_it(dp, bucket_size):
    full = _model("full")
    params = _ddp_params(full)
    new = switch.new_trainable_parameters([full], "projector")
    running = _layout(params, dp, bucket_size)
    old = switch.old_layout_for(
        params,
        new,
        bucket_sizes=[None, bucket_size],
        data_parallel_world_size=dp,
        ddp_config=_DDP,
        running=(running.param_index_map, running.per_bucket_numel_unpadded),
    )
    values = {p: torch.arange(p.numel(), dtype=torch.float64) + 1e6 * i for i, p in enumerate(params) if p not in new}
    saved = _saved_old_buckets(old, values)
    seen = {p: torch.zeros(p.numel(), dtype=torch.bool) for p in params}
    for rank in range(dp):
        maps = _rank_param_maps(running, dp, rank)
        plan = switch.plan_buffer(old, maps, new)
        for param_map, row in zip(maps, plan.slices):
            for (p, ranges), piece in zip(param_map.items(), row):
                inside = ranges["param"]
                seen[p][inside.start : inside.end] = True
                if p in new:
                    assert piece is None
                    continue
                got = saved[piece.old_bucket][piece.offset : piece.offset + piece.numel]
                assert torch.equal(got, values[p][inside.start : inside.end])
                assert piece.bucket_numel == old.per_bucket_numel_unpadded[piece.old_bucket]
        # the gaps are exactly the old padding, so params + gaps cover every old bucket once
        for b, gaps in plan.gaps.items():
            mask = torch.zeros(old.per_bucket_numel_unpadded[b], dtype=torch.int)
            for s, e in gaps:
                mask[s:e] += 1
                assert torch.isnan(saved[b][s:e]).all()
            for p, (s, e, pb) in old.param_index_map.items():
                if pb == b:
                    base = old.bucket_indices[b][0]
                    mask[s - base : e - base] += 1
            assert torch.all(mask == 1)
    assert all(bool(m.all()) for m in seen.values())  # the ranks together own every element of every parameter


def test_switching_down_or_sideways_is_refused():
    switch.check_switch("projector", "full")
    switch.check_switch("none", "full")
    switch.check_switch("none", "projector")
    for a, b in [
        ("full", "projector"),
        ("projector", "projector"),
        ("full", "full"),
        ("projector", "none"),
        ("x", "full"),
    ]:
        with pytest.raises(ValueError):
            switch.check_switch(a, b)


def test_none_to_projector_adds_only_the_merger():
    model = _model("projector")
    new = switch.new_trainable_parameters([model], "none")
    assert new and all(".merger." in n for n in new.values())


def _group(wd, lr_mult, step=None, params=("x",), **extra):
    g = {
        "wd_mult": wd,
        "lr_mult": lr_mult,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
        "params": list(params),
        **extra,
    }
    if step is not None:
        g["step"] = step
    return g


def test_param_groups_only_the_new_vit_needs_are_added_with_the_checkpoints_step():
    saved = [_group(1.0, 1.0, 7, max_lr=1e-5), _group(0.0, 1.0, 7, max_lr=1e-5)]
    running = [
        _group(1.0, 1.0, max_lr=9.0),
        _group(0.0, 1.0),
        _group(1.0, 0.2, max_lr=2e-6),
        _group(0.0, 0.2, max_lr=2e-6),
    ]
    added = switch.add_missing_param_groups(saved, running, allowed_indices={2, 3})
    assert added == [2, 3]
    assert [(g["lr_mult"], g["wd_mult"], g["step"], g["max_lr"]) for g in saved[2:]] == [
        (0.2, 1.0, 7, 2e-6),
        (0.2, 0.0, 7, 2e-6),
    ]
    assert saved[0]["max_lr"] == 1e-5  # existing groups keep the checkpointed values
    # an empty stub group (no new parameter on this rank) may be added too
    assert switch.add_missing_param_groups(
        [_group(1.0, 1.0, 7)], [_group(1.0, 1.0), _group(1.0, 0.2, params=())], set()
    ) == [1]


def test_a_missing_group_with_old_parameters_or_mixed_steps_is_refused():
    with pytest.raises(ValueError, match="holds old parameters"):
        switch.add_missing_param_groups(
            [_group(1.0, 1.0, 7)], [_group(1.0, 1.0), _group(0.5, 1.0)], allowed_indices=set()
        )
    with pytest.raises(ValueError, match="different steps"):
        switch.add_missing_param_groups([_group(1.0, 1.0, 7), _group(0.0, 1.0, 8)], [_group(1.0, 1.0)], set())


def test_install_needs_megatrons_distributed_optimizer():
    model = _model("full")
    with pytest.raises(ValueError, match="DistributedOptimizer"):
        switch.install([model], torch.optim.SGD(model.parameters(), lr=0.1), from_scope="projector", to_scope="full")


# --- the DistributedOptimizer glue, on a stand-in with exactly the attributes the switch reads ------------------------

_DTYPE = (torch.float32, torch.float32)
_STATES = ("param", "exp_avg", "exp_avg_sq")


class _FakeDistOpt:
    """Holds per-rank main param / Adam shards like Megatron's DistributedOptimizer (dense buffer, one dtype)."""

    def __init__(self, model, layout, dp, rank, bucket_size):
        self.params = _ddp_params(model)
        self.buffers = [
            SimpleNamespace(
                params=self.params,
                param_index_map=layout.param_index_map,
                buckets=[SimpleNamespace(numel_unpadded=n) for n in layout.per_bucket_numel_unpadded],
                data_parallel_world_size=dp,
                ddp_config=_DDP,
            )
        ]
        self.gbuf_ranges = [{_DTYPE: [{"param_map": m} for m in _rank_param_maps(layout, dp, rank)]}]
        self.per_bucket_numel_unpadded = [{_DTYPE: list(layout.per_bucket_numel_unpadded)}]
        self.data_parallel_group = SimpleNamespace(rank=lambda: rank)
        self.data_parallel_group_idx = 0
        self.distributed_optimizer_instance_id = 0
        owned = [p for bucket in self.gbuf_ranges[0][_DTYPE] for p in bucket["param_map"]]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(owned)}
        self.optimizer = SimpleNamespace(param_groups=[{"params": owned}])
        self.shards = {}
        for bucket in self.gbuf_ranges[0][_DTYPE]:
            for p, ranges in bucket["param_map"].items():
                n = ranges["param"].size
                self.shards[p] = {k: torch.full((n,), -7.0) for k in _STATES} | {"step": torch.tensor(5.0)}
        self.loaded_with = None

    def _get_main_param_and_optimizer_states(self, model_param):
        return dict(self.shards[model_param])

    def sharded_param_state_dp_reshardable(self, *args):
        raise AssertionError("the original request must not be built during a switch load")

    def load_parameter_state_from_dp_reshardable(self, state_dict):
        self.loaded_with = state_dict

    def load_state_dict(self, state_dict):
        self.load_parameter_state_from_dp_reshardable(state_dict["param_state"])


def _fake_load(request, old, saved_by_state):
    """What dist_checkpointing.load does with the request: fill each ShardedTensor from the checkpoint's tensor."""
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor

    out = {}
    for name, value in request.items():
        if isinstance(value, LocalNonpersistentObject):
            out[name] = value.unwrap()
        elif isinstance(value, ShardedObject):
            assert value.key == f"optimizer.distributed.dp_group_idx_0.{name}"
            out[name] = (
                [{_DTYPE: list(old.per_bucket_numel_unpadded)}]
                if name.endswith("unpadded")
                else [{_DTYPE: [e - s for s, e in old.bucket_indices]}]
            )
        else:
            loaded = {}
            for key, tensor in value.items():
                assert isinstance(tensor, ShardedTensor)
                prefix, state = tensor.key.rsplit(".", 1)
                bucket = int(prefix.rsplit("bucket_idx_", 1)[1])
                assert prefix == f"optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_{_DTYPE}.bucket_idx_{bucket}"
                assert tensor.global_shape == (old.per_bucket_numel_unpadded[bucket],)
                (offset,) = tensor.global_offset
                loaded[key] = saved_by_state[state][bucket][offset : offset + tensor.local_shape[0]].clone()
            out[name] = loaded
    return out


@pytest.mark.parametrize("dp", [1, 2])
def test_the_switch_reads_old_buckets_and_hands_megatron_the_new_layout(dp):
    full = _model("full")
    params = _ddp_params(full)
    new = switch.new_trainable_parameters([full], "projector")
    running = _layout(params, dp)
    old = _layout([p for p in params if p not in new], dp)
    values = {
        k: {
            p: torch.arange(p.numel(), dtype=torch.float32) + 1000 * i + 1e5 * j
            for i, p in enumerate(params)
            if p not in new
        }
        for j, k in enumerate(_STATES)
    }
    saved = {
        k: [b.float() for b in _saved_old_buckets(old, {p: v.double() for p, v in values[k].items()})] for k in _STATES
    }
    gap_cover = {}
    for rank in range(dp):
        opt = _FakeDistOpt(full, running, dp, rank, None)
        sw = switch._Switch(opt, new, [None])
        sw.bind()
        request = opt.sharded_param_state_dp_reshardable(None, True, None)
        assert bool(request["gaps"]) == (rank == 0 and any(sw.plans[0].gaps.values()))
        for key, tensor in request["gaps"].items():
            gap_cover.setdefault(key, tensor.global_offset)
        opt.load_parameter_state_from_dp_reshardable(_fake_load(request, old, saved))
        handed = opt.loaded_with
        assert handed["per_bucket_numel_unpadded"] == opt.per_bucket_numel_unpadded
        for bucket, rows in zip(opt.gbuf_ranges[0][_DTYPE], handed[0][_DTYPE]):
            assert len(rows) == len(bucket["param_map"])
            for (p, ranges), got in zip(bucket["param_map"].items(), rows):
                a, b = ranges["param"].start, ranges["param"].end
                assert got["padding"] is False and torch.equal(got["step"], torch.tensor(5.0))
                if p in new:
                    assert torch.equal(got["param"], p.detach().view(-1)[a:b])
                    assert not got["exp_avg"].any() and not got["exp_avg_sq"].any()
                else:
                    for k in _STATES:
                        assert torch.equal(got[k], values[k][p][a:b]), (k, new.get(p))
        assert sw.loaded and sw.summary["fresh_slices_here"] == sum(
            1 for m in opt.gbuf_ranges[0][_DTYPE] for p in m["param_map"] if p in new
        )
        # once loaded, saves and later loads take Megatron's own path
        with pytest.raises(AssertionError, match="original request"):
            opt.sharded_param_state_dp_reshardable(None, True, None)
    assert gap_cover  # the tiny model's 64-alignment leaves old padding, read on DP rank 0


def test_a_checkpoint_whose_buckets_do_not_match_the_old_layout_is_refused():
    full = _model("full")
    params = _ddp_params(full)
    new = switch.new_trainable_parameters([full], "projector")
    running = _layout(params, 1)
    opt = _FakeDistOpt(full, running, 1, 0, None)
    sw = switch._Switch(opt, new, [None])
    sw.bind()
    request = opt.sharded_param_state_dp_reshardable(None, True, None)
    old = _layout([p for p in params if p not in new], 1)
    saved = {
        k: [
            b.float()
            for b in _saved_old_buckets(
                old, {p: torch.zeros(p.numel(), dtype=torch.float64) for p in params if p not in new}
            )
        ]
        for k in _STATES
    }
    loaded = _fake_load(request, old, saved)
    loaded["per_bucket_numel_unpadded"] = opt.per_bucket_numel_unpadded  # a checkpoint already in the new layout
    with pytest.raises(ValueError, match="do not match the layout without the new vision parameters"):
        opt.load_parameter_state_from_dp_reshardable(loaded)


def test_an_optimizer_without_new_parameters_keeps_megatrons_own_load():
    projector = _model("projector")
    params = _ddp_params(projector)
    opt = _FakeDistOpt(projector, _layout(params, 1), 1, 0, None)
    sw = switch._Switch(opt, switch.new_trainable_parameters([_model("full")], "projector"), [None])
    sw.bind()
    assert not sw.translate
    with pytest.raises(AssertionError, match="original request"):
        opt.sharded_param_state_dp_reshardable(None, True, None)


def test_only_groups_whose_local_members_are_all_new_may_be_missing_from_the_checkpoint():
    """CX1b P2: a missing group that also holds old (language / merger) parameters must be refused."""
    old_a, old_b, vit_a, vit_b = (torch.nn.Parameter(torch.zeros(2)) for _ in range(4))
    index_map = {old_a: (0, 0), vit_a: (0, 1), old_b: (1, 0), vit_b: (2, 0)}
    new = {vit_a: "vision_model.blocks.0.a", vit_b: "vision_model.blocks.0.b"}
    assert switch.groups_holding_only_new(index_map, new) == {2}
    running = [_group(1.0, 1.0), _group(0.0, 1.0), _group(1.0, 0.2)]
    # the mixed group 0 (old_a + vit_a) is missing: refused, although it holds a new parameter
    with pytest.raises(ValueError, match="holds old parameters"):
        switch.add_missing_param_groups([_group(0.0, 1.0, 7)], running, switch.groups_holding_only_new(index_map, new))
    # only the all-new ViT group 2 is missing: added
    saved = [_group(1.0, 1.0, 7), _group(0.0, 1.0, 7)]
    assert switch.add_missing_param_groups(saved, running, switch.groups_holding_only_new(index_map, new)) == [2]
