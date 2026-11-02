"""CPU tests for --balance-data-lockstep-group (get_lockstep_balanced_partitions and its two callers)."""

from __future__ import annotations

import random
from argparse import Namespace

import pytest
from tests.fast.utils.test_dp_schedule import assert_invariants, make_args, make_tp

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.utils.arguments import _validate_balance_data_lockstep_group
from miles.utils.dp_schedule import build_dp_schedule
from miles.utils.seqlen_balancing import get_lockstep_balanced_partitions, get_seqlen_balanced_partitions


def lockstep_time(partitions, weights, group):
    """Step time when the ranks of a group run their k-th items together: max over groups of the sum over k
    of the heaviest k-th item of the group."""
    times = []
    for g in range(0, len(partitions), group):
        members = partitions[g : g + group]
        times.append(sum(max(weights[p[k]] for p in members) for k in range(len(members[0]))))
    return max(times)


def heavy_tailed_lengths(rng, n):
    """Mostly similar rows plus a few long and a few short ones, like a formal bucket step."""
    base = rng.choice([4_000, 20_000, 60_000, 200_000])
    lens = [int(base * rng.uniform(0.9, 1.1)) for _ in range(n)]
    for _ in range(rng.randint(0, 6)):
        lens[rng.randrange(n)] = rng.randint(200, 262_144)
    return lens


@pytest.mark.parametrize("seed", range(40))
def test_lockstep_partitions_keep_rows_and_counts(seed):
    rng = random.Random(seed)
    dp, group, per_rank = rng.choice([(8, 2, 4), (52, 2, 4), (16, 4, 3), (6, 3, 2)])
    weights = heavy_tailed_lengths(rng, dp * per_rank)
    parts = get_lockstep_balanced_partitions(weights, dp, group)
    assert len(parts) == dp
    assert all(len(p) == per_rank for p in parts)
    assert sorted(i for p in parts for i in p) == list(range(len(weights)))
    # Inside a group the ranks run the slots lightest first: the slot weight (heaviest member) never drops.
    for g in range(0, dp, group):
        slot_w = [max(weights[p[k]] for p in parts[g : g + group]) for k in range(per_rank)]
        assert slot_w == sorted(slot_w)


def test_lockstep_group_one_is_token_balance_up_to_order():
    rng = random.Random(0)
    weights = [rng.randint(1, 1000) for _ in range(32)]
    lock = get_lockstep_balanced_partitions(weights, 8, 1)
    assert sorted(map(sorted, lock)) == sorted(map(sorted, get_seqlen_balanced_partitions(weights, 8, True)))


def test_lockstep_beats_token_balance_on_formal_like_steps():
    """Under lockstep pairs the new partition is never slower than the token balance in index order, and
    faster in total over many heavy-tailed steps."""
    rng = random.Random(1234)
    old_total = new_total = 0
    for _ in range(200):
        weights = heavy_tailed_lengths(rng, 208)
        rng.shuffle(weights)
        old = lockstep_time(get_seqlen_balanced_partitions(weights, 52, True), weights, 2)
        new = lockstep_time(get_lockstep_balanced_partitions(weights, 52, 2), weights, 2)
        assert new <= old * 1.0001
        old_total += old
        new_total += new
    assert new_total < 0.99 * old_total


def test_lockstep_rejects_bad_shapes():
    with pytest.raises(AssertionError):
        get_lockstep_balanced_partitions([1] * 12, 6, 4)  # 6 ranks not a multiple of 4
    with pytest.raises(AssertionError):
        get_lockstep_balanced_partitions([1] * 10, 4, 2)  # 10 rows not a multiple of 4 ranks


def _sched_args(group, balance_data=True):
    args = make_args(micro_batch_size=1, balance_data=balance_data)
    return args.model_copy(update={"balance_data_lockstep_group": group})


def test_build_dp_schedule_default_is_token_balance():
    rng = random.Random(7)
    lens = [rng.randint(100, 262_144) for _ in range(64)]
    parts, mbi, nmb, _ = build_dp_schedule(
        _sched_args(1), make_tp(dp_size=16), lens, global_batch_size=64, rollout_indices=list(range(64))
    )
    assert parts == get_seqlen_balanced_partitions(lens, 16, equal_size=True)
    assert nmb == [4]


def test_build_dp_schedule_lockstep_static_multi_step():
    rng = random.Random(11)
    dp, steps, per_rank = 8, 3, 4
    gbs = dp * per_rank
    lens = [rng.randint(100, 262_144) for _ in range(gbs * steps)]
    args = _sched_args(2)
    parts, mbi, nmb, _ = build_dp_schedule(
        args, make_tp(dp_size=dp), lens, global_batch_size=gbs, rollout_indices=list(range(len(lens)))
    )
    assert nmb == [per_rank] * steps
    assert_invariants(parts, mbi, nmb, dp_size=dp, expected_global_sample_indices=range(len(lens)), total_lengths=lens)
    base, _, _, _ = build_dp_schedule(
        _sched_args(1), make_tp(dp_size=dp), lens, global_batch_size=gbs, rollout_indices=list(range(len(lens)))
    )
    for s in range(steps):
        # same rows in every step, same count per rank
        new_rows = [parts[r][s * per_rank : (s + 1) * per_rank] for r in range(dp)]
        old_rows = [base[r][s * per_rank : (s + 1) * per_rank] for r in range(dp)]
        assert sorted(i for p in new_rows for i in p) == sorted(i for p in old_rows for i in p)
        local = {i: lens[i] for p in new_rows for i in p}
        assert lockstep_time(new_rows, local, 2) <= lockstep_time(old_rows, local, 2)


def test_split_train_data_by_dp_raw_lockstep():
    rng = random.Random(3)
    lens = [rng.randint(1, 50) for _ in range(16)]
    data = {"tokens": [[0] * n for n in lens], "response_lengths": lens, "sample_indices": list(range(16))}
    old = split_train_data_by_dp_raw(
        Namespace(balance_data=True, balance_data_lockstep_group=1), dict(data), dp_size=4
    )
    new = split_train_data_by_dp_raw(
        Namespace(balance_data=True, balance_data_lockstep_group=2), dict(data), dp_size=4
    )
    assert [list(s["partition"]) for s in old] == get_seqlen_balanced_partitions(lens, 4, equal_size=True)
    assert [list(s["partition"]) for s in new] == get_lockstep_balanced_partitions(lens, 4, 2)
    for shard in new:
        assert [len(t) for t in shard["tokens"]] == [lens[i] for i in shard["partition"]]


def _megatron_ns(**kw):
    base = dict(
        balance_data_lockstep_group=2,
        balance_data=True,
        balance_by_flops=False,
        train_backend="megatron",
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    )
    return Namespace(**(base | kw))


def test_validate_lockstep_group_matches_layout():
    _validate_balance_data_lockstep_group(_megatron_ns())  # EP 8 x ETP 1 over TP 4 -> 2 DP ranks
    _validate_balance_data_lockstep_group(_megatron_ns(balance_data_lockstep_group=4, tensor_model_parallel_size=2))
    with pytest.raises(AssertionError):
        _validate_balance_data_lockstep_group(_megatron_ns(balance_data_lockstep_group=4))
    with pytest.raises(AssertionError):  # EP group inside one DP rank: nothing runs in lockstep across ranks
        _validate_balance_data_lockstep_group(_megatron_ns(expert_model_parallel_size=4))
    with pytest.raises(AssertionError):
        _validate_balance_data_lockstep_group(_megatron_ns(balance_data=False))
