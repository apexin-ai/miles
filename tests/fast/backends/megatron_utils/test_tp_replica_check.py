import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import model as model_module
from miles.backends.megatron_utils import tp_replica_check
from miles.backends.megatron_utils.tp_replica_check import check_tp_replicas, compare_digest_dirs, named_tensors
from tests.fast.dist_utils import init_gloo, run_multiprocess

# gloo TP2 workers: about 80 s on the qwen38 smoke pod's CPU
register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["megatron", "ckpt"])


class _Chunk(torch.nn.Module):
    """A replicated bf16 param, a replicated fp32 1-element param (like GDN's A_log), a TP shard and an expert."""

    def __init__(self) -> None:
        super().__init__()
        self.replicated = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.bfloat16))
        self.a_log = torch.nn.Parameter(torch.tensor([0.5]))
        self.sharded = torch.nn.Parameter(torch.zeros(4))
        self.sharded.tensor_model_parallel = True
        self.expert = torch.nn.Parameter(torch.zeros(4))
        self.expert.allreduce = False


class _DistributedOptimizerLike:
    """Megatron DistributedOptimizer's layout: bf16 params step through fp32 master shards, fp32 params through
    views of their own shard; the torch optimizer keys its state by those shard tensors."""

    def __init__(self, chunk: _Chunk) -> None:
        bf16 = [chunk.replicated]
        fp32 = [chunk.a_log, chunk.sharded, chunk.expert]
        self.model_float16_groups = [bf16]
        self.shard_fp32_from_float16_groups = [[p.detach().float().view(-1)[:].clone() for p in bf16]]
        self.model_fp32_groups = [fp32]
        self.shard_fp32_groups = [[p.detach().view(-1)[:] for p in fp32]]
        shards = self.shard_fp32_groups[0] + self.shard_fp32_from_float16_groups[0]
        for shard in shards:
            shard.requires_grad_(True)
        self.optimizer = torch.optim.Adam(shards, lr=1e-3)
        for shard in shards:
            shard.grad = torch.ones_like(shard)
        self.optimizer.step()


def _check_worker(rank: int, world_size: int, port: int, *, perturb: str | None) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        chunk = _Chunk()
        with torch.no_grad():
            chunk.sharded.fill_(rank)
            chunk.expert.fill_(rank)
        optimizer = _DistributedOptimizerLike(chunk)
        master = optimizer.shard_fp32_from_float16_groups[0][0]
        a_log_state = optimizer.optimizer.state[optimizer.shard_fp32_groups[0][0]]
        if rank == 1 and perturb == "low_bits":
            # CRV-FIX-001: +1, -2, +1 ulp on three equal words keeps any linear checksum unchanged
            with torch.no_grad():
                master[0:3] = 1.0
                master.view(torch.int32)[0:3] += torch.tensor([1, -2, 1], dtype=torch.int32)
        elif rank == 0 and perturb == "low_bits":
            with torch.no_grad():
                master[0:3] = 1.0
        if rank == 1 and perturb == "a_log_moment":
            a_log_state["exp_avg_sq"][0] *= 1.5
        if rank == 1 and perturb == "step":
            a_log_state["step"] += 1

        expected = {
            "low_bits": "pp0.replicated#main ",
            "a_log_moment": "pp0.a_log#exp_avg_sq ",
            "step": "pp0.a_log#step ",
        }
        # replicated: 2 params; bf16 param: #main + step/exp_avg/exp_avg_sq; fp32 a_log: #main + 3 state entries
        if perturb is None:
            assert check_tp_replicas([chunk], optimizer, group=dist.group.WORLD, where="unit") == 10
        else:
            with pytest.raises(RuntimeError) as error:
                check_tp_replicas([chunk], optimizer, group=dist.group.WORLD, where="unit")
            message = str(error.value)
            assert "1 of 10 TP-replicated tensors differ" in message
            assert expected[perturb] in message
            assert "sharded" not in message and "expert" not in message
    finally:
        dist.destroy_process_group()


class TestNamedTensors:
    def test_fp32_params_and_scalar_optimizer_state_are_included(self) -> None:
        """CRV-FIX-002: fp32 params have no main_param; their state and the scalar step must still be checked."""
        chunk = _Chunk()
        names = [name for name, _ in named_tensors([chunk], _DistributedOptimizerLike(chunk), replicated_only=True)]
        assert names == [
            "pp0.replicated",
            "pp0.a_log",
            "pp0.a_log#main",
            "pp0.a_log#exp_avg",
            "pp0.a_log#exp_avg_sq",
            "pp0.a_log#step",
            "pp0.replicated#main",
            "pp0.replicated#exp_avg",
            "pp0.replicated#exp_avg_sq",
            "pp0.replicated#step",
        ]

    def test_a_plain_torch_optimizer_over_model_params_is_traced(self) -> None:
        """Without Megatron's shard groups the stepped tensors are the params themselves."""
        chunk = _Chunk()
        optimizer = torch.optim.Adam([chunk.a_log], lr=1e-3)
        chunk.a_log.grad = torch.ones(1)
        optimizer.step()
        names = [name for name, _ in named_tensors([chunk], optimizer, replicated_only=True)]
        assert names == ["pp0.replicated", "pp0.a_log", "pp0.a_log#exp_avg", "pp0.a_log#exp_avg_sq", "pp0.a_log#step"]

    def test_an_untraceable_optimizer_tensor_is_an_error_not_a_skip(self) -> None:
        """A layout the check cannot map must not report agreement it never verified."""
        stray = torch.nn.Parameter(torch.zeros(3))
        with pytest.raises(RuntimeError, match="not traceable to a model parameter"):
            list(named_tensors([_Chunk()], torch.optim.SGD([stray], lr=0.1), replicated_only=True))


class TestCheckTpReplicas:
    def test_identical_replicas_pass_while_sharded_and_expert_params_may_differ(self) -> None:
        """Replicated params, their fp32 master and every optimizer state entry are compared."""
        run_multiprocess(partial(_check_worker, perturb=None), world_size=2)

    def test_offsetting_low_bit_drift_is_caught(self) -> None:
        """CRV-FIX-001: the comparison is bitwise, not a checksum that offsetting changes can cancel."""
        run_multiprocess(partial(_check_worker, perturb="low_bits"), world_size=2)

    def test_drift_in_an_fp32_params_optimizer_moment_is_caught(self) -> None:
        """CRV-FIX-002: GDN's A_log stays fp32; its Adam state is reached through the shard mapping."""
        run_multiprocess(partial(_check_worker, perturb="a_log_moment"), world_size=2)

    def test_a_different_step_count_is_caught(self) -> None:
        """CRV-FIX-002: scalar state entries are compared too."""
        run_multiprocess(partial(_check_worker, perturb="step"), world_size=2)

    def test_check_is_a_no_op_without_torch_distributed(self) -> None:
        """Single-process tools and stubbed unit tests call it freely."""
        assert not dist.is_initialized()
        assert check_tp_replicas([_Chunk()], None, where="unit") == 0


class TestSaveOrder:
    def _run_save(self, monkeypatch: pytest.MonkeyPatch, *, check_fails: bool) -> list[str]:
        calls = []
        monkeypatch.setattr(model_module, "should_disable_forward_pre_hook", lambda args: True)
        monkeypatch.setattr(model_module, "disable_forward_pre_hook", lambda model: calls.append("param sync"))
        monkeypatch.setattr(model_module, "enable_forward_pre_hook", lambda model: calls.append("enable hook"))
        monkeypatch.setattr(model_module, "is_lora_model", lambda model: False)
        monkeypatch.setattr(model_module, "save_checkpoint", lambda *a, **k: calls.append("save"))

        def check(model, optimizer, *, where):
            calls.append(f"check {where}")
            if check_fails:
                raise RuntimeError("replicas differ")

        monkeypatch.setattr(model_module, "check_tp_replicas", check)
        args = make_trainer_args(ci_test=False)
        try:
            model_module.save(args=args, iteration=3, model=[torch.nn.Linear(2, 2)], optimizer=None, opt_param_scheduler=None)
        except RuntimeError:
            calls.append("raised")
        return calls

    def test_the_check_runs_after_the_pending_param_sync_and_before_the_save(self, monkeypatch) -> None:
        """CRV-FIX-003: with overlapped param gather the params are final only after the sync."""
        assert self._run_save(monkeypatch, check_fails=False) == [
            "param sync",
            "check save iteration 3",
            "save",
            "enable hook",
        ]

    def test_a_failed_check_skips_the_save_and_restores_the_hook(self, monkeypatch) -> None:
        """No checkpoint of drifted replicas; the forward pre-hook comes back on the error path."""
        assert self._run_save(monkeypatch, check_fails=True) == [
            "param sync",
            "check save iteration 3",
            "enable hook",
            "raised",
        ]


def _dump(root: Path, tag: str, ranks: dict[int, dict], world_size: int = 2) -> None:
    out = root / tag
    out.mkdir(parents=True)
    for rank, tensors in ranks.items():
        record = {"rank": rank, "world_size": world_size, "where": tag, "tensors": tensors}
        (out / f"rank{rank:05d}.json").write_text(json.dumps(record))


class TestDigestDump:
    def test_dumps_compare_tensor_by_tensor(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save/reload pair that changed one tensor is reported with that tensor's name."""
        monkeypatch.setenv("MILES_TENSOR_DIGEST_DIR", str(tmp_path))
        chunk = _Chunk()
        check_tp_replicas([chunk], None, where="save iteration 3")
        with torch.no_grad():
            chunk.sharded[0] = 7.0
        check_tp_replicas([chunk], None, where="load iteration 3")

        before, after = tmp_path / "save_iteration_3", tmp_path / "load_iteration_3"
        record = json.loads((before / "rank00000.json").read_text())
        assert record["world_size"] == 1
        assert set(record["tensors"]) == {"pp0.replicated", "pp0.a_log", "pp0.sharded", "pp0.expert"}
        [problem] = compare_digest_dirs(before, after)
        assert problem.startswith("rank 0: pp0.sharded:")
        assert compare_digest_dirs(before, before) == []

    @pytest.mark.parametrize(
        ("setup", "message"),
        [
            ("missing", "does not exist"),
            ("empty", "holds no rank dumps"),
            ("missing_rank", "incomplete dump, ranks present [0]"),
            ("missing_tensor", "rank 1: b: 2 -> missing"),
        ],
    )
    def test_incomplete_or_absent_dumps_never_compare_as_identical(self, tmp_path, setup, message) -> None:
        """CRV-FIX-004: a wrong tag, an empty directory, a missing rank or tensor is a failure."""
        full = {0: {"a": "1", "b": "2"}, 1: {"a": "1", "b": "2"}}
        _dump(tmp_path, "save", full)
        if setup == "empty":
            (tmp_path / "load").mkdir()
        elif setup == "missing_rank":
            _dump(tmp_path, "load", {0: full[0]})
        elif setup == "missing_tensor":
            _dump(tmp_path, "load", {0: full[0], 1: {"a": "1"}})
        problems = compare_digest_dirs(tmp_path / "save", tmp_path / "load")
        assert any(message in p for p in problems), problems

    def test_the_cli_exits_non_zero_for_a_missing_tag(self, tmp_path, monkeypatch, capsys) -> None:
        """The command SMOKE runs must not print identical for a typo."""
        _dump(tmp_path, "save", {0: {"a": "1"}, 1: {"a": "1"}})
        monkeypatch.setattr("sys.argv", ["tp_replica_check", str(tmp_path), "save", "lod"])
        with pytest.raises(SystemExit) as exit_info:
            exec(compile(Path(tp_replica_check.__file__).read_text(), tp_replica_check.__file__, "exec"), {"__name__": "__main__"})
        assert exit_info.value.code == 1
        assert "does not exist" in capsys.readouterr().out
