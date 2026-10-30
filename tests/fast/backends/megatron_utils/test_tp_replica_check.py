import json
from functools import partial

import pytest
import torch
import torch.distributed as dist

from miles.backends.megatron_utils import tp_replica_check
from miles.backends.megatron_utils.tp_replica_check import check_tp_replicas, compare_digest_dirs, tensor_digest
from tests.fast.dist_utils import init_gloo, run_multiprocess


class TestTensorDigest:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16, torch.int64, torch.uint8])
    @pytest.mark.parametrize("numel", [1, 7, 1000])
    def test_any_single_bit_flip_changes_the_digest(self, dtype: torch.dtype, numel: int) -> None:
        """A one-ulp drift anywhere in the tensor is visible."""
        base = (torch.arange(numel) % 13).to(dtype)
        reference = tensor_digest(base)
        assert torch.equal(tensor_digest(base.clone()), reference)
        words = base.view(-1).view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[base.element_size()])
        for position in {0, numel // 2, numel - 1}:
            flipped = words.clone()
            flipped[position] ^= 1
            assert not torch.equal(tensor_digest(flipped.view(dtype)), reference)

    def test_swapping_two_elements_changes_the_digest(self) -> None:
        """The position-weighted sum catches permutations the plain sum misses."""
        base = torch.tensor([1.0, 2.0, 3.0])
        assert not torch.equal(tensor_digest(base[[1, 0, 2]]), tensor_digest(base))

    def test_digest_does_not_depend_on_the_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunking bounds the int64 temporaries without changing the result."""
        data = torch.randn(1001, dtype=torch.float32)
        whole = tensor_digest(data)
        monkeypatch.setattr(tp_replica_check, "_CHUNK", 64)
        assert torch.equal(tensor_digest(data), whole)

    def test_non_contiguous_views_hash_their_values(self) -> None:
        """A strided view hashes like its contiguous copy."""
        data = torch.randn(8, 6)
        assert torch.equal(tensor_digest(data.t()), tensor_digest(data.t().contiguous()))


class _Chunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.replicated = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.bfloat16))
        self.sharded = torch.nn.Parameter(torch.zeros(4))
        self.sharded.tensor_model_parallel = True
        self.expert = torch.nn.Parameter(torch.zeros(4))
        self.expert.allreduce = False


def _check_worker(rank: int, world_size: int, port: int, *, perturb: str | None) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        chunk = _Chunk()
        with torch.no_grad():
            chunk.sharded.fill_(rank)
            chunk.expert.fill_(rank)
        optimizer = torch.optim.Adam(chunk.parameters())
        main = chunk.replicated.detach().float().clone()
        chunk.replicated.main_param = main
        optimizer.state[main] = {"exp_avg": torch.ones(12), "exp_avg_sq": torch.ones(12), "step": torch.tensor(3.0)}
        if rank == 1 and perturb == "param":
            with torch.no_grad():
                chunk.replicated.view(torch.int16)[5] += 1
        if rank == 1 and perturb == "moment":
            optimizer.state[main]["exp_avg_sq"][3] = 1.5

        if perturb is None:
            assert check_tp_replicas([chunk], optimizer, group=dist.group.WORLD, where="unit") == 4
        else:
            with pytest.raises(RuntimeError) as error:
                check_tp_replicas([chunk], optimizer, group=dist.group.WORLD, where="unit")
            message = str(error.value)
            expected = "pp0.replicated " if perturb == "param" else "pp0.replicated#exp_avg_sq "
            assert "1 of 4 TP-replicated tensors differ" in message
            assert expected in message
            assert "TP ranks [1]" in message
            assert "sharded" not in message and "expert" not in message
    finally:
        dist.destroy_process_group()


class TestCheckTpReplicas:
    def test_identical_replicas_pass_while_sharded_and_expert_params_may_differ(self) -> None:
        """Only TP-replicated params, their fp32 master and moments are compared."""
        run_multiprocess(partial(_check_worker, perturb=None), world_size=2)

    def test_one_ulp_drift_in_a_replicated_param_is_reported_by_name(self) -> None:
        """The error names the param, the TP rank and the max |diff|."""
        run_multiprocess(partial(_check_worker, perturb="param"), world_size=2)

    def test_drift_in_an_optimizer_moment_is_reported_by_name(self) -> None:
        """Equal weights with different Adam state still diverge after the next step."""
        run_multiprocess(partial(_check_worker, perturb="moment"), world_size=2)

    def test_check_is_a_no_op_without_torch_distributed(self) -> None:
        """Single-process tools and stubbed unit tests call it freely."""
        assert not dist.is_initialized()
        assert check_tp_replicas([_Chunk()], None, where="unit") == 0


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
        assert set(json.loads((before / "rank00000.json").read_text())) == {"pp0.replicated", "pp0.sharded", "pp0.expert"}
        [problem] = compare_digest_dirs(before, after)
        assert problem.startswith("rank00000.json: pp0.sharded:")
        assert compare_digest_dirs(before, before) == []
