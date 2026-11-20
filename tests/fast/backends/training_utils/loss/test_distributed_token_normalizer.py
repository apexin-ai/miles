"""Check the Miles loss normalizer through Megatron's PP broadcast and DP×CP reduction."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils import loss as loss_module
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args


register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["precision"])


class _ScalarChunk(torch.nn.Module):
    """One trainable scalar with the two gradient operations used by finalize_model_grads."""

    def __init__(self, replicas):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)
        self.replicas = replicas

    def finish_grad_sync(self, force_all_reduce=False):
        dist.all_reduce(self.weight.grad, group=self.replicas)

    def scale_gradients(self, scale):
        self.weight.grad.mul_(scale)


def _linear_loss(args, batch, logits, reduce_tokens):
    value = reduce_tokens(logits)
    return value, {"loss": value.detach()}


def _group_info(group):
    return GroupInfo(rank=group.rank(), size=group.size(), group=group)


def _check_step(groups, *, context_parallel, all_masked):
    # The same 2x2 grid exercises PP2/DP2 and PP2/CP2 in turn.
    replicas = _group_info(groups.dp_cp)
    pp = _group_info(groups.pp)
    single = _group_info(groups.tp)
    cp = replicas if context_parallel else single
    set_parallel_state(
        ParallelState(
            intra_dp=single if context_parallel else replicas,
            intra_dp_cp=replicas,
            cp=cp,
            tp=single,
            pp=pp,
            ep=single,
            etp=single,
            indep_dp=single,
            is_pp_last_stage=pp.rank == 1,
        )
    )
    model = _ScalarChunk(groups.dp_cp)
    tokens = torch.zeros((), dtype=torch.int64)
    if pp.rank == 1:
        args = make_args(loss_type="sft_loss", calculate_per_token_loss=True)
        for mask, slope in (([1, 1, 1, 1], 2), ([0, 0, 0, 1], 5)):
            # One DP replica contributes nothing. CP ranks share the full sample mask.
            if all_masked or (not context_parallel and replicas.rank == 0):
                mask = [0] * 4
            batch = {"loss_masks": [torch.tensor(mask)], "total_lengths": [8], "response_lengths": [4]}
            # In zigzag CP2, response logits at positions 3..6 split 1:3 over the ranks.
            local_length = (1, 3)[cp.rank] if context_parallel else 4
            logits = (model.weight * slope).expand(local_length)
            with patch.object(loss_module, "get_loss_function", return_value=_linear_loss):
                loss, count, _ = loss_module.loss_function(args, batch, 2, logits, apply_megatron_loss_scaling=True)
            loss.backward()
            tokens += count
    else:
        model.weight.grad = torch.zeros_like(model.weight)

    # Represent the backward signal on the preceding stage with a scalar copy.
    # The token-count broadcast and the division below are Megatron's production code.
    dist.broadcast(model.weight.grad, group=groups.pp, group_src=1)
    finalize_model_grads([model], tokens, pg_collection=groups)

    # Full-sequence reference: four tokens have slope 2 and one has slope 5.
    torch.testing.assert_close(model.weight.grad, torch.tensor(0.0 if all_masked else 13 / 5))
    assert tokens.item() == (0 if all_masked else 5 * cp.size)


def _worker(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    try:
        singles = [dist.new_group([r]) for r in range(world_size)]
        pipelines = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        replicas = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        groups = SimpleNamespace(
            tp=singles[rank], pp=pipelines[rank // 2], dp_cp=replicas[rank % 2], embd=None, pos_embd=None
        )
        for context_parallel in (False, True):
            for all_masked in (False, True):
                _check_step(groups, context_parallel=context_parallel, all_masked=all_masked)
    finally:
        dist.destroy_process_group()


def test_distributed_token_normalizer():
    run_multiprocess(_worker, world_size=4)
