"""Qwen3.8-Flash-Next text SFT: ``--log-probs-backend fused`` must train like the torch backend.

The run's setting: ``--loss-type sft_loss --calculate-per-token-loss``, packed thd docs with thd padding after
them, a multi-turn loss mask, bf16 logits from an untied lm_head over the 248320-token vocabulary split over TP.
Each case runs the whole ``loss_function`` on ``hidden @ lm_head.T`` and compares, per token and after the
backward, the two backends and a plain-torch log_softmax over the true vocabulary.

Megatron pads the vocabulary to ``128 * TP``: 248320 needs no padding at TP 1, 2 and 4, and 512 columns at TP 8,
all on the last shard. There the torch backend normalizes over the padding (see ``--log-probs-backend``) and the
fused one does not, so the padded case holds the fused backend to the true-vocabulary reference.

TP=2 runs over NCCL with two GPUs and over gloo on one.
"""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

import os
from argparse import Namespace
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.dist_utils import find_free_port

from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

_VOCAB = 248_320  # the HF text_config.vocab_size, passed as --vocab-size
_TP8_PADDED_VOCAB = 248_832  # Megatron's padded vocabulary at TP 8: a multiple of 128 * 8
_HIDDEN = 256
# (prompt, response) per packed doc. The second is sft_rollout's fully masked sample, a one-token response.
_DOCS = [(120, 300), (64, 1), (200, 450), (30, 260), (90, 70)]
_THD_PADDING_ROWS = 13

_MAX_LOG_PROB_DIFF = 1e-3
_MAX_REL_DIFF = 1e-3


@dataclass(frozen=True)
class _Case:
    hidden: torch.Tensor  # [T, H] bf16, the output layer's input
    lm_head: torch.Tensor  # [padded vocab, H] bf16; padding rows zero, as loaded from the HF checkpoint
    tokens: list[torch.Tensor]
    loss_masks: list[torch.Tensor]
    total_lengths: list[int]
    response_lengths: list[int]


def _response_loss_mask(response: int) -> torch.Tensor:
    """Assistant turns trained, the observation turns between them not; a one-token response is the fully
    masked sample."""
    mask = torch.ones(response)
    if response == 1:
        return mask * 0
    mask[response // 3 : response // 3 + response // 6] = 0
    mask[-response // 8 :] = 0
    mask[-response // 16 :] = 1
    return mask


def _make_case(device, *, vocab: int, width: int, docs=_DOCS, hidden_size=_HIDDEN, seed=0) -> _Case:
    """The same case on every rank: each rank keeps its own lm_head shard."""
    gen = torch.Generator(device=device).manual_seed(seed)
    total_lengths = [prompt + response for prompt, response in docs]
    n_rows = sum(total_lengths) + _THD_PADDING_ROWS
    hidden = torch.randn(n_rows, hidden_size, device=device, generator=gen).to(torch.bfloat16)
    lm_head = torch.randn(width, hidden_size, device=device, generator=gen) * (4 / hidden_size**0.5)
    lm_head[vocab:] = 0
    lm_head = lm_head.to(torch.bfloat16)
    # every third row's target is its argmax: a confident token, where 1 - p cancels
    best = (hidden @ lm_head[:vocab].T).argmax(dim=-1)
    flat = torch.randint(0, vocab, (n_rows,), device=device, generator=gen)
    confident = torch.arange(n_rows, device=device) % 3 == 0
    flat[1:] = torch.where(confident[:-1], best[:-1], flat[1:])  # row j scores token j + 1
    return _Case(
        hidden=hidden,
        lm_head=lm_head,
        tokens=list(flat[: sum(total_lengths)].split(total_lengths)),
        loss_masks=[_response_loss_mask(response).to(device) for _, response in docs],
        total_lengths=total_lengths,
        response_lengths=[response for _, response in docs],
    )


@dataclass(frozen=True)
class _Step:
    log_probs: torch.Tensor  # [response tokens], fp32
    loss: torch.Tensor
    logits_grad_norm: float  # over every TP shard
    lm_head_grad_norm: float
    lm_head_grad: torch.Tensor  # this rank's shard, fp32


def _set_tp(rank: int, size: int, group) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=GroupInfo(rank=rank, size=size, group=group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _global_norm(local: torch.Tensor, group) -> float:
    squares = local.float().square().sum()
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(squares, group=group)
    return squares.sqrt().item()


def _sft_args(*, backend: str, recompute: bool, vocab: int) -> Namespace:
    """The trainer args the SFT loss reads, as the run sets them."""
    return Namespace(
        loss_type="sft_loss",
        log_probs_backend=backend,
        recompute_loss_function=recompute,
        qkv_format="thd",
        allgather_cp=False,
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        debug_unified_grad_fused_logprob=False,
        custom_loss_function_path=None,
        use_dynamic_global_batch_size=False,
        multi_lora=False,
        train_backend="megatron",
        backend=Namespace(calculate_per_token_loss=True, vocab_size=vocab, global_batch_size=1),
    )


def _sft_step(case: _Case, *, backend: str, recompute: bool, vocab: int, tp_rank: int, tp_size: int, group) -> _Step:
    """One micro-batch of the run's loss on this rank's vocab shard, as Megatron's train step calls it."""
    args = _sft_args(backend=backend, recompute=recompute, vocab=vocab)
    batch = dict(
        unconcat_tokens=case.tokens,
        response_lengths=case.response_lengths,
        total_lengths=case.total_lengths,
        loss_masks=case.loss_masks,
    )
    width = case.lm_head.size(0) // tp_size
    hidden = case.hidden.clone().requires_grad_(True)
    lm_head = case.lm_head[tp_rank * width : (tp_rank + 1) * width].clone().requires_grad_(True)
    logits = (hidden @ lm_head.T).unsqueeze(0)  # [1, T, V / TP] bf16: the model output with labels=None
    logits_grads = []
    # the fused backward hands over the logits' own storage as their gradient
    logits.register_hook(lambda grad: logits_grads.append(grad.float()))

    with torch.no_grad():
        scored = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=case.tokens,
            total_lengths=case.total_lengths,
            response_lengths=case.response_lengths,
        )
    loss, _, _ = loss_function(args, batch, 1, logits, apply_megatron_loss_scaling=True)
    loss.backward()
    return _Step(
        log_probs=torch.cat(scored["log_probs"]).float(),
        loss=loss.detach().float(),
        logits_grad_norm=_global_norm(logits_grads[0], group),
        lm_head_grad_norm=_global_norm(lm_head.grad, group),
        lm_head_grad=lm_head.grad.float(),
    )


def _true_vocab_reference(case: _Case, *, vocab: int, tp_rank: int, tp_size: int) -> _Step:
    """Plain torch on the full logits: row ``j`` of a doc scores its token ``j + 1``, over the first ``vocab``
    columns, summed over the loss mask (the per-token loss before Megatron divides by the token count)."""
    hidden = case.hidden.clone().requires_grad_(True)
    lm_head = case.lm_head.clone().requires_grad_(True)
    logits = hidden @ lm_head.T
    logits.retain_grad()
    log_softmax = torch.log_softmax(logits[:, :vocab].float(), dim=-1)
    rows, targets = [], []
    doc_start = 0
    for tokens, total, response in zip(case.tokens, case.total_lengths, case.response_lengths, strict=True):
        last_row = doc_start + total - 1  # the last token predicts nothing
        rows.append(torch.arange(last_row - response, last_row, device=logits.device))
        targets.append(tokens[total - response :])
        doc_start += total
    log_probs = log_softmax[torch.cat(rows), torch.cat(targets)]
    loss = -(log_probs * torch.cat(case.loss_masks)).sum()
    loss.backward()
    width = lm_head.size(0) // tp_size
    shard = lm_head.grad[tp_rank * width : (tp_rank + 1) * width].float()
    return _Step(
        log_probs=log_probs.detach(),
        loss=loss.detach(),
        logits_grad_norm=logits.grad.float().norm().item(),
        lm_head_grad_norm=lm_head.grad.float().norm().item(),
        lm_head_grad=shard,
    )


def _differences(actual: _Step, expected: _Step) -> dict[str, float]:
    def rel(a, b):
        return abs(float(a) - float(b)) / max(abs(float(b)), 1e-12)

    return {
        "log_prob_max_abs": (actual.log_probs - expected.log_probs).abs().max().item(),
        "loss_rel": rel(actual.loss, expected.loss),
        "logits_grad_norm_rel": rel(actual.logits_grad_norm, expected.logits_grad_norm),
        "lm_head_grad_norm_rel": rel(actual.lm_head_grad_norm, expected.lm_head_grad_norm),
    }


def _assert_close_step(actual: _Step, expected: _Step, label: str) -> None:
    diffs = _differences(actual, expected)
    print(f"{label}: {diffs}", flush=True)
    assert actual.log_probs.shape == expected.log_probs.shape, label
    assert diffs["log_prob_max_abs"] <= _MAX_LOG_PROB_DIFF, (label, diffs)
    for key in ("loss_rel", "logits_grad_norm_rel", "lm_head_grad_norm_rel"):
        assert diffs[key] < _MAX_REL_DIFF, (label, key, diffs)


def check_backends_agree(
    device, *, tp_rank: int, tp_size: int, group, vocab: int, width: int, recompute: bool, **case_kwargs
) -> None:
    """The checks of one rank; module-level so a CPU dry run can drive it with a stand-in fused op."""
    _set_tp(tp_rank, tp_size, group)
    case = _make_case(device, vocab=vocab, width=width, **case_kwargs)
    reference = _true_vocab_reference(case, vocab=vocab, tp_rank=tp_rank, tp_size=tp_size)
    fused = _sft_step(
        case, backend="fused", recompute=recompute, vocab=vocab, tp_rank=tp_rank, tp_size=tp_size, group=group
    )
    torch_backend = _sft_step(
        case, backend="torch", recompute=False, vocab=vocab, tp_rank=tp_rank, tp_size=tp_size, group=group
    )
    label = f"tp={tp_size} rank={tp_rank} width={width} recompute={recompute}"
    _assert_close_step(fused, reference, f"fused vs true-vocab reference, {label}")
    shard_width = width // tp_size
    first_padding = vocab - tp_rank * shard_width
    if first_padding < shard_width:
        # Megatron's padding rows of lm_head get no gradient, so they stay as loaded
        assert (fused.lm_head_grad[max(first_padding, 0) :] == 0).all(), label
    if width == vocab:
        _assert_close_step(fused, torch_backend, f"fused vs torch backend, {label}")
    else:
        # the torch backend's softmax includes the padding columns: report how far that moves it
        print(f"torch backend vs true-vocab reference, {label}: {_differences(torch_backend, reference)}", flush=True)


def _worker(rank: int, tp_size: int, port: int, width: int, recompute: bool) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    backend = "nccl" if torch.cuda.device_count() >= tp_size else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=tp_size)
    try:
        check_backends_agree(
            "cuda",
            tp_rank=rank,
            tp_size=tp_size,
            group=dist.group.WORLD,
            vocab=_VOCAB,
            width=width,
            recompute=recompute,
        )
    finally:
        dist.destroy_process_group()


def _spawn(tp_size: int, width: int, recompute: bool = False) -> None:
    mp.spawn(_worker, args=(tp_size, find_free_port(), width, recompute), nprocs=tp_size, join=True)


@pytest.mark.parametrize("recompute", [False, True], ids=["inplace_backward", "recompute_loss_function"])
def test_tp1_fused_matches_the_torch_backend(recompute):
    _spawn(1, _VOCAB, recompute)


def test_tp2_vocab_split_fused_matches_the_torch_backend():
    _spawn(2, _VOCAB)


def test_tp8_vocab_padding_on_the_last_shard_is_outside_the_fused_softmax():
    """Two shards of the TP 8 padded width: the last one ends in the 512 padding columns, as on TP rank 7."""
    _spawn(2, _TP8_PADDED_VOCAB)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
