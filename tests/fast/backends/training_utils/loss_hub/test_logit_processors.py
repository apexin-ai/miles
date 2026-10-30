from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.loss.hub import logit_processors


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,responses,padding",
    [([17, 13, 9, 12], [16, 6, 0, 11], 7), ([5, 4], [4, 4], 0), ([7], [0], 0)],
)
def test_packed_response_views_preserve_values_and_gradients(monkeypatch, dtype, lengths, responses, padding):
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False)
    logits = torch.randn(1, sum(lengths) + padding, 16, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    tokens = [torch.arange(length) for length in lengths]
    chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=lengths,
            response_lengths=responses,
            include_response_indices=True,
        )
    )
    expected = []
    end = 0
    for i, (length, response) in enumerate(zip(lengths, responses, strict=True)):
        end += length
        expected.append(reference[0, end - response - 1 : end - 1])
        torch.testing.assert_close(chunks[i][0], expected[-1], rtol=0, atol=0)
        torch.testing.assert_close(chunks[i][1], tokens[i][-response:] if response else tokens[i][:0])
        assert list(chunks[i][2]) == list(range(response))
    sum((i + 1) * chunk[0].square().sum() for i, chunk in enumerate(chunks)).backward()
    sum((i + 1) * chunk.square().sum() for i, chunk in enumerate(expected)).backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)


def test_packed_response_backward_does_not_allocate_a_full_gradient_per_sample(monkeypatch):
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    logits = torch.ones(1, 16 * 17 + 7, 256, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profiler:
        chunks = logit_processors._iter_response_chunks(
            logits,
            args=Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False),
            unconcat_tokens=[torch.arange(17) for _ in range(16)],
            total_lengths=[17] * 16,
            response_lengths=[16] * 16,
            include_response_indices=False,
        )
        sum(chunk.sum() for chunk, _, _ in chunks).backward()
    gradient_allocations = sum(
        event.cpu_memory_usage for event in profiler.events() if event.name == "aten::slice_backward"
    )
    assert gradient_allocations <= 2 * logits.numel() * logits.element_size()


class TestGetLogProbsAndEntropy:
    def test_stored_log_probs_forward_the_debug_grad_switch_from_training_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        """Stored log probabilities forward the debug grad contract from training arguments."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0, group=None),
            cp=SimpleNamespace(rank=0, size=1),
        )
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
        args = Namespace(
            log_probs_backend="torch",
            loss_type="policy_loss",
            qkv_format="thd",
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            debug_unified_grad_fused_logprob=True,
            train_backend="fsdp",
        )

        with torch.no_grad():
            result = logit_processors.get_log_probs_and_entropy(
                torch.randn(1, 4, 8),
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
                total_lengths=[4],
                response_lengths=[3],
            )

        assert fused_cross_entropy_calls == [(True, True)]
        assert not result["log_probs"][0].requires_grad


@pytest.mark.parametrize("recompute_loss_function", [False, True])
def test_fused_sft_loss_scores_the_torch_backend_rows_over_the_true_vocab(monkeypatch, recompute_loss_function):
    """Packed SFT docs with thd padding rows and vocab padding columns: the fused op gets the rows and next
    tokens the torch backend scores, the true vocabulary, and the in-place backward only when the loss is not
    recomputed. The loss mask then weighs response token ``i`` with ``loss_mask[i]``."""
    from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
    from miles.backends.training_utils.loss.hub.losses import sft_loss_function

    fused_log_probs = pytest.importorskip("miles.backends.training_utils.loss.hub.fused_log_probs")
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, size=1, group=object()), cp=SimpleNamespace(rank=0, size=1)
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    calls = []

    def fake_fused_log_probs_and_entropy(logits, rows, targets, **kwargs):
        calls.append((logits, rows, targets, kwargs))
        log_softmax = torch.log_softmax(logits[rows, : kwargs["vocab_size"]].float(), dim=-1)
        return log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1), None

    monkeypatch.setattr(fused_log_probs, "fused_log_probs_and_entropy", fake_fused_log_probs_and_entropy)
    lengths, responses, padding_rows, vocab_size, width = [17, 13, 2, 12], [16, 6, 1, 11], 7, 37, 40
    args = Namespace(
        loss_type="sft_loss",
        log_probs_backend="fused",
        recompute_loss_function=recompute_loss_function,
        qkv_format="thd",
        allgather_cp=False,
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        train_backend="megatron",
        backend=SimpleNamespace(vocab_size=vocab_size),
    )
    logits = torch.randn(1, sum(lengths) + padding_rows, width)
    tokens = [torch.randint(0, vocab_size, (length,)) for length in lengths]
    loss_masks = [torch.randint(0, 2, (response,)).float() for response in responses]
    batch = dict(unconcat_tokens=tokens, total_lengths=lengths, response_lengths=responses, loss_masks=loss_masks)

    loss, _ = sft_loss_function(
        args,
        batch,
        logits,
        get_sum_of_sample_mean(lengths, responses, loss_masks, calculate_per_token_loss=True),
    )

    [(scored_logits, rows, targets, kwargs)] = calls
    torch_chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=lengths,
            response_lengths=responses,
            include_response_indices=False,
        )
    )
    assert torch.equal(scored_logits[rows], torch.cat([chunk for chunk, _, _ in torch_chunks]))
    assert torch.equal(targets, torch.cat([chunk_tokens for _, chunk_tokens, _ in torch_chunks]))
    doc_ends = torch.tensor(lengths).cumsum(0) - 1  # a doc's last token predicts nothing
    assert rows.tolist() == [row for end, n in zip(doc_ends.tolist(), responses) for row in range(end - n, end)]
    assert torch.equal(targets, torch.cat([doc[-n:] for doc, n in zip(tokens, responses, strict=True)]))
    assert kwargs["vocab_size"] == vocab_size
    assert kwargs["temperature"] == 1.0
    assert kwargs["inplace_backward"] is not recompute_loss_function
    assert kwargs["tp_group"] is parallel_state.tp.group
    log_softmax = torch.log_softmax(logits[0, rows, :vocab_size], dim=-1)
    expected = -(log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1) * torch.cat(loss_masks)).sum()
    torch.testing.assert_close(loss, expected)
