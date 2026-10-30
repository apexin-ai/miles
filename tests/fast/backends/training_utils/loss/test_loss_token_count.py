import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss.objective import count_loss_tokens, loss_function
from miles.backends.training_utils.metrics import log_utils

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_VOCAB_SIZE = 13
# (prompt length, response loss mask) per real sample
_REAL_SAMPLES = ((3, [1, 1, 0, 1]), (2, [1, 1, 1]), (4, [0, 1, 1, 1, 1]))
_SENTINEL = (5, [0, 0, 0, 0])


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    # the vocab-parallel cross entropy all-reduces over its (one-rank) TP group
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("loss-token-count") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _sample(prompt_len: int, mask: list[int], seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "tokens": torch.randint(_VOCAB_SIZE, (prompt_len + len(mask),), generator=generator),
        "loss_mask": torch.tensor(mask),
    }


def _micro_batch(samples: list[dict]) -> dict:
    return {
        "unconcat_tokens": [sample["tokens"] for sample in samples],
        "loss_masks": [sample["loss_mask"] for sample in samples],
        "total_lengths": [sample["tokens"].numel() for sample in samples],
        "response_lengths": [sample["loss_mask"].numel() for sample in samples],
    }


def _step_grad(micro_batches: list[dict], megatron_scaling: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """One per-token step: Megatron sums the returned normalizers and divides the summed grads by them
    (finalize_model_grads); the other backends divide each loss by the step's token count up front."""
    make_parallel_state()
    args = make_args(loss_type="sft_loss", true_on_policy_mode=False, calculate_per_token_loss=True)
    torch.manual_seed(0)
    weight = torch.randn(_VOCAB_SIZE, _VOCAB_SIZE, requires_grad=True)
    step_tokens = count_loss_tokens([mask for batch in micro_batches for mask in batch["loss_masks"]])
    normalizer_sum = torch.zeros((), dtype=torch.int)
    for batch in micro_batches:
        logits = weight[torch.cat(batch["unconcat_tokens"])].unsqueeze(0)
        loss, normalizer, _ = loss_function(
            args=args,
            batch=batch,
            num_microbatches=len(micro_batches),
            logits=logits,
            apply_megatron_loss_scaling=megatron_scaling,
            num_step_tokens=None if megatron_scaling else step_tokens,
        )
        loss.backward()
        normalizer_sum += normalizer
    grad = weight.grad * (1.0 / torch.clamp(normalizer_sum, min=1)) if megatron_scaling else weight.grad
    return grad, normalizer_sum


class TestCountLossTokens:
    def test_a_fully_masked_sample_adds_nothing_to_the_token_count(self) -> None:
        """The per-token denominator is the real supervised-token count, so a sentinel sample does not dilute it."""
        masks = [torch.tensor([1, 1, 0, 1]), torch.tensor([0, 0, 0]), torch.tensor([1, 1])]

        assert count_loss_tokens(masks).item() == 5

    def test_a_batch_without_fully_masked_samples_counts_as_before(self) -> None:
        """Batches where every sample has a supervised token keep the old normalizer exactly."""
        generator = torch.Generator().manual_seed(0)
        masks = [torch.randint(2, (n,), generator=generator) for n in (7, 3, 11, 5)]
        masks = [torch.cat([mask, torch.ones(1, dtype=mask.dtype)]) for mask in masks]
        old_count = sum(torch.clamp_min(mask.sum(), 1) for mask in masks)

        assert torch.equal(count_loss_tokens(masks), old_count)

    def test_a_globally_unsupervised_step_counts_zero_tokens(self) -> None:
        """The step-level zero count is what lets the trainers recognise a step with nothing to learn."""
        assert count_loss_tokens([torch.zeros(4, dtype=torch.long), torch.zeros(2, dtype=torch.long)]).item() == 0


class TestFullyMaskedSampleInvariance:
    @pytest.mark.parametrize("megatron_scaling", [True, False])
    @pytest.mark.parametrize("placement", ["own_micro_batch", "packed_with_real_samples"])
    def test_a_fully_masked_sample_leaves_the_step_gradient_bitwise_unchanged(
        self, process_group: None, megatron_scaling: bool, placement: str
    ) -> None:
        """Same real samples in the same step give the same normalizer and the same gradient bits,
        whether or not a padding/sentinel sample also takes part in the step."""
        real = [_sample(prompt_len, mask, seed) for seed, (prompt_len, mask) in enumerate(_REAL_SAMPLES)]
        sentinel = _sample(*_SENTINEL, seed=99)
        without = [_micro_batch(real[:2]), _micro_batch(real[2:])]
        if placement == "own_micro_batch":
            with_sentinel = [*without, _micro_batch([sentinel])]
        else:
            with_sentinel = [_micro_batch(real[:2]), _micro_batch([real[2], sentinel])]

        grad_without, tokens_without = _step_grad(without, megatron_scaling)
        grad_with, tokens_with = _step_grad(with_sentinel, megatron_scaling)

        num_real_tokens = sum(sum(mask) for _, mask in _REAL_SAMPLES)
        if megatron_scaling:
            assert tokens_without.item() == tokens_with.item() == num_real_tokens
        assert torch.equal(grad_with, grad_without)


class TestUnsupervisedStepMetrics:
    def test_a_step_without_supervised_tokens_reports_no_per_token_means(self, monkeypatch) -> None:
        """Zero summed over zero tokens has no mean: the skipped step logs no loss instead of dividing by zero."""
        make_parallel_state()
        monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", staticmethod(lambda *args, **kwargs: None))
        losses = [{"keys": ["loss"], "values": torch.tensor([0.0, 0.0])} for _ in range(2)]

        assert log_utils.aggregate_train_losses(losses) == {}
