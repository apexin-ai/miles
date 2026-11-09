"""Per-row loss-numerator scale for merged SFT steps (apex/pr/qwen38-dynb-merge).

A merged step trains K length buckets of the same logical plan in one optimizer update. Each row r of bucket k
carries ``row_loss_scale`` a_r = N_p / N_k (N_k: the bucket's supervised tokens, N_p: the step's), its
``logical_bucket_id`` and its ``row_instance_id`` (``Sample.metadata``, set by the data source). With the per-token
loss, Megatron divides the step's summed loss by N_p (``count_loss_tokens``, unchanged here), so

    (1 / N_p) * sum_k a_k * sum_{t in k} l_t  =  sum_k (1 / N_k) * sum_{t in k} l_t

i.e. the step keeps every bucket's weight of the unmerged objective. Only the numerator changes; the denominator
(``loss/objective.py`` ``count_loss_tokens``) and every other loss path stay as they are. Rows without the fields
take the original code path (no change at all).

The observer (``set_observer``) lets the training repository account every micro-batch exactly once: ``loss_function``
reports the per-micro-batch denominator it returns to Megatron ("objective"), the SFT loss reports each row's
unweighted token-loss sum and actual mask count ("sft"). Both only when the batch has ``q38_mb_id`` (set by
``train_one_step``'s forward step from ``get_batch``'s micro-batch index); the observer de-duplicates by that id, so a
recomputed loss (``--recompute-loss-function``) counts once. Nothing here raises on bad values: the observer records
them and the after-forward-backward hook stops every rank together.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

ROW_FIELDS = ("row_loss_scale", "logical_bucket_id", "row_instance_id")
MB_ID_KEY = "q38_mb_id"
MB_INDEX_KEY = "q38_mb_index"

_observer: Callable[..., None] | None = None


def set_observer(fn: Callable[..., None] | None) -> Callable[..., None] | None:
    """Install the observer (None removes it); returns the previous one."""
    global _observer
    previous, _observer = _observer, fn
    return previous


def get_observer() -> Callable[..., None] | None:
    return _observer


def has_row_fields(batch: dict[str, Any]) -> bool:
    return batch.get("row_loss_scale") is not None


def notify(kind: str, batch: dict[str, Any], **payload: Any) -> None:
    """Report one micro-batch's contribution to the observer, if there is one and the batch is a train micro-batch."""
    if _observer is not None and batch.get(MB_ID_KEY) is not None:
        _observer(kind, batch, **payload)


def scaled_sft_loss(
    args: Any, batch: dict[str, Any], log_probs: list[torch.Tensor], logits: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SFT loss of a micro-batch whose rows carry ``row_loss_scale``: sum_r a_r * S_r, S_r = -sum_t mask * log p.

    Per-token loss and CP 1 only (each row's response log-probs and loss mask are whole on this rank). The reported
    "loss" is the unweighted sum (the raw token loss after Megatron's division), so the logged train loss keeps its
    meaning; with every a_r = 1 the returned loss is bit-for-bit the original path's (same summation order)."""
    from miles.backends.training_utils.parallel import get_parallel_state

    assert args.backend.calculate_per_token_loss, "row_loss_scale needs --calculate-per-token-loss"
    assert get_parallel_state().cp.size == 1, "row_loss_scale supports context parallel 1 only"
    loss_masks = batch["loss_masks"]
    row_sums = [-(lp * mask).sum() for lp, mask in zip(log_probs, loss_masks, strict=True)]
    device = logits.device
    # A bad field (wrong length, a non-number) must not raise here, on this rank alone, before the step's collectives:
    # train with scale 1 so every rank reaches the after-forward-backward check, and report the problem there.
    scale_values, problem = _scales(batch.get("row_loss_scale"), len(row_sums))
    scale_t = torch.as_tensor(scale_values, dtype=torch.float32, device=device)
    loss = sum(s * a for s, a in zip(row_sums, scale_t.unbind(0), strict=True)) if row_sums else 0
    unweighted = sum(row_sums) if row_sums else 0
    if not torch.is_tensor(loss):
        loss = torch.zeros((), dtype=torch.float32, device=device)
        unweighted = torch.zeros((), dtype=torch.float32, device=device)
    if sum(int(x.numel()) for x in log_probs) == 0:
        loss = loss + 0 * logits.sum(dtype=torch.float32)
    if row_sums:
        sums = torch.stack([s.detach().float() for s in row_sums])
        counts = torch.stack([m.sum().to(torch.int64) for m in loss_masks])
    else:
        sums = torch.zeros(0, dtype=torch.float32, device=device)
        counts = torch.zeros(0, dtype=torch.int64, device=device)
    notify("sft", batch, row_sums=sums, row_counts=counts, problem=problem)
    return loss, {"loss": unweighted.clone().detach()}


def _scales(raw: Any, n: int) -> tuple[list[float], str | None]:
    """(n float scales, None) or (n ones, the problem) for a row_loss_scale field that is not n numbers."""
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError) as e:
        return [1.0] * n, f"row_loss_scale is not a list of numbers: {e!r}"[:300]
    if len(values) != n:
        return [1.0] * n, f"row_loss_scale has {len(values)} values for {n} rows"
    return values, None
