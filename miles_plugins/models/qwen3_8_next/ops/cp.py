"""allgather-CP helpers for Qwen3.8-Next.

Under ``--allgather-cp`` the data pipeline concatenates the packed samples, pads the global
stream to a multiple of ``cp_size * pad_size``, and slices it once: rank ``r`` owns the
contiguous global token range ``[r * T_local, (r + 1) * T_local)`` while ``cu_seqlens``
carries the global document boundaries, with the padding appended as its own segment
(``miles/backends/training_utils/data/rollout.py``, the branch commented "DSA mode:
concatenate all sequences first, then slice once with CP").

That single affine map is why CP is expressible for this model at all. The QSA indexer and
PLE both need whole-sequence order, and under the default zigzag layout each rank holds two
disjoint ranges per sequence, where no affine local-to-global map exists.
``get_qwen3_8_next_spec`` refuses that layout rather than computing a wrong answer.

The global padding is also what makes ``local_rows`` safe: sample lengths themselves are
not multiples of ``cp_size`` (the CP validation corpus has lengths like 262,134), but the
concatenated stream is padded before it is sliced.
"""

import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from torch import Tensor


def cp_world_size() -> int:
    """Context-parallel world size, 1 when CP is off.

    ``parallel_state.get_context_parallel_world_size()`` returns 0, not 1, when
    torch.distributed is not initialized, which is the normal case in the CPU tests. Treat
    that as 1 so a size of 0 can never reach the arithmetic below.
    """
    return parallel_state.get_context_parallel_world_size() or 1


def cp_rank_of() -> int:
    """This rank's index in the context-parallel group, 0 when CP is off."""
    if cp_world_size() <= 1:
        return 0
    return parallel_state.get_context_parallel_rank()


def cp_enabled(config) -> bool:
    """True when this forward must reconstruct the whole sequence across CP ranks."""
    if cp_world_size() <= 1:
        return False
    # Set by get_qwen3_8_next_spec, which refuses CP under any other layout. Absent on the
    # generic test configs that build a layer directly.
    allgather_cp = getattr(
        config, "qwen3_8_next_allgather_cp", False
    )  # config-access-exempt: set only by the Qwen3.8 spec
    if not allgather_cp:
        raise RuntimeError(
            "Qwen3.8-Next reached a CP forward without the allgather-CP layout. "
            "get_qwen3_8_next_spec sets qwen3_8_next_allgather_cp; reaching here means the "
            "layer was built outside it, and the whole-sequence order this layer assumes "
            "does not hold."
        )
    return True


def local_rows(total: int, cp_size: int | None = None, cp_rank: int | None = None) -> tuple[int, int]:
    """This rank's contiguous ``[lo, hi)`` rows of a ``total``-token global packed stream."""
    size = cp_world_size() if cp_size is None else cp_size
    rank = parallel_state.get_context_parallel_rank() if cp_rank is None else cp_rank
    if total % size:
        raise RuntimeError(
            f"allgather-CP slices the packed stream into {size} equal pieces, but the global "
            f"length {total} is not a multiple of it. rollout.py pads the concatenated stream "
            "to a multiple of cp_size before slicing, so an unpadded length here means the "
            "batch did not come through that path."
        )
    local = total // size
    return rank * local, (rank + 1) * local


def gather_sequence(x: Tensor) -> Tensor:
    """Concatenate the CP ranks' contiguous pieces back into the global sequence.

    The backward reduce-scatters (SUM), which is what a gathered key set needs: every
    rank's queries read every rank's keys, so each piece's gradient is the sum of the
    contributions all ranks made to it.
    """
    return gather_from_sequence_parallel_region(
        x, tensor_parallel_output_grad=True, group=parallel_state.get_context_parallel_group()
    )


def gather_sequence_no_grad(x: Tensor) -> Tensor:
    """``gather_sequence`` for tensors that carry no gradient (token ids, indexer inputs)."""
    group = parallel_state.get_context_parallel_group()
    size = torch.distributed.get_world_size(group=group)
    out = x.new_empty((x.shape[0] * size, *x.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def compose_row_shard(
    cp_rows: tuple[int, int] | None, tp_shard: tuple[int, int] | None, total: int
) -> tuple[tuple[int, int] | None, int]:
    """Combine the CP row window with the indexer's TP row stride.

    The indexer shards its query rows two independent ways and both are on in production:

    * CP gives this rank a *contiguous* window ``[lo, hi)`` of the whole sequence -- the
      rows it holds queries for.
    * ``--qwen38-indexer-tp-shard`` gives each TP rank every ``n``-th row -- a *stride*.

    Composed, a rank scores ``lo + tp_rank, lo + tp_rank + n, ...`` up to ``hi``, which is
    the stride restricted to the window. Expressed as a stride over the whole sequence that
    is exactly ``(lo + tp_rank, n)``, so the two compose into one ``row_shard`` with no new
    indexing mode -- but the row COUNT has to come from the window, not from the sequence.

    Returns ``(row_shard, rows)``: the pair to hand ``iter_selected_blocks`` and how many
    rows it will yield, i.e. the first dimension of this rank's block buffer.
    """
    lo, hi = (0, total) if cp_rows is None else cp_rows
    span = hi - lo
    if tp_shard is None:
        return (None if cp_rows is None else (lo, 1)), span
    tp_rank, n_shards = tp_shard
    # rows of the window this TP rank takes: ceil((span - tp_rank) / n), clamped at 0
    rows = max(0, -(-(span - tp_rank) // n_shards)) if span > tp_rank else 0
    return (lo + tp_rank, n_shards), rows
