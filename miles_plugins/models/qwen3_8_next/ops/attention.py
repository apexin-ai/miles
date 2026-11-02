"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

Each full-attention layer keeps ``indexer_budget`` key tokens per query and attends
only those.
"""

import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles.kernels.attention.qsa import build_qsa_block_index, qsa_block_attention, qsa_sparse_attention
from miles_plugins.models.qwen3_8_next.ops.cp import (
    SINGLE_RANK_GROUP,
    compose_row_shard,
    cp_enabled,
    cp_local_packed_seq_params,
    cp_local_rope,
    gather_sequence,
    gather_sequence_no_grad,
    local_rows,
)
from miles_plugins.models.qwen3_8_next.ops.indexer_monitor import get_monitor
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer, indexer_rope_table


class Qwen38NextQSACoreAttention(MegatronModule):
    """Core attention restricted to the indexer's selection."""

    def __init__(self, config, layer_number: int, owner):
        super().__init__(config)
        self.layer_number = layer_number
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        selection = getattr(
            self._owner, "_qsa_selection", None
        )  # config-access-exempt: the owner attaches sparse-attention state only during its forward
        selected = getattr(
            self._owner, "_qsa_blocks", None
        )  # config-access-exempt: the owner attaches sparse-attention state only during its forward
        if query.dim() == 3:
            # thd reads only the selected block ids; there is no token selection on this path
            if selected is None:
                raise RuntimeError(
                    "QSA core attention ran on a thd query with no selected blocks published. "
                    "Qwen38NextAttention.forward sets them before delegating; reaching here "
                    "means core_attention was called out of band."
                )
            blocks, cu_seqlens = selected
            ratio = self._owner.compress_ratio
            if cp_enabled(self.config):
                # The block ids address the whole sequence, so the keys have to be the whole
                # sequence too; the queries stay this rank's rows and the kernel takes the query
                # and key extents separately. The gather's backward reduce-scatters, summing what
                # every rank's queries read from each piece.
                key, value = gather_sequence(key), gather_sequence(value)
            monitor = get_monitor()
            # the recompute pass would measure the same rows twice
            if monitor.enabled and not (is_checkpointing() and torch.is_grad_enabled()):
                monitor.maybe_record(
                    self.layer_number,
                    query,
                    key,
                    blocks,
                    cu_seqlens,
                    ratio,
                    self.softmax_scale,
                    self._owner.indexer.block_topk,
                )
            index = build_qsa_block_index(
                blocks,
                cu_seqlens,
                compress_ratio=ratio,
                num_heads=query.shape[1],
                num_kv_heads=key.shape[1],
            )
            return qsa_block_attention(query, key, value, index, self.softmax_scale).reshape(query.shape[0], -1)

        if query.dim() != 4:
            raise RuntimeError(f"QSA core attention expected a 3D (thd) or 4D (sbhd) query, got {tuple(query.shape)}")
        if selection is None:
            raise RuntimeError(
                "QSA core attention ran on an sbhd query with no selection published. "
                "Qwen38NextAttention.forward sets it before delegating; reaching here "
                "means core_attention was called out of band."
            )

        s, b, hq, d = query.shape
        out = [
            qsa_sparse_attention(query[:, i], key[:, i], value[:, i], selection, self.softmax_scale) for i in range(b)
        ]
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    # ``verify_allgather_cp_support`` rejects --allgather-cp for every CP attention layer that
    # does not opt in, because stock Megatron attention reads the zigzag layout and would
    # compute silently wrong results on a contiguous slice. This layer reads the contiguous
    # one: it gathers the whole sequence for the indexer and for the keys, and restricts only
    # the query rows to this rank's window (``ops/cp.py``). Zigzag is refused earlier, in
    # ``get_qwen3_8_next_spec``, so opting in here cannot enable the wrong layout.
    supports_allgather_cp = True

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if self.checkpoint_core_attention:
            # the selection is cleared after forward, so a backward-time core_attn recompute would find none
            raise NotImplementedError(
                "Qwen3.8-Next QSA does not support selective core_attn recompute; "
                "use full recompute or drop core_attn from --recompute-modules"
            )
        if config.rotary_interleaved:
            raise NotImplementedError("the QSA indexer mirrors sglang's neox RoPE; rotary_interleaved is unsupported")
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.core_attention = Qwen38NextQSACoreAttention(config, layer_number, owner=self)
        self._qsa_selection = None
        self._qsa_blocks = None
        self._qsa_cp_rows = None

    @staticmethod
    def _packed_positions(cu_seqlens: Tensor, total: int) -> Tensor:
        """Positions restarting at 0 for each sequence in a packed batch."""
        idx = torch.arange(total, device=cu_seqlens.device)
        starts = cu_seqlens[:-1].long()
        seg = torch.zeros(total, dtype=torch.long, device=cu_seqlens.device)
        seg[starts[1:]] = 1
        seg = seg.cumsum(0)
        return idx - starts[seg]

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        packed = kwargs.get("packed_seq_params")

        indexer_states = hidden_states
        if getattr(
            self.config, "sequence_parallel", False
        ):  # config-access-exempt: model configs may omit sequence_parallel
            if get_tensor_model_parallel_world_size() > 1:
                with torch.no_grad():
                    indexer_states = gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=get_tensor_model_parallel_group(),
                    )
        seq = indexer_states.shape[0]
        cp = cp_enabled(self.config)
        if cp:
            if packed is None:
                raise NotImplementedError(
                    "Qwen3.8-Next QSA under context parallelism needs the packed (thd) layout: "
                    "the indexer has to know the global document boundaries to place a gathered "
                    "key set. Run with --qkv-format thd."
                )
            # The indexer runs under no_grad and selects over whole-sequence natural order, so it
            # needs every rank's tokens; the rows it scores stay this rank's own (cp_rows below).
            with torch.no_grad():
                indexer_states = gather_sequence_no_grad(indexer_states)
            self._qsa_cp_rows = local_rows(indexer_states.shape[0])
            seq = indexer_states.shape[0]
        else:
            self._qsa_cp_rows = None
        if packed is not None:
            cu = getattr(
                packed, "cu_seqlens_q", None
            )  # config-access-exempt: packed-sequence implementations may omit cumulative lengths
            if cu is None:
                raise NotImplementedError(
                    "packed_seq_params without cu_seqlens_q: QSA needs the sequence "
                    "boundaries to place positions and to keep attention inside a "
                    "document."
                )
            self._qsa_cu_seqlens = cu
            positions = self._packed_positions(cu, seq)
            # miles' get_batch keeps the same boundaries on the host: the indexer plans its tiles from them
            cu_host = getattr(
                packed, "cu_seqlens_host", None
            )  # config-access-exempt: only miles' PackedSeqParamsWithHostCuSeqlens carries the host copy
        else:
            self._qsa_cu_seqlens = None
            positions = torch.arange(seq, device=hidden_states.device)
            cu_host = None
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        if rotary_pos_emb is None:
            raise RuntimeError("the QSA indexer reuses the attention RoPE, but no rotary_pos_emb was passed")
        # the attention layer keeps raw mRoPE freqs (fused kernel); the indexer reads the materialized table
        rotary_pos_emb = indexer_rope_table(rotary_pos_emb, self.config)
        materialized_rope = getattr(
            self.config, "qwen3_8_next_materialized_rope", False
        )  # config-access-exempt: set only by the Qwen3.8 spec, absent on generic test configs
        with torch.no_grad():
            if packed is not None:
                # thd: the kernel reads only the selected block ids, written tile by tile
                self._qsa_selection = None
                self._qsa_blocks = qsa_selected_blocks(
                    self.indexer,
                    indexer_states[:, 0],
                    positions,
                    rotary_pos_emb,
                    self._qsa_cu_seqlens,
                    materialized_rope=materialized_rope,
                    tp_group=self._indexer_tp_group(),
                    cu_seqlens_host=cu_host,
                    cp_rows=self._qsa_cp_rows,
                )
            else:
                self._qsa_selection = self._token_selection(
                    indexer_states[:, 0], positions, rotary_pos_emb, materialized_rope
                )
                self._qsa_blocks = None
        original_cp_group = None
        if cp:
            # SelfAttention applies RoPE with apply_rotary_pos_emb(cp_group=pg_collection.cp), which
            # maps a THD token to its position with the zigzag split; this rank holds a contiguous
            # slice. Hand it this rank's rows of the frequencies, one packed segment of those rows
            # and a CP group of one: every local token then gets the frequencies the CP1 run gives
            # it. The core attention reads the global boundaries kept above, not this packing.
            # With a CP group of one, SelfAttention's CP layout conversion is a no-op too.
            kwargs["rotary_pos_emb"] = cp_local_rope(
                kwargs["rotary_pos_emb"], positions, self._qsa_cp_rows, self.config
            )
            lo, hi = self._qsa_cp_rows
            # the query rows after SelfAttention's projection (SP gathers over TP, not CP)
            kwargs["packed_seq_params"] = cp_local_packed_seq_params(packed, hi - lo)
            original_cp_group = self.pg_collection.cp
            self.pg_collection.cp = SINGLE_RANK_GROUP
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            if original_cp_group is not None:
                self.pg_collection.cp = original_cp_group
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
            self._qsa_blocks = None
            self._qsa_cp_rows = None

    def _indexer_tp_group(self):
        """The TP group the indexer splits its query rows over, or None (every rank scores every row).

        --qwen38-indexer-tp-shard: the indexer input is the full sequence on every TP rank (SP-gathered), so the
        replicated indexer scores the same rows TP times. Sharded, each rank scores every TP-th row and the
        selections are all-gathered; every rank ends with the same ``[T, K]`` block ids.
        """
        if not getattr(
            self.config, "qwen3_8_next_indexer_tp_shard", False
        ):  # config-access-exempt: optional knob, absent from older configs
            return None
        if get_tensor_model_parallel_world_size() <= 1:
            return None
        return get_tensor_model_parallel_group()

    def _token_selection(
        self, indexer_states: Tensor, positions: Tensor, rotary_pos_emb: Tensor, materialized_rope: bool
    ) -> Tensor:
        """Indexer tokens plus the query's own partial block, ``[T, budget + ratio]``, ``-1`` unused."""
        seq = positions.shape[0]
        selection = self.indexer(
            indexer_states,
            positions,
            rotary_pos_emb,
            cu_seqlens=self._qsa_cu_seqlens,
            materialized_rope=materialized_rope,
        )
        seq_start = torch.arange(seq, device=positions.device) - positions
        r = self.compress_ratio
        tail_in_seq = (positions + 1) // r * r
        offs = torch.arange(r, device=positions.device)
        tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)  # in-seq
        tail_idx = seq_start.unsqueeze(1) + tail_pos  # pack index
        tail_idx = torch.where(
            tail_pos <= positions.unsqueeze(1),
            tail_idx,
            torch.full_like(tail_idx, -1),
        )
        merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
        pack_pos = torch.arange(seq, device=positions.device).unsqueeze(1)
        seg_lo = seq_start.unsqueeze(1)
        ok = (merged >= seg_lo) & (merged <= pack_pos) & (merged >= 0)
        return torch.where(ok, merged, torch.full_like(merged, -1))


def qsa_selected_blocks(
    indexer: Qwen38NextQSAIndexer,
    indexer_states: Tensor,
    positions: Tensor,
    rotary_pos_emb: Tensor,
    cu_seqlens: Tensor | None,
    materialized_rope: bool = False,
    tp_group=None,
    cu_seqlens_host: tuple[int, ...] | None = None,
    cp_rows: tuple[int, int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Indexer selection -> ``(blocks, cu_seqlens)``, the input of ``build_qsa_block_index``.

    ``blocks`` is ``[T, K]`` int32 with ``K = min(block_topk, blocks in the batch)``: per query the
    distinct whole blocks the indexer picked, as ids on the per-sequence grid (``PackedBlockLayout``;
    a ceil(seq_start / ratio) base collides across sequences for some lengths), ``-1`` where unused.
    The query's own partial block is not listed: ``build_qsa_block_index`` adds every query's causal
    tail itself, and a tail id here would be a block the kernel ignores. The ids are written tile by
    tile into one buffer, so neither a ``[T, token_budget]`` token selection nor a ``[T, blocks]``
    bitmap is built. ``cu_seqlens_host`` (the same boundaries on the host) spares the indexer's tile
    plan its device reads, see ``iter_selected_blocks``.

    With ``tp_group`` each of its ``n`` ranks scores the rows ``rank, rank + n, ...`` and the shards are
    all-gathered and interleaved back; a row's selection does not depend on the other rows, so the result is the
    replicated one.

    ``cp_rows = (lo, hi)`` is the contiguous window of the whole sequence this context-parallel rank
    holds queries for; ``indexer_states`` and ``positions`` still cover the whole sequence, so every
    block stays a candidate and the selection equals the unsharded one for those rows. The returned
    ``blocks`` then has ``hi - lo`` rows, matching this rank's queries. Composed with ``tp_group``,
    the two shardings become one strided-within-the-window row set (``ops/cp.py``).
    """
    seq = positions.shape[0]
    device = positions.device
    cu = cu_seqlens
    if cu is None or cu.numel() <= 2:
        cu = torch.tensor([0, seq], dtype=torch.int32, device=device)
    tp_shard = None
    if tp_group is not None:
        n_shards = torch.distributed.get_world_size(tp_group)
        tp_shard = (torch.distributed.get_rank(tp_group), n_shards)
    row_shard, _ = compose_row_shard(cp_rows, tp_shard, seq)
    cp_lo, cp_hi = (0, seq) if cp_rows is None else cp_rows
    row_stop = None if cp_rows is None else cp_hi
    # every shard padded to the longest (the one whose first row is the window's first)
    rows = cp_hi - cp_lo if tp_shard is None else -(-(cp_hi - cp_lo) // tp_shard[1])
    width = indexer_block_width(indexer, positions, cu_seqlens, cu_seqlens_host)
    blocks = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for start, end, block_idx in indexer.iter_selected_blocks(
        indexer_states,
        positions,
        rotary_pos_emb,
        cu_seqlens,
        materialized_rope=materialized_rope,
        row_shard=row_shard,
        row_stop=row_stop,
        cu_seqlens_host=cu_seqlens_host,
    ):
        blocks[start:end] = block_idx
    if tp_shard is not None:
        n_shards = tp_shard[1]
        gathered = torch.empty((n_shards * rows, width), dtype=torch.int32, device=device)
        torch.distributed.all_gather_into_tensor(gathered, blocks, group=tp_group)
        # [shard, i] holds query shard + i * n_shards, counted from the window's first row
        blocks = gathered.view(n_shards, rows, width).transpose(0, 1).reshape(n_shards * rows, width)[: cp_hi - cp_lo]
    return blocks, cu


def indexer_block_width(
    indexer: Qwen38NextQSAIndexer,
    positions: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_host: tuple[int, ...] | None = None,
) -> int:
    """``K`` of ``qsa_selected_blocks``: ``min(block_topk, blocks in the batch)``."""
    layout_blocks = indexer.num_blocks(positions, cu_seqlens, cu_seqlens_host)
    return min(indexer.block_topk, layout_blocks)
