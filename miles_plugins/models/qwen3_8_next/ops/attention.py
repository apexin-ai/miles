"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

Each full-attention layer keeps ``indexer_budget`` key tokens per query and attends
only those.
"""

import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles.kernels.attention.qsa import qsa_block_sparse_attention, qsa_sparse_attention
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout, Qwen38NextQSAIndexer


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
        block_form = getattr(
            self._owner, "_qsa_block_form", None
        )  # config-access-exempt: the owner attaches sparse-attention state only during its forward
        if query.dim() == 3:
            if block_form is None:
                raise RuntimeError(
                    "QSA core attention ran on a thd query with no block form published. "
                    "Qwen38NextAttention.forward sets it before delegating; reaching here "
                    "means core_attention was called out of band."
                )
            sel_bitmap, lo, hi, blk_base, tok_base, block_first, block_last, blk = block_form
            return qsa_block_sparse_attention(
                query,
                key,
                value,
                sel_bitmap,
                lo,
                hi,
                blk_base,
                tok_base,
                block_first,
                block_last,
                self.softmax_scale,
                blk,
            ).reshape(query.shape[0], -1)

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
        self._qsa_block_form = None

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
        else:
            self._qsa_cu_seqlens = None
            positions = torch.arange(seq, device=hidden_states.device)
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        if rotary_pos_emb is None:
            raise RuntimeError("the QSA indexer reuses the attention RoPE, but no rotary_pos_emb was passed")
        materialized_rope = getattr(
            self.config, "qwen3_8_next_materialized_rope", False
        )  # config-access-exempt: set only by the Qwen3.8 spec, absent on generic test configs
        with torch.no_grad():
            if packed is not None:
                # thd: the kernel reads only the block bitmap, built tile by tile
                self._qsa_selection = None
                self._qsa_block_form = qsa_block_form(
                    self.indexer,
                    indexer_states[:, 0],
                    positions,
                    rotary_pos_emb,
                    self._qsa_cu_seqlens,
                    materialized_rope=materialized_rope,
                )
            else:
                self._qsa_selection = self._token_selection(
                    indexer_states[:, 0], positions, rotary_pos_emb, materialized_rope
                )
                self._qsa_block_form = None
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
            self._qsa_block_form = None

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


def qsa_block_form(
    indexer: Qwen38NextQSAIndexer,
    indexer_states: Tensor,
    positions: Tensor,
    rotary_pos_emb: Tensor,
    cu_seqlens: Tensor | None,
    materialized_rope: bool = False,
):
    """Indexer selection -> the block form the tensor-core kernel consumes.

    The bitmap marks each query's indexer blocks plus the block holding the query while
    that block is still partial (the indexer only ranks complete blocks). Block ids come
    from the per-sequence grid (``PackedBlockLayout``); a ceil(seq_start / ratio) base
    collides across sequences for some lengths.
    """
    ratio = indexer.compress_ratio
    seq = positions.shape[0]
    device = positions.device
    cu = cu_seqlens
    if cu is None or cu.numel() <= 2:
        cu = torch.tensor([0, seq], device=device)
    layout = PackedBlockLayout(cu, positions, ratio)
    num_blocks = layout.num_blocks

    sel_bitmap = torch.zeros(seq, num_blocks, dtype=torch.uint8, device=device)
    for start, end, block_idx in indexer.iter_selected_blocks(
        indexer_states, positions, rotary_pos_emb, cu_seqlens, materialized_rope=materialized_rope
    ):
        valid = block_idx >= 0
        rows = torch.arange(start, end, device=device).unsqueeze(1).expand_as(block_idx)
        sel_bitmap[rows[valid], block_idx[valid]] = 1
    partial = (positions + 1) % ratio != 0
    sel_bitmap[torch.arange(seq, device=device)[partial], layout.token_block[partial]] = 1

    seq_start = torch.arange(seq, device=device) - positions
    blk_base = layout.token_block_start.to(torch.int32)
    tok_base = layout.token_start.to(torch.int32)
    hi = (positions + seq_start).to(torch.int32)
    block_first = layout.block_first_token.to(torch.int32)
    block_last = layout.block_last_token.to(torch.int32)
    return sel_bitmap, tok_base, hi, blk_base, tok_base, block_first, block_last, ratio
