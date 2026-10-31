"""Qwen3.8-Next QSA (Qwen Sparse Attention) indexer.

Runs on the full-attention layers and picks, per query token, which
``indexer_budget`` key tokens the sparse attention will look at.
"""

import math
import os

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.indexer import freeze_indexer_parameters


def _indexer_acc_dtype(x):
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def gemma_rmsnorm_last_dim(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last dim with a Gemma-style ``1 + weight`` scale."""
    acc = _indexer_acc_dtype(x)
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def compress_keys_by_mean(token_k: Tensor, compress_ratio: int) -> Tensor:
    """``[T, head_dim] -> [ceil(T / r), head_dim]`` by averaging each run of ``r``."""
    tokens, dim = token_k.shape
    blocks = -(-tokens // compress_ratio)
    padded = blocks * compress_ratio
    if padded != tokens:
        pad = token_k.new_zeros(padded - tokens, dim)
        counts = token_k.new_ones(padded, 1)
        counts[tokens:] = 0
        summed = torch.cat([token_k, pad], dim=0).view(blocks, compress_ratio, dim).sum(1)
        denom = counts.view(blocks, compress_ratio, 1).sum(1).clamp_min(1)
        return summed / denom
    return token_k.view(blocks, compress_ratio, dim).mean(dim=1)


def apply_indexer_rope(x: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
    """Neox RoPE on the leading ``rotary_dim`` dims of ``x`` ``[T, heads, head_dim]``.

    ``rotary_pos_emb`` is Megatron's non-interleaved ``[max_pos, 1, 1, rotary_dim]`` fp32
    table, ``cat(freqs, freqs)``. Mirrors sglang's eager ``apply_rotary_emb``: cos/sin
    rounded to ``x.dtype`` first, then per-op rounding.
    """
    freqs = rotary_pos_emb.flatten(1)
    half = freqs.shape[-1] // 2
    # cos/sin over the contiguous table, then gather, like sglang's cache: strided or gathered cos can move an ulp
    freqs = freqs[:, :half].contiguous()
    cos = freqs.cos()[positions.long()].unsqueeze(-2).to(x.dtype)
    sin = freqs.sin()[positions.long()].unsqueeze(-2).to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]], dim=-1)


def block_causal_mask(query_positions: Tensor, num_blocks: int, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool: which compressed blocks a query may attend to."""
    blocks = torch.arange(num_blocks, device=query_positions.device)
    first_invalid = (query_positions + 1) // compress_ratio
    return blocks.unsqueeze(0) < first_invalid.unsqueeze(1)


class PackedBlockLayout:
    """Per-sequence compressed-block grid for a packed (thd) batch.

    The rollout engine scores one request at a time, so each sequence's block grid
    must start at its own first token.
    """

    def __init__(self, cu_seqlens: Tensor, positions: Tensor, compress_ratio: int):
        device = positions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        lengths = cu[1:] - cu[:-1]
        blocks_per_seq = -(-lengths // compress_ratio)
        self.num_blocks = int(blocks_per_seq.sum())

        seq_block_start = torch.cat([blocks_per_seq.new_zeros(1), blocks_per_seq.cumsum(0)[:-1]])
        seq_token_start = cu[:-1]

        total = positions.numel()
        seg = torch.zeros(total, dtype=torch.long, device=device)
        seg[seq_token_start[1:]] = 1
        seg = seg.cumsum(0)

        # per token
        self.token_seq = seg
        self.token_block = seq_block_start[seg] + positions // compress_ratio
        self.token_block_start = seq_block_start[seg]
        self.token_start = seq_token_start[seg]
        # per block
        self.block_seq = torch.repeat_interleave(torch.arange(blocks_per_seq.numel(), device=device), blocks_per_seq)
        self.block_local = torch.arange(self.num_blocks, device=device) - seq_block_start[self.block_seq]
        self.block_first_token = seq_token_start[self.block_seq] + self.block_local * compress_ratio
        self.block_last_token = torch.minimum(self.block_first_token + compress_ratio, cu[1:][self.block_seq]) - 1


def compress_keys_by_mean_packed(token_k: Tensor, layout: PackedBlockLayout) -> Tensor:
    """``[T, head_dim] -> [layout.num_blocks, head_dim]``, averaging inside each sequence."""
    acc = _indexer_acc_dtype(token_k)
    dim = token_k.shape[-1]
    summed = torch.zeros(layout.num_blocks, dim, dtype=acc, device=token_k.device)
    summed.index_add_(0, layout.token_block, token_k.to(acc))
    counts = torch.zeros(layout.num_blocks, 1, dtype=acc, device=token_k.device)
    counts.index_add_(0, layout.token_block, torch.ones(token_k.shape[0], 1, dtype=acc, device=token_k.device))
    return (summed / counts.clamp_min(1)).to(token_k.dtype)


def packed_block_causal_mask(query_positions: Tensor, layout: PackedBlockLayout, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool, restricted to each query's own sequence."""
    blocks = torch.arange(layout.num_blocks, device=query_positions.device).unsqueeze(0)
    lo = layout.token_block_start.unsqueeze(1)
    first_invalid = lo + ((query_positions + 1) // compress_ratio).unsqueeze(1)
    return (blocks >= lo) & (blocks < first_invalid)


# working set one indexer query tile may hold; the full [T, T / 4, 4 heads] fp32 score is 4 T^2 bytes
DEFAULT_TILE_BYTES = int(os.environ.get("MILES_QWEN38_INDEXER_TILE_BYTES", str(1 << 30)))


def tile_bytes_per_cell(n_heads: int) -> int:
    """Bytes held per (query row, visible block) while a tile is scored.

    The fp32 ``[rows, blocks, heads]`` score plus the summed fp32 logits. Once the score is
    freed, the masks and the tie-break in ``select_topk_blocks`` hold at most 17 bytes per
    cell, so fewer than 4 heads still cost 20.
    """
    return 4 * max(n_heads, 4) + 4


def plan_query_tiles(
    block_lo: list[int], block_hi: list[int], tile_bytes: int, bytes_per_cell: int, row_bytes: int = 0
) -> list[tuple[int, int]]:
    """Split query rows into ``[start, end)`` tiles whose working set fits ``tile_bytes``.

    Row ``i`` may only see blocks ``[block_lo[i], block_hi[i])``; both are nondecreasing
    in ``i``. A tile ``[s, e)`` scores blocks ``[block_lo[s], block_hi[e - 1])`` and costs
    ``(e - s) * (width * bytes_per_cell + row_bytes)``.

    The budget binds every tile of two or more rows. A tile never has fewer than one row,
    so a single row wider than the budget is scored alone: 1.25 MiB of score for a
    262K-token row with 4 heads.
    """
    if tile_bytes <= 0:
        raise ValueError(f"indexer tile budget must be positive, got {tile_bytes} bytes")
    rows = len(block_lo)

    def fits(start: int, n: int) -> bool:
        width = max(1, block_hi[start + n - 1] - block_lo[start])
        return n * (width * bytes_per_cell + row_bytes) <= tile_bytes

    tiles = []
    start = 0
    while start < rows:
        # largest n with fits(start, n); fits is monotone in n because block_hi is nondecreasing
        lo, hi = 1, rows - start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(start, mid):
                lo = mid
            else:
                hi = mid - 1
        tiles.append((start, start + lo))
        start += lo
    return tiles


def select_topk_blocks(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """``torch.topk`` over the last dim with a fixed tie rule: higher score, then lower index.

    ``torch.topk`` leaves which of several equal scores it keeps unspecified, and the
    choice can change with the tensor shape, so a tiled scorer would not reproduce an
    untiled one. Output is ordered by score descending, then index ascending. Ties at
    ``-inf`` (invalid blocks) are left as they are; the caller maps those to ``-1``.
    """
    values, idx = torch.topk(logits, k, dim=-1)
    if k == 0:
        return values, idx
    kth = values[:, -1:]
    tied = ((logits >= kth).sum(dim=-1) > k) & torch.isfinite(kth.squeeze(-1))
    if bool(tied.any()):
        rows = tied.nonzero().squeeze(-1)
        sub = logits[rows]
        sub_kth = kth[rows]
        above = sub > sub_kth
        at = sub == sub_kth
        need = k - above.sum(dim=-1, keepdim=True)
        keep = above | (at & (at.cumsum(dim=-1, dtype=torch.int32) <= need))
        sub_idx = keep.nonzero()[:, 1].view(-1, k)
        idx[rows] = sub_idx
        values[rows] = sub.gather(-1, sub_idx)
    # order: score descending, then block index ascending
    idx, order = idx.sort(dim=-1)
    values = values.gather(-1, order)
    values, order = values.sort(dim=-1, descending=True, stable=True)
    return values, idx.gather(-1, order)


INDEXER_PARAM_GLOBS = ("index_qk_proj.*", "q_layernorm", "k_layernorm")


class Qwen38NextQSAIndexer(MegatronModule):
    """Selects the sparse-attention budget for one full-attention layer."""

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.layer_number = layer_number
        self.n_heads = config.qwen3_8_next_indexer_n_heads
        self.kv_heads = config.qwen3_8_next_indexer_kv_heads
        self.head_dim = config.qwen3_8_next_indexer_head_dim
        self.token_topk = config.qwen3_8_next_indexer_budget
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        self.norm_eps = config.layernorm_epsilon
        self.tile_bytes = getattr(
            config, "qwen3_8_next_indexer_tile_bytes", DEFAULT_TILE_BYTES
        )  # config-access-exempt: optional knob, absent from older configs

        self.index_qk_proj = TELinear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        dtype = config.params_dtype
        self.q_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.k_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        for p in (self.q_layernorm, self.k_layernorm):
            p.sequence_parallel = config.sequence_parallel
        # No training objective reaches the indexer (it runs under no_grad). Freeze it so
        # DDP gives it no main_grad and the optimizer never applies weight decay to it.
        # The helper raises when the globs match nothing; assert_qsa_indexers_frozen in the
        # provider checks that no ``.indexer.`` parameter escaped the globs.
        freeze_indexer_parameters(self, INDEXER_PARAM_GLOBS)

    def project_qk(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        layout: PackedBlockLayout | None = None,
        materialized_rope: bool = False,
    ):
        """``[T, hidden] -> (q [T, n_heads, head_dim], block_k [B, head_dim])``."""
        qk, _ = self.index_qk_proj(hidden_states)
        split = self.n_heads * self.head_dim
        q_raw, token_k = qk[..., :split], qk[..., split:]

        q = gemma_rmsnorm_last_dim(q_raw.reshape(-1, self.head_dim), self.q_layernorm, self.norm_eps).reshape(
            -1, self.n_heads, self.head_dim
        )

        token_k = token_k.reshape(-1, self.head_dim)
        if layout is None:
            block_k = compress_keys_by_mean(token_k, self.compress_ratio)
        else:
            block_k = compress_keys_by_mean_packed(token_k, layout)
        block_k = gemma_rmsnorm_last_dim(block_k, self.k_layernorm, self.norm_eps)

        if layout is None:
            block_local = torch.arange(block_k.shape[0], device=positions.device)
        else:
            block_local = layout.block_local
        # sglang rotates a compressed key at the position of its block's first token
        block_positions = block_local * self.compress_ratio
        rope_positions = positions
        if materialized_rope:
            # Image mRoPE is a per-token table, not a reusable text-position lookup.
            rope_positions = torch.arange(hidden_states.shape[0], device=positions.device)
            if layout is not None:
                block_positions = layout.block_first_token
        q = apply_indexer_rope(q, rotary_pos_emb, rope_positions)
        block_k = apply_indexer_rope(block_k.unsqueeze(1), rotary_pos_emb, block_positions).squeeze(1)
        return q, block_k

    def iter_selected_blocks(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Tensor | None = None,
        row_bytes: int = 0,
        materialized_rope: bool = False,
    ):
        """Yield ``(start, end, block_idx)`` per query tile; ``block_idx`` is ``[end - start, k]``.

        Block ids index the packed block grid (``PackedBlockLayout``), ``-1`` where unused,
        ordered by score then block id. Each tile scores only the blocks its rows can see
        and is reduced to its top-k before the next one starts, so the full ``[T, blocks]``
        score never exists: a tile holds at most ``tile_bytes``, counting ``row_bytes`` per
        row that the caller spends on each yielded tile.
        """
        layout = self.layout_for(positions, cu_seqlens)
        q, block_k = self.project_qk(
            hidden_states, positions, rotary_pos_emb, layout=layout, materialized_rope=materialized_rope
        )
        num_blocks = block_k.shape[0]
        k = min(self.block_topk, num_blocks)
        if layout is None:
            row_lo = torch.zeros_like(positions)
        else:
            row_lo = layout.token_block_start
        row_hi = row_lo + (positions + 1) // self.compress_ratio
        lo_list, hi_list = row_lo.tolist(), row_hi.tolist()
        block_k = block_k.float()
        scale = math.sqrt(self.head_dim)
        # per row: the fp32 query copy; the top-k scores and ids, their sort and mask
        # temporaries and the yielded ids
        row_bytes = row_bytes + 4 * self.n_heads * self.head_dim + 40 * k
        tiles = plan_query_tiles(lo_list, hi_list, self.tile_bytes, tile_bytes_per_cell(self.n_heads), row_bytes)
        for start, end in tiles:
            col_lo, col_hi = lo_list[start], hi_list[end - 1]
            width = col_hi - col_lo
            block_idx = torch.full((end - start, k), -1, dtype=torch.long, device=q.device)
            if width > 0 and k > 0:
                scores = torch.einsum("mhd,nd->mnh", q[start:end].float(), block_k[col_lo:col_hi])
                logits = scores.relu_().sum(dim=-1)
                del scores
                logits.div_(scale)
                cols = torch.arange(col_lo, col_hi, device=q.device).unsqueeze(0)
                valid = (cols >= row_lo[start:end].unsqueeze(1)) & (cols < row_hi[start:end].unsqueeze(1))
                logits.masked_fill_(~valid, float("-inf"))
                del valid
                kk = min(k, width)
                values, idx = select_topk_blocks(logits, kk)
                del logits
                block_idx[:, :kk] = torch.where(values == float("-inf"), -1, idx + col_lo)
            yield start, end, block_idx

    def layout_for(self, positions: Tensor, cu_seqlens: Tensor | None) -> PackedBlockLayout | None:
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            return PackedBlockLayout(cu_seqlens, positions, self.compress_ratio)
        return None

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Tensor | None = None,
        materialized_rope: bool = False,
    ) -> Tensor:
        """``[T, hidden] -> [T, token_topk]`` int32 token indices, ``-1`` where unused.

        ``positions`` restart at 0 per sequence; the returned indices are absolute in
        the packed buffer, so a packed batch must pass ``cu_seqlens``. ``rotary_pos_emb``
        is the attention's RoPE table, see ``apply_indexer_rope``.

        Blocks are ranked by score, then by lower block id among equal scores (see
        ``select_topk_blocks``), and the query rows are scored in tiles, see
        ``iter_selected_blocks``.
        """
        layout = self.layout_for(positions, cu_seqlens)
        r = self.compress_ratio
        if layout is None:
            num_blocks = -(-positions.shape[0] // r)
        else:
            num_blocks = layout.num_blocks
        k = min(self.block_topk, num_blocks)
        out = torch.full(
            (positions.shape[0], min(k * r, self.token_topk)), -1, dtype=torch.int32, device=positions.device
        )
        offsets = torch.arange(r, device=positions.device)
        # int64 token ids of a tile and two masked copies, plus two bool masks
        row_bytes = k * r * 26
        for start, end, block_idx in self.iter_selected_blocks(
            hidden_states,
            positions,
            rotary_pos_emb,
            cu_seqlens,
            row_bytes=row_bytes,
            materialized_rope=materialized_rope,
        ):
            if layout is None:
                tokens = block_idx.unsqueeze(-1) * r + offsets
                limit = positions[start:end]
            else:
                block_local = block_idx - layout.token_block_start[start:end].unsqueeze(-1)
                token_start = layout.token_start[start:end]
                tokens = token_start.unsqueeze(-1).unsqueeze(-1) + (block_local.unsqueeze(-1) * r + offsets)
                limit = token_start + positions[start:end]
            tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
            tokens = tokens.masked_fill(tokens > limit.unsqueeze(-1), -1)
            if layout is not None:
                tokens = tokens.masked_fill(tokens < token_start.unsqueeze(-1), -1)
            out[start:end] = tokens[..., : self.token_topk]
        return out
