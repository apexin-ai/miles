"""Qwen3.8-Next hyper-connections for Megatron's HC ModuleSpec slots.

Differs from Megatron's mHC (DeepSeek-V4): per-stream per-feature low-rank read
gate with a MEAN over streams, and identity residual mixing (h_res is None).
"""

import torch
import torch.nn.functional as F
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles.kernels.hyper_connection.hc import check_matmul_precision, hc_combine, hc_mix_inject
from miles_plugins.models.qwen3_8_next.ops.cp import cp_enabled, cp_rank_of, cp_world_size
from miles_plugins.models.qwen3_8_next.ops.ple import Qwen38NextPLE, current_ple_batch, window_cu_seqlens


def _hc_matmul_precision(config) -> str:
    """--qwen38-hc-matmul-precision (fp32 / tf32 / bf16) of the gate chain's matmuls; see hc.py."""
    return check_matmul_precision(
        getattr(config, "qwen3_8_next_hc_matmul_precision", "fp32")
    )  # config-access-exempt: optional knob, absent from older configs


class Qwen38NextHyperConnection(MegatronModule):
    """Per-layer HC filling the attention/MLP spec slots. Params replicated across TP
    (sequence_parallel attr set); kept bf16 to match sglang."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hc_count: int | None = None,
        use_combine: bool = True,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.use_combine = use_combine
        self.matmul_precision = _hc_matmul_precision(config)

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))
        params = [self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up]

        if use_combine:
            self.block_inject_weight = torch.nn.Parameter(torch.empty(self.n, wide, dtype=dtype))
            params.append(self.block_inject_weight)
        else:
            self.block_inject_weight = None

        for p in params:
            p.sequence_parallel = config.sequence_parallel

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)
            if use_combine:
                torch.nn.init.xavier_uniform_(self.block_inject_weight)

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        """Returns (aggregated, h_res=None, h_post, residual)."""
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError(
                "Qwen38NextHyperConnection does not support the mHC recompute arena yet; "
                "run without 'mhc' in --recompute-modules."
            )
        assert self.use_combine, "per-layer HC needs the inject weight; use the Mixer for read-only"
        aggregated, h_post = hc_mix_inject(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            self.block_inject_weight,
            self.n,
            self.norm_eps,
            matmul_precision=self.matmul_precision,
        )
        return aggregated, None, h_post, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor | None,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias,
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        """``X'_c = X_c + a_c * (y + bias)``, dropout applied to the injection."""
        assert h_res is None, "Qwen3.8-Next hyper-connection has no residual mixing matrix"
        if manager is not None:
            raise NotImplementedError("mHC recompute arena not supported yet")

        if isinstance(layer_output_with_bias, tuple):
            x, bias = layer_output_with_bias
        else:
            x, bias = layer_output_with_bias, None

        if bias is not None:
            x = x + bias.view(*([1] * (x.dim() - 1)), -1)
        if dropout_prob > 0.0 and training:
            x = F.dropout(x, p=dropout_prob)
        return hc_combine(original_residual, x, h_post, self.n)


class Qwen38NextHCHeadContraction(MegatronModule):
    """Model-level output contraction [s,b,n*C]->[s,b,C]: same gated mean as the
    per-layer HC (per-stream RMS) -- NOT interchangeable with DSv4's built-in."""

    def __init__(self, config: TransformerConfig, hc_count: int | None = None):
        super().__init__(config)
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.matmul_precision = _hc_matmul_precision(config)

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))

        for p in (self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up):
            p.sequence_parallel = config.sequence_parallel
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)

    def forward(self, hidden_states: Tensor) -> Tensor:
        mixed, _ = hc_mix_inject(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            None,
            self.n,
            self.norm_eps,
            matmul_precision=self.matmul_precision,
        )
        return mixed


class Qwen38NextPLEHyperConnection(Qwen38NextHyperConnection):
    """Attention-site HC for the PLE layer: applies the PLE increment before the read.

    Under SP each rank computes the increment for its own slice only, on a window that adds the
    conv's look-back halo (the previous rank's last ``(kernel - 1) * dilation`` tokens); see
    ``_apply_ple_local_window``. ``ple_sp_local_slice = False`` selects the older full-sequence
    path (all-gather, PLE on all T tokens, keep the own slice), which is also the fallback when
    the window does not apply.
    """

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        super().__init__(config, layer_number, **kwargs)

        tp_group = None
        try:
            tp_group = get_tensor_model_parallel_group()
        except AssertionError:
            pass  # not initialised (shape audits); falls back to unsharded
        self.ple = Qwen38NextPLE(config, layer_number=layer_number, tp_group=tp_group)
        self.ple_sp_local_slice = True

    def _resolve_ple_batch(self):
        """The published batch, made safe under activation recompute.

        The recompute pass runs in backward, after the side channel was cleared or
        overwritten by a later microbatch, so the checkpointed pass enqueues its batch
        and the recompute pass pops it. FIFO matches non-interleaved 1F1B.
        """
        if not hasattr(self, "_ple_recompute_fifo"):  # config-access-exempt: the recompute FIFO is created lazily
            self._ple_recompute_fifo = []

        if is_checkpointing() and torch.is_grad_enabled():
            if not self._ple_recompute_fifo:
                raise RuntimeError(
                    "PLE recompute ran with no queued n-gram batch; the checkpointed "
                    "original pass did not enqueue (or the recompute order diverged "
                    "from FIFO, e.g. interleaved VPP)."
                )
            return self._ple_recompute_fifo.pop(0)

        batch = current_ple_batch()
        if is_checkpointing():
            self._ple_recompute_fifo.append(batch)
        return batch

    @staticmethod
    def _flat_ple_ids(ngram_ids, seq, batch):
        if ngram_ids.dim() == 3:
            if ngram_ids.shape[:2] != (batch, seq):
                raise RuntimeError(
                    f"PLE ngram ids are [B, S, heads]={tuple(ngram_ids.shape)}, which "
                    f"does not match the hidden state's B={batch}, S={seq}"
                )
            ngram_ids = ngram_ids.transpose(0, 1)
        return ngram_ids.reshape(seq * batch, ngram_ids.shape[-1])

    def _apply_ple(self, hidden_states, ngram_ids, cu_seqlens):
        seq, batch = hidden_states.shape[0], hidden_states.shape[1]
        flat_ids = self._flat_ple_ids(ngram_ids, seq, batch)
        flat_state = hidden_states.reshape(seq * batch, hidden_states.shape[-1])

        increment = self.ple(flat_state, flat_ids, cu_seqlens)

        assert (
            increment.shape == flat_state.shape
        ), f"PLE increment {tuple(increment.shape)} != state {tuple(flat_state.shape)}"
        hidden_states = hidden_states + increment.view_as(hidden_states)
        return hidden_states

    def _ple_shard(self, hidden_states):
        """``(rank, size)`` of the sequence shard this rank holds, over SP and CP together.

        The two shardings nest: CP slices the global packed stream into contiguous pieces, then SP
        slices the piece a CP rank holds. A rank's position in the global sequence is therefore
        ``cp_rank * sp_size + sp_rank`` out of ``cp_size * sp_size`` equal shards, which is exactly
        what the local-window path needs -- it only ever asks "which shard am I of how many" to find
        its halo. Returning the composed pair lets one code path serve SP, CP, and both.

        Returns ``(0, 1)`` when neither is on, i.e. this rank holds the whole sequence.
        """
        sp_size = 1
        sp_rank = 0
        if getattr(
            self.config, "sequence_parallel", False
        ):  # config-access-exempt: model configs may omit sequence_parallel
            sp_size = get_tensor_model_parallel_world_size()
            if sp_size > 1:
                sp_rank = get_tensor_model_parallel_rank()
        if not cp_enabled(self.config):
            return sp_rank, sp_size
        return cp_rank_of() * sp_size + sp_rank, cp_world_size() * sp_size

    def _apply_ple_local_window(self, hidden_states, ngram_ids, cu_seqlens, sp_rank, sp_size, groups=None):
        """The PLE update of this rank's SP slice ``[S/sp, 1, n*C]``, computed on its slice only.

        The window is ``[start, start + local + lookback)`` with ``start = rank * local - lookback``:
        the previous rank's last ``lookback`` tokens (the halo) and this rank's slice. The halo
        state comes from an all-gather of every rank's tail (``sp * lookback`` rows); its backward
        is the reduce-scatter (SUM) that returns the halo's gradient to the previous rank, as the
        full-sequence path's reduce-scatter does. Rank 0 has no previous tokens: it takes the last
        rank's tail as filler rows in a separate segment, which its tokens never read, so every
        rank runs the same shapes and collectives. The n-gram embedding lookup and its TP
        all-reduce stay on the full sequence; only the window's rows go on.

        Gradient order matches the full-sequence path bitwise: the residual is added on the window,
        so this rank's own (residual + PLE) gradient is summed first and the next rank's halo
        gradient is added to it afterwards, as the reduce-scatter there adds it.

        ``sp_rank`` / ``sp_size`` are the shard's index and count over whatever is sharding the
        sequence: SP alone, or SP and CP composed (``_ple_shard``). ``groups`` is the process group
        the tails are exchanged over, one entry per sharding, outermost first; it defaults to the TP
        group. With CP the halo may come from a rank of the CP group rather than the TP group, so
        the exchange runs over both and the tails are indexed in the composed order.
        """
        local, batch, width = hidden_states.shape
        lookback = self.ple.conv_lookback
        total = local * sp_size
        start = sp_rank * local - lookback
        length = local + lookback
        if groups is None:
            groups = (get_tensor_model_parallel_group(),)

        flat_ids = self._flat_ple_ids(ngram_ids, total, batch)
        if flat_ids.shape[0] != total:
            raise RuntimeError(
                f"PLE ngram ids cover {flat_ids.shape[0]} tokens; the SP slices cover {sp_size} x {local} = {total}"
            )
        flat_state = hidden_states.reshape(local, width)

        tails = self._gather_tails(flat_state[local - lookback :], groups)
        prev = (sp_rank - 1) % sp_size
        halo = tails[prev * lookback : (prev + 1) * lookback]
        window_state = torch.cat([halo, flat_state])

        rows = torch.arange(start, start + length, device=flat_ids.device) % total
        embeddings = self.ple.embed(flat_ids).index_select(0, rows)
        window_cu = window_cu_seqlens(cu_seqlens, total, start, length, device=flat_ids.device)

        increment = self.ple.increment(window_state, embeddings, window_cu)
        assert (
            increment.shape == window_state.shape
        ), f"PLE increment {tuple(increment.shape)} != window state {tuple(window_state.shape)}"
        updated = window_state + increment
        return updated[lookback:].view_as(hidden_states)

    @staticmethod
    def _gather_tails(tail, groups):
        """Every shard's tail rows, ordered by composed shard index (outermost group first).

        One gather per sharding, innermost first: gathering over TP gives this CP rank's
        ``sp_size`` tails, and gathering that over CP gives all ``cp_size * sp_size`` of them in
        ``cp_rank * sp_size + sp_rank`` order -- the order ``_ple_shard`` numbers shards in. Every
        gather takes ``tensor_parallel_output_grad=True``, so each backward reduce-scatters (SUM)
        and a halo's gradient reaches the rank that owns those tokens.
        """
        for group in reversed(groups):
            tail = gather_from_sequence_parallel_region(tail, tensor_parallel_output_grad=True, group=group)
        return tail

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        ngram_ids, cu_seqlens = self._resolve_ple_batch()

        # shard index / count over SP and CP composed; (0, 1) when this rank holds everything
        shard_rank, shard_size = self._ple_shard(hidden_states)
        cp = cp_enabled(self.config)
        groups = (
            (get_context_parallel_group(), get_tensor_model_parallel_group())
            if cp
            else (get_tensor_model_parallel_group(),)
        )
        if cp and shard_size == cp_world_size():
            # Sequence parallelism off (or TP1): the TP ranks of a CP rank are replicas of one CP shard, not
            # shards of it, so the tails are exchanged over the CP group only. Over TP too, the gathered tails
            # would hold tp copies of each CP shard's tail and a CP rank >= 2 would read an earlier shard's.
            groups = (get_context_parallel_group(),)
        if (
            shard_size > 1
            and self.ple_sp_local_slice
            and hidden_states.shape[1] == 1
            and hidden_states.shape[0] >= self.ple.conv_lookback
        ):
            hidden_states = self._apply_ple_local_window(
                hidden_states, ngram_ids, cu_seqlens, shard_rank, shard_size, groups=groups
            )
        elif shard_size > 1:
            if cp:
                raise NotImplementedError(
                    "Qwen3.8-Next PLE under context parallelism needs the local-window path "
                    "(--qwen38-ple-sp-local-slice): the full-sequence fallback gathers over the TP "
                    "group only, which under CP is a fraction of the sequence the n-gram conv reads. "
                    f"ple_sp_local_slice={self.ple_sp_local_slice}, rows={hidden_states.shape[0]}, "
                    f"lookback={self.ple.conv_lookback}, batch={hidden_states.shape[1]}"
                )
            sp_rank = get_tensor_model_parallel_rank()
            local_seq = hidden_states.shape[0]
            # The causal conv reads (kernel - 1) * dilation tokens back, so this rank's loss also
            # reaches the tail of the previous rank's slice: the backward must reduce-scatter
            # (SUM) the full-sequence grad, not just keep this rank's slice of it.
            full = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=True,
                group=get_tensor_model_parallel_group(),
            )
            updated = self._apply_ple(full, ngram_ids, cu_seqlens)
            hidden_states = updated[sp_rank * local_seq : (sp_rank + 1) * local_seq]
        else:
            hidden_states = self._apply_ple(hidden_states, ngram_ids, cu_seqlens)
        return super().forward(
            hidden_states,
            mhc_recompute_manager=mhc_recompute_manager,
            output_slot=output_slot,
        )
