"""Native image/text routing-bias selection for image SFT.

Biases only select a discrete top-k; they are checkpointed buffers, not Adam
parameters. Gate weights remain differentiable and are not frozen here.
"""

import torch
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router import TopKRouter


class DeepSeekV41VisionRouter(TopKRouter):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if not self.enable_expert_bias:
            raise ValueError("V4.1 image routing requires native expert biases")
        if config.moe_router_bias_update_rate != 0 or self.is_aux_loss_enabled():
            raise ValueError("V4.1 image SFT currently requires fixed selection biases and zero auxiliary loss")
        self.register_buffer("expert_bias_vl", torch.zeros_like(self.expert_bias, dtype=torch.float32))

    def routing(self, logits, padding_mask=None, input_ids=None, packed_seq_params=None):
        ids = self.config.v41_runtime.input_ids
        if ids is None or ids.shape[0] != logits.shape[1]:
            raise ValueError("V4.1 image router did not receive the matching token IDs")
        length = logits.shape[0]
        mask = (ids == self.config.v41_image_token_id).transpose(0, 1)
        if self.config.sequence_parallel and mask.shape[0] != length:
            rank = self.tp_group.rank()
            mask = mask[rank * length : (rank + 1) * length]
        if tuple(mask.shape) != tuple(logits.shape[:2]):
            raise ValueError("V4.1 image routing mask is not aligned with its sequence shard")
        scores = self.apply_z_loss(
            logits.reshape(-1, self.config.num_moe_experts),
            padding_mask=None if padding_mask is None else padding_mask.reshape(-1),
        )
        bias = torch.where(mask.reshape(-1, 1), self.expert_bias_vl.float(), self.expert_bias.float())
        return topk_routing_with_score_function(
            scores,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=bias,
            fused=False,
            router_replay=self.router_replay,
            is_mtp=self.is_mtp_layer,
        )
