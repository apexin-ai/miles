"""Qwen3.8 MTP inputs shared between vision SFT and the native draft layer."""

from dataclasses import dataclass
from typing import Callable

import torch

from miles_plugins.models.qwen3_8_next.mtp_config import is_qwen38_spec
from miles_plugins.models.qwen3_8_next.vision_inputs import scatter_image_embeddings


def validate_vision_mtp(args):
    if not is_qwen38_spec(args.backend.spec):
        raise ValueError("Qwen3.8 vision MTP requires the get_qwen3_8_next_spec layer spec")
    if args.pipeline_model_parallel_size > 1 and args.qwen38_vision_data != "text":
        raise ValueError("Qwen3.8 image MTP needs pipeline size 1 to share ViT embeddings with the draft stage")


def mask_image_targets(loss_mask, target_ids, *, image_token_id):
    """Mask placeholder targets before Megatron rolls both targets and masks within each packed row."""
    valid = target_ids.ne(image_token_id)
    return valid if loss_mask is None else loss_mask.masked_fill(~valid, 0)


@dataclass
class MTPImageEmbedding:
    """The shared word embedding with this pack's ViT outputs at the shifted token positions.

    Every native image span follows a vision-start token inside its own validated sample. Moving the image
    positions left by one therefore never crosses a sample boundary. CP and sequence-parallel ownership
    must be evaluated *after* moving them, including when a visual token moves to a different rank.
    """

    embedding: Callable
    image_positions: torch.Tensor
    image_embeddings: torch.Tensor
    tp_rank: int
    sequence_parallel: bool
    window_start: int | None = None

    # With mRoPE, position IDs belong to the target hidden states. Megatron only rolls the token IDs.
    add_position_embedding = False

    def __call__(self, input_ids, position_ids):
        return scatter_image_embeddings(
            self.embedding(input_ids=input_ids, position_ids=position_ids),
            self.image_positions - 1,
            self.image_embeddings,
            tp_rank=self.tp_rank,
            sequence_parallel=self.sequence_parallel,
            window_start=self.window_start,
        )
