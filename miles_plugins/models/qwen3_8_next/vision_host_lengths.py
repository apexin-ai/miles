"""Give the HF Qwen3-VL vision tower a host copy of its attention cu_seqlens: one device sync per tower forward.

On a non-flash attention implementation (miles runs the tower with sdpa) every ``Qwen3VLVisionAttention`` splits the
packed q / k / v per image with ``torch.split(t, (cu_seqlens[1:] - cu_seqlens[:-1]).tolist())``, three times per
block. With ``cu_seqlens`` on the GPU each ``.tolist()`` is a device-to-host sync: 3 x depth per pass, and the same
again when gradient checkpointing recomputes the blocks.

``Qwen3VLVisionModel.forward`` takes a precomputed ``cu_seqlens`` kwarg (``get_vision_cu_seqlens`` pops it) and uses
it only to pass to the blocks; ``get_max_seqlen`` does not read it off the flash path. A block passes it only to its
attention, which on the non-flash path uses it only for the split lengths (rotary embeddings come from
``position_ids``). A forward pre-hook on the tower therefore computes ``cu_seqlens`` once from a host copy of
``grid_thw``, with the same transformers function and the same integer values. The checkpointed blocks hold that
host tensor in their saved kwargs, so the recompute needs no sync at all.

The flash path needs ``cu_seqlens`` on the device for the varlen kernel: there the hook is not installed, and an
installed hook steps aside if the implementation is switched to flash later.
"""

from transformers.utils.generic import is_flash_attention_requested
from transformers.vision_utils import get_vision_cu_seqlens


def host_cu_seqlens(grid_thw):
    """The tower's packed cu_seqlens computed on the host (one device-to-host copy of the small grid tensor)."""
    return get_vision_cu_seqlens(grid_thw.cpu())


def _pre_hook(module, args, kwargs):
    if "cu_seqlens" in kwargs or is_flash_attention_requested(module.config):
        return None
    grid_thw = kwargs["grid_thw"] if "grid_thw" in kwargs else args[1]  # forward(hidden_states, grid_thw, **kwargs)
    kwargs["cu_seqlens"] = host_cu_seqlens(grid_thw)
    return args, kwargs


def install_host_cu_seqlens(visual):
    """Hook ``visual`` (a Qwen3VLVisionModel) to run its blocks with host cu_seqlens; returns the hook handle.

    Returns None (no hook) when the tower runs a flash attention implementation.
    """
    if is_flash_attention_requested(visual.config):
        return None
    return visual.register_forward_pre_hook(_pre_hook, with_kwargs=True)
