"""Backend selection and checkpoint writing for Megatron HF exports."""

import json
import logging
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.utils import is_lora_model
from miles.backends.megatron_utils import ple_provenance
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.megatron_bridge_utils import load_auto_bridge, patch_megatron_model

logger = logging.getLogger(__name__)


_SAFETENSORS_INDEX = "model.safetensors.index.json"


def missing_hf_weights(source_dir: str | Path, export_dir: str | Path) -> list[str]:
    """Tensors in the source checkpoint's safetensors index that the export does not deliver.

    A tensor is delivered when the export's index names it and the shard file it names
    exists (the index is planned up front; a shard is written only once all its tensors
    arrive). Empty when the source has no local index to compare against.
    """
    source_index = Path(source_dir) / _SAFETENSORS_INDEX
    if not source_index.is_file():
        return []
    expected = json.loads(source_index.read_text())["weight_map"]
    export_index = Path(export_dir) / _SAFETENSORS_INDEX
    if not export_index.is_file():
        return sorted(expected)
    written = json.loads(export_index.read_text())["weight_map"]
    shards = {f for f in set(written.values()) if (Path(export_dir) / f).is_file()}
    return sorted(k for k in expected if written.get(k) not in shards)


def _bridge_export_gap(hf_checkpoint: str, checkpoint_dir: Path) -> str | None:
    """Why a bridge export in ``checkpoint_dir`` is incomplete, or None. Runs on rank 0 only."""
    if not any(checkpoint_dir.glob("*.safetensors")) and not any(checkpoint_dir.glob("*.bin")):
        return "no weight files"
    if missing := missing_hf_weights(hf_checkpoint, checkpoint_dir):
        # A partly mapped model still writes files, but drops every shard
        # holding an unmapped tensor -- and whatever else shares it.
        return f"{len(missing)} tensor(s) of {hf_checkpoint} missing, e.g. {missing[:8]}"
    return None


@cache
def _get_hf_bridge(hf_checkpoint: str):
    return load_auto_bridge(hf_checkpoint)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    publisher: SnapshotPublisher,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Collectively write an HF model, with an additional HF adapter for LoRA.

    Writes a ``.complete`` marker after all ranks finish. Export errors are logged
    unless ``raise_on_error`` is set.
    """
    should_log = get_parallel_state().effective_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))

    def write_weights(checkpoint_dir: Path):
        if args.megatron_to_hf_mode == "raw":
            weights = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
            publisher.write_model(
                checkpoint_dir,
                weights=merge_lora_into_weights(model, weights) if is_lora_model(model) else weights,
                hf_checkpoint=args.hf_checkpoint,
            )
            is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            # a run bound to a PLE table at load must complete its export, whatever the base's config says now
            required = ple_provenance.run_manifest() is not None
            if is_writer and (required or hf_passthrough.applies(args.hf_checkpoint)):
                # base tensors the model does not hold (Qwen3.8: PLE table, untrained MTP, unbuilt vision tower)
                receipt = hf_passthrough.complete_hf_export(
                    checkpoint_dir, args.hf_checkpoint, mtp_trained=args.enable_mtp_training, required=required
                )
                if receipt is not None:
                    ple_provenance.check_export(receipt["passthrough_sha256"])
                    logger.info(
                        f"HF export {checkpoint_dir}: {receipt['exported_keys']} exported + "
                        f"{receipt['passthrough_keys']} passed through from {receipt['base']} {receipt['passthrough_groups']}"
                    )
        else:
            bridge = _get_hf_bridge(args.hf_checkpoint)
            with patch_megatron_model(model):
                bridge.save_hf_pretrained(model, path=checkpoint_dir)
            torch.distributed.barrier(group=get_gloo_group())
            # Every rank waits in the broadcast below, so rank 0 must reach it whatever its local check does:
            # a local error (unreadable or malformed index) travels as the payload, then all ranks raise.
            incomplete = [None]
            local_error = None
            if torch.distributed.get_rank() == 0:
                try:
                    incomplete[0] = _bridge_export_gap(args.hf_checkpoint, checkpoint_dir)
                except Exception as exc:
                    local_error = exc
                    incomplete[0] = f"coverage check failed: {type(exc).__name__}: {exc}"
            torch.distributed.broadcast_object_list(incomplete, src=0, group=get_gloo_group())
            if incomplete[0] is not None:
                raise RuntimeError(
                    f"HF export to {path} is incomplete ({incomplete[0]}) — the megatron bridge likely has "
                    "no mapping for part of this model architecture."
                ) from local_error
        if is_lora_model(model):
            publisher.write_adapter(None, checkpoint_dir / "adapter")

    if should_log:
        logger.info(f"Saving model in HuggingFace format to {path}")
    try:
        write_checkpoint_dir(path, write_weights, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    except Exception as e:
        if raise_on_error:
            raise
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
    else:
        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
