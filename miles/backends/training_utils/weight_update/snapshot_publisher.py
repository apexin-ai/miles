import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_base_dtype import BaseDtypeCaster
from miles.backends.training_utils.weight_update.hf_expert_packing import ExpertPacker
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.lora.utils import AdapterSpec, get_adapter_target_modules

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    # The base checkpoint's weight index must not overwrite the exported shard mapping.
    return path.is_file() and path.suffix not in HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json")


class SnapshotPublisher:
    def __init__(self, iterator: HfWeightIteratorBase, adapter_config: dict | None = None) -> None:
        assert iterator.placement.is_full_gather, "publishing requires full weights on rank 0"
        self._iterator = iterator
        self._adapter_config = adapter_config

    def publish_adapter(self, adapter: AdapterSpec | None, path: str, metadata: dict | None = None) -> None:
        write_checkpoint_dir(
            path,
            lambda checkpoint_dir: self.write_adapter(adapter, checkpoint_dir),
            metadata=metadata,
            overwrite=False,
        )

    @torch.no_grad()
    def write_adapter(self, adapter: AdapterSpec | None, path: str | Path) -> None:
        """Collectively write HF adapter files into the caller's checkpoint directory."""
        assert self._adapter_config is not None, "adapter export requires adapter_config"
        path = Path(path)
        is_writer = dist.get_rank() == 0
        adapter_tensors = {
            name: tensor.detach().contiguous().cpu()
            for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
        }
        adapter_bytes = safetensors.torch.save(adapter_tensors) if is_writer else None
        adapter_config = self._adapter_config
        if adapter is not None:
            adapter_config = adapter_config | {"r": adapter.rank, "lora_alpha": adapter.alpha}

        if is_writer:
            adapter_config = adapter_config | {"target_modules": get_adapter_target_modules(adapter_tensors)}
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps(adapter_config))
            (path / "adapter_model.safetensors").write_bytes(adapter_bytes)

    def write_model(self, path: str | Path, *, weights: Mapping[str, torch.Tensor], hf_checkpoint: str) -> None:
        """Collectively write HF model shards into the caller's checkpoint directory."""
        path = Path(path)
        is_writer = dist.get_rank() == 0

        weight_map: dict[str, str] = {}
        total_size = 0
        shard_index = 0
        write_error = None
        # base checkpoints that store MoE experts fused get the same fused keys back; tensors
        # training holds wider than the base (GDN A_log) go back to the base dtype
        packer = caster = None
        if is_writer and Path(hf_checkpoint).is_dir():
            try:
                packer = ExpertPacker.for_base(hf_checkpoint)
                caster = BaseDtypeCaster.from_checkpoint(hf_checkpoint)
            except Exception as exc:
                # Raised only after every rank drained the iterator: its weight gathers are collective.
                write_error = exc
        for hf_named_tensors in self._iterator.iter_hf_weights(weights):
            if not is_writer or write_error is not None:
                continue
            shard_tensors = {}
            try:
                for name, tensor in hf_named_tensors:
                    for out_name, out_tensor in packer.feed(name, tensor) if packer else [(name, tensor)]:
                        if caster is not None:
                            out_tensor = caster.cast(out_name, out_tensor)
                        shard_tensors[out_name] = out_tensor.detach().to("cpu").contiguous()
                if not shard_tensors:
                    continue  # every tensor of this bucket is buffered for an expert layer
                shard_index += 1
                shard_name = f"model-{shard_index:05d}.safetensors"
                for name, tensor in shard_tensors.items():
                    weight_map[name] = shard_name
                    total_size += tensor.numel() * tensor.element_size()
                safetensors.torch.save_file(shard_tensors, path / shard_name)
            except Exception as exc:
                # Peers must finish the remaining weight gathers before reporting a write failure.
                write_error = exc
            del shard_tensors

        if write_error is None and packer is not None:
            try:
                packer.finish()
            except Exception as exc:
                write_error = exc
        if write_error is not None:
            raise write_error
        if is_writer:
            assert weight_map, f"HF export to {path} produced no weights"
            base_checkpoint = Path(hf_checkpoint)
            if base_checkpoint.is_dir():
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                logger.warning(f"hf_checkpoint {hf_checkpoint} is not a local dir; metadata not copied to {path}")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
