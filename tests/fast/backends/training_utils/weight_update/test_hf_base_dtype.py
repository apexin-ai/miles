import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update.hf_base_dtype import BaseDtypeCaster, read_safetensors_dtypes

A_LOG = "model.layers.0.linear_attn.A_log"
EMBED = "model.embed_tokens.weight"


def _write_sharded_base(directory: Path, shards: dict[str, dict[str, torch.Tensor]]) -> Path:
    directory.mkdir()
    weight_map = {}
    for file, tensors in shards.items():
        save_file(tensors, str(directory / file))
        weight_map |= {name: file for name in tensors}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return directory


def _trained_a_log() -> torch.Tensor:
    # values bf16 cannot hold exactly, so a missing cast or a different rounding shows
    return torch.linspace(-3.0, 1.0, 8, dtype=torch.float32) + 1e-4


class TestReadSafetensorsDtypes:
    def test_reads_the_header_dtypes_and_skips_the_metadata(self, tmp_path) -> None:
        file = tmp_path / "model.safetensors"
        save_file(
            {A_LOG: torch.zeros(8, dtype=torch.bfloat16), "ids": torch.zeros(2, dtype=torch.int64)},
            str(file),
            metadata={"format": "pt"},
        )
        assert read_safetensors_dtypes(file) == {A_LOG: "BF16", "ids": "I64"}


class TestBaseDtypeCaster:
    def test_casts_only_the_allow_listed_tensor_to_the_base_dtype(self, tmp_path) -> None:
        base = _write_sharded_base(
            tmp_path / "base",
            {
                "model-00001.safetensors": {EMBED: torch.zeros(2, dtype=torch.bfloat16)},
                "model-00002.safetensors": {A_LOG: torch.zeros(8, dtype=torch.bfloat16)},
            },
        )
        caster = BaseDtypeCaster.from_checkpoint(base)
        a_log = _trained_a_log()

        cast = caster.cast(A_LOG, a_log)
        assert cast.dtype == torch.bfloat16
        assert torch.equal(cast, a_log.to(torch.bfloat16))
        embed = torch.ones(2, dtype=torch.float32)
        assert caster.cast(EMBED, embed) is embed

    def test_reads_a_single_file_base_without_an_index(self, tmp_path) -> None:
        (tmp_path / "base").mkdir()
        save_file({A_LOG: torch.zeros(8, dtype=torch.float16)}, str(tmp_path / "base" / "model.safetensors"))
        assert BaseDtypeCaster.from_checkpoint(tmp_path / "base").cast(A_LOG, _trained_a_log()).dtype == torch.float16

    def test_is_a_no_op_when_the_base_has_no_copy_or_the_same_dtype(self, tmp_path) -> None:
        a_log = _trained_a_log()
        same = _write_sharded_base(tmp_path / "same", {"model.safetensors": {A_LOG: torch.zeros(8)}})
        assert BaseDtypeCaster.from_checkpoint(same).cast(A_LOG, a_log) is a_log
        other = _write_sharded_base(tmp_path / "other", {"model.safetensors": {EMBED: torch.zeros(2)}})
        assert BaseDtypeCaster.from_checkpoint(other).cast(A_LOG, a_log) is a_log
        (tmp_path / "empty").mkdir()
        assert BaseDtypeCaster.from_checkpoint(tmp_path / "empty").cast(A_LOG, a_log) is a_log

    def test_never_casts_integer_tensors(self, tmp_path) -> None:
        base = _write_sharded_base(
            tmp_path / "base", {"model.safetensors": {A_LOG: torch.zeros(8, dtype=torch.bfloat16)}}
        )
        ids = torch.arange(8)
        assert BaseDtypeCaster.from_checkpoint(base).cast(A_LOG, ids) is ids

        int_base = _write_sharded_base(
            tmp_path / "int_base", {"model.safetensors": {A_LOG: torch.zeros(8, dtype=torch.int32)}}
        )
        a_log = _trained_a_log()
        assert BaseDtypeCaster.from_checkpoint(int_base).cast(A_LOG, a_log) is a_log

    def test_custom_allow_list(self, tmp_path) -> None:
        base = _write_sharded_base(
            tmp_path / "base",
            {
                "model.safetensors": {
                    A_LOG: torch.zeros(8, dtype=torch.bfloat16),
                    EMBED: torch.zeros(2, dtype=torch.bfloat16),
                }
            },
        )
        caster = BaseDtypeCaster.from_checkpoint(base, patterns=[re.compile(re.escape(EMBED))])
        a_log = _trained_a_log()
        assert caster.cast(A_LOG, a_log) is a_log
        assert caster.cast(EMBED, torch.ones(2)).dtype == torch.bfloat16
