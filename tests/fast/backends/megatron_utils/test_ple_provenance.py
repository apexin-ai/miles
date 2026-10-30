"""A resumed Qwen3.8 run must read the same PLE table bytes it started with (RV P1-1)."""

import json
import logging
import shutil
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.weight_update.test_hf_passthrough import BASE_LAYOUT, PLE, _tensor, _write
from tests.fast.dist_utils import init_gloo, run_multiprocess
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import model as model_module
from miles.backends.megatron_utils import ple_provenance
from miles.backends.megatron_utils.ple_provenance import PleProvenanceError, check_export, compute_manifest, on_load, on_save

PLE_SHARD_FILE = "model-00002-of-00004.safetensors"


def _make_base(directory: Path, *, overrides: dict | None = None) -> Path:
    tensors = {k: _tensor(k, 0) for keys in BASE_LAYOUT.values() for k in keys} | (overrides or {})
    _write(directory, BASE_LAYOUT, tensors)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 4}}))
    download = directory / ".cache" / "huggingface" / "download"
    download.mkdir(parents=True)
    for name in ("config.json", PLE_SHARD_FILE):
        (download / f"{name}.metadata").write_text("de4b8e4d43b917e7706784d8bb445c9af86a3540\nsha\n0\n")
    return directory


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch):
    """Each test starts like a new trainer process: no run manifest in memory."""
    monkeypatch.setattr(ple_provenance, "_run_manifest", None)
    monkeypatch.delenv("MILES_PLE_MANIFEST", raising=False)
    monkeypatch.delenv("MILES_PLE_PROVENANCE_ADOPT", raising=False)


def _train_and_save(base: Path, ckpt: Path, iteration: int = 4) -> dict:
    """First process: fresh start from the base, then a save."""
    manifest = on_load(base, load_dir=None, iteration=0, resumed=False)
    on_save(ckpt, iteration)
    ple_provenance._run_manifest = None  # the process ends
    return manifest


class TestManifest:
    def test_the_manifest_covers_every_ple_key_with_its_bytes_and_the_base_identity(self, tmp_path) -> None:
        """Keys, dtype, shape and raw-byte sha256 of the PLE table, plus revision and config / index digests."""
        base = _make_base(tmp_path / "base")
        manifest = compute_manifest(base)
        assert sorted(manifest["ple"]) == sorted(k for keys in BASE_LAYOUT.values() for k in keys if ".ple_embedding." in k)
        assert manifest["ple"][f"{PLE}.layer_multipliers"]["dtype"] == "I64"
        assert manifest["ple"][f"{PLE}.ngram_embedding.shard_0.weight"]["shape"] == [4, 6]
        assert manifest["revision"] == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        assert len(manifest["config_sha256"]) == len(manifest["index_sha256"]) == 64

    def test_non_qwen38_checkpoints_and_unset_paths_are_ignored(self, tmp_path) -> None:
        """Other models keep their behaviour; stubbed tests pass hf_checkpoint=None."""
        assert on_load(None, load_dir=None, iteration=0, resumed=False) is None
        other = tmp_path / "other"
        other.mkdir()
        (other / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
        assert on_load(other, load_dir=None, iteration=0, resumed=False) is None


class TestResume:
    def test_a_resume_against_the_same_bytes_passes_across_processes(self, tmp_path) -> None:
        """The manifest travels with the checkpoint; the new process re-hashes the base against it."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        first = _train_and_save(base, ckpt)
        assert (ckpt / "iter_0000004" / "ple_provenance.json").is_file()
        assert on_load(base, load_dir=ckpt, iteration=4, resumed=True)["ple"] == first["ple"]

    @pytest.mark.parametrize(
        "key",
        [
            f"{PLE}.ngram_embedding.shard_1.weight",
            f"{PLE}.layer_multipliers",
            f"{PLE}.ngram_heads_offsets",
            f"{PLE}.ngram_heads_vocab_sizes",
        ],
    )
    def test_the_same_path_with_a_replaced_ple_tensor_is_refused(self, tmp_path, key) -> None:
        """Any of the 128 shards or the three hash parameters changing under the same path fails the resume."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        _train_and_save(base, ckpt)
        shutil.rmtree(base)
        _make_base(base, overrides={key: _tensor(key, 9)})
        with pytest.raises(PleProvenanceError, match="1 PLE tensors differ"):
            on_load(base, load_dir=ckpt, iteration=4, resumed=True)

    def test_another_path_with_the_same_bytes_passes_and_is_logged(self, tmp_path, caplog) -> None:
        """Provenance is the content; a moved or re-downloaded base with identical bytes is fine."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        _train_and_save(base, ckpt)
        moved = tmp_path / "moved"
        shutil.copytree(base, moved)
        with caplog.at_level(logging.WARNING):
            on_load(moved, load_dir=ckpt, iteration=4, resumed=True)
        assert "hf_checkpoint:" in caplog.text

    def test_a_changed_config_is_refused(self, tmp_path) -> None:
        """The config decides the architecture the PLE table plugs into."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        _train_and_save(base, ckpt)
        (base / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 5}}))
        with pytest.raises(PleProvenanceError, match="config.json differs"):
            on_load(base, load_dir=ckpt, iteration=4, resumed=True)

    def test_a_missing_base_is_refused(self, tmp_path) -> None:
        """A checkpoint bound to a PLE table cannot resume without it."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        _train_and_save(base, ckpt)
        shutil.rmtree(base)
        with pytest.raises(PleProvenanceError, match="missing or not a Qwen3.8 checkpoint"):
            on_load(base, load_dir=ckpt, iteration=4, resumed=True)

    def test_a_trained_checkpoint_without_a_manifest_needs_an_explicit_adoption(self, tmp_path, monkeypatch) -> None:
        """Checkpoints from before this check: refuse, or adopt the current base on request."""
        base, ckpt = _make_base(tmp_path / "base"), tmp_path / "ckpt"
        (ckpt / "iter_0000004").mkdir(parents=True)
        with pytest.raises(PleProvenanceError, match="is missing"):
            on_load(base, load_dir=ckpt, iteration=4, resumed=True)
        monkeypatch.setenv("MILES_PLE_PROVENANCE_ADOPT", "1")
        assert on_load(base, load_dir=ckpt, iteration=4, resumed=True)["ple"] == compute_manifest(base)["ple"]


class TestFreshStartAndExport:
    def test_a_fresh_start_can_be_pinned_to_an_expected_manifest(self, tmp_path, monkeypatch) -> None:
        """MILES_PLE_MANIFEST checks the base before the first step, e.g. against the manifest in lock.json."""
        base = _make_base(tmp_path / "base")
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps(compute_manifest(base)))
        monkeypatch.setenv("MILES_PLE_MANIFEST", str(expected))
        on_load(base, load_dir=None, iteration=0, resumed=False)

        key = f"{PLE}.ngram_embedding.shard_0.weight"
        shutil.rmtree(base)
        _make_base(base, overrides={key: _tensor(key, 9)})
        with pytest.raises(PleProvenanceError, match="does not match"):
            on_load(base, load_dir=None, iteration=0, resumed=False)

    def test_an_export_must_carry_the_runs_ple_bytes(self, tmp_path) -> None:
        """complete_hf_export copies PLE from --hf-checkpoint; it must still be the run's table."""
        base = _make_base(tmp_path / "base")
        manifest = on_load(base, load_dir=None, iteration=0, resumed=False)
        hashes = {k: e["sha256"] for k, e in manifest["ple"].items()}
        check_export(hashes)
        with pytest.raises(PleProvenanceError, match="not the ones this run trained with"):
            check_export(hashes | {f"{PLE}.layer_multipliers": "0" * 64})

    def test_model_save_writes_the_manifest_next_to_the_checkpoint(self, tmp_path, monkeypatch) -> None:
        """The save path, not only the helper, leaves iter_NNNNNNN/ple_provenance.json."""
        base = _make_base(tmp_path / "base")
        on_load(base, load_dir=None, iteration=0, resumed=False)
        monkeypatch.setattr(model_module, "should_disable_forward_pre_hook", lambda args: False)
        monkeypatch.setattr(model_module, "is_lora_model", lambda model: False)
        monkeypatch.setattr(model_module, "save_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(model_module, "check_tp_replicas", lambda *a, **k: 0, raising=False)
        args = make_trainer_args(ci_test=False, save=str(tmp_path / "ckpt"))
        model_module.save(args=args, iteration=12, model=[torch.nn.Linear(2, 2)], optimizer=None, opt_param_scheduler=None)
        saved = json.loads((tmp_path / "ckpt" / "iter_0000012" / "ple_provenance.json").read_text())
        assert saved["ple"] == compute_manifest(base)["ple"]


def _distributed_worker(rank: int, world_size: int, port: int, *, base: str, out: str) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        manifest = compute_manifest(base)
        if rank == 0:
            Path(out).write_text(json.dumps(manifest))
    finally:
        dist.destroy_process_group()


def test_the_hashing_split_over_ranks_gives_the_single_process_manifest(tmp_path) -> None:
    """Each rank hashes a slice of the keys; the gathered manifest is the same."""
    base = _make_base(tmp_path / "base")
    out = tmp_path / "manifest.json"
    run_multiprocess(partial(_distributed_worker, base=str(base), out=str(out)), world_size=2)
    assert json.loads(out.read_text()) == compute_manifest(base)
