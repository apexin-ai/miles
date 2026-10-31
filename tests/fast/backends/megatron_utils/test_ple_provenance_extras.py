"""PLE provenance behaviour that test_ple_provenance.py does not cover.

- ``MILES_PLE_MANIFEST`` pins a new binding to a known manifest (e.g. the one in the run's lock file);
- the manifest records the HF revision of a hub download;
- load_model_state picks the manifest directory: none for an in-memory checkpoint, the adapter's iteration
  directory for a LoRA training state.

Same harness as test_ple_provenance.py: the real ``model.load_model_state`` / ``model.save`` and miles'
``load_checkpoint`` on a small synthetic base, Megatron's checkpoint I/O replaced by ``FakeMegatron``.
"""

import json
import shutil
from pathlib import Path

import pytest
from tests.fast.backends.megatron_utils.test_ple_provenance import (
    SHARD_KEY,
    _args,
    _fake_megatron_load,
    _load,
    _make_base,
    _resume,
    _train,
)

from miles.backends.megatron_utils import checkpoint as checkpoint_module
from miles.backends.megatron_utils import model as model_module
from miles.backends.megatron_utils import ple_provenance


@pytest.fixture(autouse=True)
def _no_gpu_side_effects(monkeypatch):
    monkeypatch.setattr(ple_provenance, "_run_manifest", None)
    monkeypatch.delenv("MILES_PLE_PROVENANCE_ADOPT", raising=False)
    monkeypatch.delenv("MILES_PLE_MANIFEST", raising=False)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(model_module, "should_disable_forward_pre_hook", lambda args: False)
    monkeypatch.setattr(model_module, "is_lora_model", lambda model: False)


def _replace_base(base: Path) -> None:
    shutil.rmtree(base)
    _make_base(base, changed=SHARD_KEY)


class TestPinnedManifest:
    def test_a_new_binding_is_verified_against_the_pinned_manifest(self, tmp_path, monkeypatch, caplog) -> None:
        base = _make_base(tmp_path / "base")
        pinned = tmp_path / "pinned.json"
        pinned.write_text(json.dumps(ple_provenance.compute_manifest(base)))
        monkeypatch.setenv("MILES_PLE_MANIFEST", str(pinned))
        with caplog.at_level("INFO"):
            _load(monkeypatch, _args(base, tmp_path / "empty", tmp_path / "ckpt"))
        assert "verified" in caplog.text

        _replace_base(base)
        with pytest.raises(
            ple_provenance.PleProvenanceError, match=r"does not match .*pinned\.json.*1 PLE tensors differ"
        ):
            _load(monkeypatch, _args(base, tmp_path / "empty", tmp_path / "ckpt"))

    def test_a_resume_checks_the_checkpoints_own_manifest_not_the_pinned_one(self, tmp_path, monkeypatch) -> None:
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        other = _make_base(tmp_path / "other", changed=SHARD_KEY)
        pinned = tmp_path / "pinned.json"
        pinned.write_text(json.dumps(ple_provenance.compute_manifest(other)))
        monkeypatch.setenv("MILES_PLE_MANIFEST", str(pinned))
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 4


def test_the_manifest_records_the_hf_revision_of_a_hub_download(tmp_path) -> None:
    base = _make_base(tmp_path / "base")
    download = base / ".cache" / "huggingface" / "download"
    download.mkdir(parents=True)
    for name in ("config.json", "model-00001-of-00002.safetensors"):
        (download / f"{name}.metadata").write_text("de4b8e4d43b917e7706784d8bb445c9af86a3540\nsha\n0\n")
    manifest = ple_provenance.compute_manifest(base)
    assert manifest["revision"] == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert manifest["ple"][SHARD_KEY]["dtype"] == "BF16"
    assert len(manifest["config_sha256"]) == len(manifest["index_sha256"]) == 64


def test_no_hf_checkpoint_is_no_binding(tmp_path) -> None:
    assert ple_provenance.on_load(None, trained_checkpoint_dir=None) is None
    assert ple_provenance.run_manifest() is None


class TestManifestDirectory:
    def test_an_in_memory_checkpoint_starts_a_new_binding(self, tmp_path, monkeypatch) -> None:
        """No sidecar travels with a local_checkpoint_manager checkpoint, so nothing is verified."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        _replace_base(base)
        args = _args(base, ckpt, ckpt)
        monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", _fake_megatron_load(args))
        model_module.load_model_state(
            args,
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context={"local_checkpoint_manager": object()},
            resume_from_ckpt=True,
        )
        old = json.loads((ckpt / "iter_0000004" / "ple_provenance.json").read_text())
        assert ple_provenance.run_manifest()["ple"][SHARD_KEY] != old["ple"][SHARD_KEY]

    def test_a_lora_training_state_is_verified_against_the_adapters_iteration(self, tmp_path, monkeypatch) -> None:
        """A native adapter lives in <save>/iter_NNNNNNN/adapter, next to the manifest of that iteration."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        adapter = ckpt / "iter_0000004" / "adapter"
        adapter.mkdir()
        monkeypatch.setattr(model_module, "is_lora_enabled", lambda args: True)
        monkeypatch.setattr(model_module, "restore_lora_adapter", lambda *a, **k: 4)

        def load():
            return _load(
                monkeypatch, _args(base, tmp_path / "empty2", tmp_path / "ckpt2", lora_adapter_path=str(adapter))
            )

        assert load().loaded_rollout_id == 4
        _replace_base(base)
        with pytest.raises(
            ple_provenance.PleProvenanceError, match=r"iter_0000004/ple_provenance\.json.*1 PLE tensors differ"
        ):
            load()
