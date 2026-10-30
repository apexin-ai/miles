import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


def _record(monkeypatch, tmp_path, overrides):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_8_next_vision_sft.py")
    call_entrypoint(module, "execute", overrides, sandbox=tmp_path)
    return "\n".join(recording.commands)


def test_full_visual_recipe_never_kills_existing_processes(monkeypatch, tmp_path):
    command = _record(monkeypatch, tmp_path, {"freeze_language": True})
    assert "pkill" not in command and "ray stop" not in command and "ray start" not in command
    for setting in (
        "--num-layers 48",
        "--position-embedding-type mrope",
        "--mrope-section 11 11 10",
        "--no-rope-fusion",
        "--qwen38-vision-trainable full",
        "--qwen38-freeze-language",
        "--num-steps-per-rollout 1",
        "--qwen38-visual-snapshot-interval 8",
        "--custom-megatron-init-path miles_plugins.models.qwen3_8_next.vision_audit.install_final_save_check",
    ):
        assert setting in command


def test_existing_process_cleanup_is_explicitly_opt_in_for_this_recipe(monkeypatch, tmp_path):
    assert "pkill" in _record(monkeypatch, tmp_path, {"skip_process_cleanup": False})


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_size": 3},
        {"pp_size": 3},
        {"ep_size": 3},
        {"freeze_language": True, "vision_trainable": "none"},
        {"global_batch_size": 16},
        {"audit": False},
    ],
)
def test_unqualified_layouts_fail_at_launch(overrides):
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_8_next_vision_sft.py")
    with pytest.raises(ValueError):
        module.ScriptArgs(**overrides)
