"""Typed-config contracts for the PR #5/#6 image providers.

Upstream #3496/#3507 passes the custom provider a frozen flat view via ``args=``
instead of exposing Megatron's global ``get_args()``. These tests drive the
providers through the real ``get_model_provider_func`` wrapper.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config, replace_config_values

from miles.backends.megatron_utils import model_provider
from miles.backends.training_utils import parallel
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.args.trainer_utils import compute_trainer_config


def _trainer_config(provider: str, *argv: str, **updates: Any):
    parsed = parse_megatron_test_config(
        "--custom-model-provider-path",
        provider,
        "--megatron-to-hf-mode",
        "raw",
        "--transformer-impl",
        "local",
        "--vocab-size",
        "8",
        "--make-vocab-size-divisible-by",
        "1",
        *argv,
    )
    args = compute_trainer_config(all_config=parsed, trainer=parsed.raw_megatron.trainers[0])
    # Set after parsing: the parser validates --hf-checkpoint against the toy Megatron shape.
    return replace_config_values(args, **updates)


@pytest.fixture
def base_model(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    constructed: list[dict[str, Any]] = []
    config = SimpleNamespace(use_kitchen=False, use_kitchen_attention=False, kitchen_attention_backend=None)

    def build_model(**kwargs: Any) -> torch.nn.Module:
        constructed.append(kwargs)
        result = torch.nn.Module()
        result.config = kwargs["config"]
        return result

    rank = SimpleNamespace(rank=0)
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(pp=rank, tp=rank, cp=rank, intra_dp=rank))
    monkeypatch.setattr(model_provider, "core_transformer_config_from_args", lambda _args: config)
    monkeypatch.setattr(model_provider, "get_gpt_layer_local_spec", lambda **kwargs: object())
    monkeypatch.setattr(model_provider, "GPTModel", build_model)
    return constructed


def _record_base_provider_args(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    original = model_provider.get_model_provider_func

    def recording(args, role="actor"):
        if args.custom_model_provider_path is None:
            seen.append(args)
        return original(args, role=role)

    monkeypatch.setattr(model_provider, "get_model_provider_func", recording)
    return seen


class TestQwenVisionProvider:
    PROVIDER = "miles_plugins.models.qwen3_8_next.vision.get_qwen38_vision_model_provider"

    def _args(self, tmp_path, **updates: Any):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
        return _trainer_config(
            self.PROVIDER,
            "--qkv-format",
            "thd",
            "--position-embedding-type",
            "mrope",
            "--mrope-section",
            "1",
            "1",
            "2",
            "--no-rope-fusion",
            "--qwen38-vision-trainable",
            "projector",
            "--qwen38-vision-recompute",
            hf_checkpoint=str(tmp_path),
            **updates,
        )

    def test_framework_args_reach_the_provider_and_base_model_is_built_once(
        self, monkeypatch, tmp_path, base_model
    ) -> None:
        """The provider reads plugin fields from the framework view and never recurses into itself."""
        from miles_plugins.models.qwen3_8_next import vision

        args = self._args(tmp_path)
        original = args.model_dump()
        wired = []
        monkeypatch.setattr(vision, "wire_visual", lambda model, config, **kwargs: wired.append(kwargs))
        seen = _record_base_provider_args(monkeypatch)

        model = model_provider.get_model_provider_func(args)(pre_process=True, post_process=True, vp_stage=None)

        assert args.model_dump() == original
        assert len(base_model) == 1 and len(seen) == 1
        assert isinstance(seen[0], ImmutableNamespace)
        assert wired == [dict(scope="projector", recompute=True, freeze_language=False, data="image")]
        assert len(model._forward_pre_hooks) == 0  # no PLE embedding in the stub model

    def test_provider_no_longer_reads_megatron_global_args(self, monkeypatch, tmp_path, base_model) -> None:
        """A stale get_args() call would fail here, since the global Megatron args are never set."""
        import megatron.training.global_vars as global_vars

        from miles_plugins.models.qwen3_8_next import vision

        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
        monkeypatch.setattr(vision, "wire_visual", lambda *a, **k: None)
        model_provider.get_model_provider_func(self._args(tmp_path))(pre_process=False, post_process=True)
        assert len(base_model) == 1

    def test_guards_read_backend_fields_from_the_flat_view(self, monkeypatch, tmp_path, base_model) -> None:
        from miles_plugins.models.qwen3_8_next import vision

        monkeypatch.setattr(vision, "wire_visual", lambda *a, **k: None)
        args = self._args(tmp_path, context_parallel_size=2)
        with pytest.raises(ValueError, match="CP1, THD"):
            model_provider.get_model_provider_func(args)(pre_process=True, post_process=True)
        assert base_model == []


class TestDeepSeekVisionProvider:
    PROVIDER = "miles_plugins.models.deepseek_v4_1.multimodal.deepseek_v41_vlm_model_provider"

    def test_framework_args_reach_the_provider_with_the_image_flag(self, monkeypatch, tmp_path, base_model) -> None:
        from miles_plugins.models.deepseek_v4_1 import audit, multimodal

        args = _trainer_config(
            self.PROVIDER,
            "--qkv-format",
            "bshd",
            "--dsv41-vision-trainable",
            "full",
            "--dsv41-freeze-language",
            hf_checkpoint=str(tmp_path),
        )
        original = args.model_dump()
        wired = []
        monkeypatch.setattr(multimodal, "wire_visual", lambda model, ckpt, **kwargs: wired.append((ckpt, kwargs)))
        monkeypatch.setattr(audit, "record_inventory", lambda model: None)
        seen = _record_base_provider_args(monkeypatch)

        model_provider.get_model_provider_func(args)(pre_process=True, post_process=True, vp_stage=None)

        assert args.model_dump() == original
        assert len(base_model) == 1 and len(seen) == 1
        assert seen[0].dsv41_image_sft is True
        assert wired == [(str(tmp_path), dict(scope="full", recompute=False, freeze_language=True))]
