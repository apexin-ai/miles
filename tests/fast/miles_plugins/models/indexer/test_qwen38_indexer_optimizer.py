"""Frozen Qwen3.8 QSA indexers must keep their checkpoint weights across AdamW steps."""

import pytest
import torch
import torch.distributed as dist
from megatron.core.optimizer import OptimizerConfig, _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.qwen3_8_next.ops import qsa_indexer


@pytest.fixture
def gloo(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _model():
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
    config.qwen3_8_next_indexer_n_heads = 2
    config.qwen3_8_next_indexer_kv_heads = 1
    config.qwen3_8_next_indexer_head_dim = 4
    config.qwen3_8_next_indexer_budget = 4
    config.qwen3_8_next_indexer_compress_ratio = 2
    config.indexer_topk_backend = "torch"
    model = torch.nn.Module()
    model.indexer = qsa_indexer.Qwen38NextQSAIndexer(config, layer_number=1)
    model.backbone = torch.nn.Linear(8, 1, bias=False)
    return model


def _optimizer(model):
    config = OptimizerConfig(lr=0.1, min_lr=0.1, weight_decay=0.2)
    groups = _get_param_groups([model], config, get_standard_config_overrides(config))
    optimizer = torch.optim.AdamW(groups, lr=config.lr, foreach=False)
    # Apply the same per-group weight decay multipliers as the Megatron trainer.
    OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=0,
        lr_decay_steps=4,
        lr_decay_style="constant",
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=4,
        wd_incr_style="constant",
    )
    return optimizer


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.backbone.weight.sum().backward()
    # MCore DDP owns a zero main_grad buffer even for unused trainable parameters.
    # Supply its CPU equivalent to AdamW. A None gradient would skip decay entirely.
    for param in model.indexer.parameters():
        if param.requires_grad:
            assert param.grad is None
            param.grad = torch.zeros_like(param)
    optimizer.step()


def _check_indexer(model, optimizer, expected):
    parameters = dict(model.indexer.named_parameters())
    assert parameters.keys() == expected.keys()
    for name, param in parameters.items():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0, msg=f"indexer weight changed: {name}")
    states = [name for name, param in parameters.items() if param in optimizer.state]
    assert not states, f"indexer has AdamW state: {states}"
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert all(id(param) not in optimized for param in parameters.values())
    assert all(not param.requires_grad for param in parameters.values())
    assert model.backbone.weight in optimizer.state


def test_qwen38_indexer_stays_frozen_across_steps_and_reload(gloo, monkeypatch, tmp_path):
    # The QSA constructor registers its usual parameters; only replace the CUDA projection.
    def cpu_projection(in_features, out_features, *, bias, **kwargs):
        return torch.nn.Linear(in_features, out_features, bias=bias)

    monkeypatch.setattr(qsa_indexer, "TELinear", cpu_projection)
    model = _model()
    with torch.no_grad():
        for i, param in enumerate(model.indexer.parameters(), start=1):
            param.fill_(i)
        model.backbone.weight.fill_(1)
    expected = {name: param.detach().clone() for name, param in model.indexer.named_parameters()}
    assert {"index_qk_proj.weight", "q_layernorm", "k_layernorm"} <= expected.keys()
    optimizer = _optimizer(model)
    for _ in range(2):
        _step(model, optimizer)
    _check_indexer(model, optimizer, expected)
    assert not torch.equal(model.backbone.weight, torch.ones_like(model.backbone.weight))

    path = tmp_path / "step2.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = _model()
    restored_optimizer = _optimizer(restored)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    _check_indexer(restored, restored_optimizer, expected)

    _step(model, optimizer)
    _step(restored, restored_optimizer)
    _check_indexer(restored, restored_optimizer, expected)
    torch.testing.assert_close(restored.backbone.weight, model.backbone.weight, rtol=0, atol=0)
    assert restored_optimizer.state[restored.backbone.weight]["step"].item() == 3
