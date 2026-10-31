"""The QSA indexer has no training objective, so its parameters must stay out of the optimizer.

It runs under ``no_grad``, but that alone does not protect it: Megatron's DDP gives
every ``requires_grad`` parameter a (zero) main_grad, and the distributed optimizer
then applies decoupled weight decay to it every step.
"""

import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
from megatron.core.optimizer import OptimizerConfig, _get_param_groups  # noqa: E402

from miles_plugins.models.qwen3_8_next.ops import qsa_indexer  # noqa: E402

HIDDEN = 16
HEADS = 2
HEAD_DIM = 8


class _DuplicatedLinear(torch.nn.Module):
    """CPU stand-in for TELinear(parallel_mode="duplicated")."""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * 0.1)

    def forward(self, x):
        return x @ self.weight.t(), None


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        init_method=None,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=HEADS,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=HEAD_DIM,
        qwen3_8_next_indexer_budget=8,
        qwen3_8_next_indexer_compress_ratio=4,
    )


class _Layer(torch.nn.Module):
    """An indexer next to a trainable projection, as in a QSA attention layer."""

    def __init__(self):
        super().__init__()
        self.indexer = qsa_indexer.Qwen38NextQSAIndexer(_config(), layer_number=1)
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)


def _digest(module):
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture
def layer(monkeypatch, tmp_path):
    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    # A file rendezvous avoids a fixed port; destroy only a group this fixture created, so later
    # tests (e.g. the indexer monitor) do not see a stray initialized process group.
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    torch.manual_seed(0)
    try:
        yield _Layer()
    finally:
        if created:
            dist.destroy_process_group()


def test_indexer_parameters_do_not_require_grad(layer):
    params = dict(layer.indexer.named_parameters())
    assert set(params) == {"index_qk_proj.weight", "q_layernorm", "k_layernorm"}
    assert not any(p.requires_grad for p in params.values())


def test_upstream_freeze_globs_cover_every_indexer_parameter(layer):
    """The shared helper's globs, not a blanket requires_grad_(False), must reach every parameter."""
    from miles_plugins.models.indexer import freeze_indexer_parameters

    frozen = freeze_indexer_parameters(layer.indexer, qsa_indexer.INDEXER_PARAM_GLOBS)
    assert set(frozen) == {name for name, _ in layer.indexer.named_parameters()}


def test_indexer_parameters_are_in_no_optimizer_param_group(layer):
    groups = _get_param_groups([layer], OptimizerConfig(lr=1e-3, weight_decay=0.1), None)
    grouped = {id(p) for g in groups for p in g["params"]}
    assert not grouped & {id(p) for p in layer.indexer.parameters()}
    assert {id(p) for p in layer.proj.parameters()} <= grouped


def test_indexer_weights_survive_training_steps_with_weight_decay(layer):
    groups = _get_param_groups([layer], OptimizerConfig(lr=1e-3, weight_decay=0.1), None)
    optimizer = torch.optim.AdamW(groups, lr=1e-2, weight_decay=0.1)
    before = _digest(layer.indexer)
    seq = 12
    rope = torch.randn(64, 1, 1, HEAD_DIM)
    for _ in range(5):
        hidden = torch.randn(seq, HIDDEN)
        with torch.no_grad():
            selection = layer.indexer(hidden, torch.arange(seq), rope)
        loss = (layer.proj(hidden) * (selection >= 0).float().mean()).square().mean()
        optimizer.zero_grad()
        loss.backward()
        for group in groups:
            for p in group["params"]:
                # Megatron DDP hands every buffered parameter a main_grad, zero if unused
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        optimizer.step()
    assert _digest(layer.indexer) == before


def test_provider_check_rejects_a_trainable_indexer(layer):
    from miles_plugins.models.qwen3_8_next.model_provider import assert_qsa_indexers_frozen

    wrapper = torch.nn.ModuleDict({"self_attention": layer})
    assert_qsa_indexers_frozen(wrapper)
    layer.indexer.index_qk_proj.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="self_attention.indexer.index_qk_proj.weight"):
        assert_qsa_indexers_frozen(wrapper)
