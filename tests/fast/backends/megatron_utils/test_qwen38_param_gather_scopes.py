"""CPU contracts for the PERF-F v4 scope hooks (Codex RECHECK1 N1): a module that reads its sub-modules' weights
functionally (as GDN's in_proj_sections / project and gated norm do) must not see a bucket before its gather. A stand-in
DDP: Megatron's per-module pre-hook (recurse=False, the real _make_forward_pre_hook logic) and bucket groups whose
finish_param_sync completes the gather (copies the owner value) only when called."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from miles.backends.megatron_utils import model as mm


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


class GDNLike(nn.Module):
    """reads a.weight / b.weight / norm.weight without calling them (like linear_attn.project / forward)"""
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.ones(2))
        self.a, self.b, self.norm = Proj(), Proj(), Proj()

    def forward(self, x):
        y = F.linear(x, torch.cat([self.a.weight, self.b.weight]))
        return y[:, :2] * self.norm.weight[0, 0] * self.A_log[0]


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = GDNLike()

    def forward(self, x):
        return self.mixer(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.embed = Proj()

    def forward(self, x):
        x = x * self.embed.weight[0, 0]
        for layer in self.decoder.layers:
            x = layer(x)
        return x.sum()


class Group:
    """a bucket group: its parameters hold 1 (previous step) until finish_param_sync gathers 2 (the update)"""
    def __init__(self, params, log):
        self.params, self.log, self.done = params, log, False

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        if not self.done:
            with torch.no_grad():
                for p in self.params:
                    p.fill_(2.0)
            self.done = True
            self.log.append(id(self))


class Chunk(nn.Module):
    def __init__(self, split):
        super().__init__()
        self.module = Model()
        self.log = []
        self.param_to_bucket_group = {}
        mixers = [layer.mixer for layer in self.module.decoder.layers]
        groups = []
        for m in mixers:   # Codex's legal split: the parent's own parameter in one bucket, the read weights in another
            g0 = Group([m.A_log], self.log)
            g1 = Group([m.a.weight, m.b.weight, m.norm.weight], self.log) if split else g0
            if not split:
                g0.params += [m.a.weight, m.b.weight, m.norm.weight]
            groups += [g0, g1]
            for p in g0.params:
                self.param_to_bucket_group[p] = g0
            for p in g1.params:
                self.param_to_bucket_group[p] = g1
        ge = Group([self.module.embed.weight], self.log)
        self.param_to_bucket_group[self.module.embed.weight] = ge
        self.remove_forward_pre_hook_handles = {}
        for mod in self.module.modules():   # Megatron DDP._make_forward_pre_hook: own parameters only
            def hook(module, *unused):
                for p in module.parameters(recurse=False):
                    if p in self.param_to_bucket_group:
                        self.param_to_bucket_group[p].finish_param_sync()
            self.remove_forward_pre_hook_handles[mod] = mod.register_forward_pre_hook(hook)

    def forward(self, x):
        return self.module(x)


def test_functional_reader_sees_the_previous_step_without_scopes():
    chunk = Chunk(split=True)
    seen = []
    chunk.module.decoder.layers[0].mixer.register_forward_hook(lambda m, i, o: seen.append(float(m.a.weight[0, 0])))
    orig = GDNLike.forward

    def spy(self, x):
        seen.append(float(self.a.weight[0, 0]))   # the value project() reads
        return orig(self, x)
    GDNLike.forward = spy
    try:
        chunk(torch.ones(3, 2))
    finally:
        GDNLike.forward = orig
    assert seen[0] == 1.0   # the PERFFREV counterexample: read before its bucket was gathered


def test_scopes_make_every_read_follow_its_gather(monkeypatch):
    chunk = Chunk(split=True)
    calls = []
    real = mm._qwen38_finish_scope
    monkeypatch.setattr(mm, "_qwen38_finish_scope", lambda c, s, g: (calls.append(s), real(c, s, g)))
    scopes = mm.install_param_gather_scopes([chunk])
    assert [s for s, *_ in scopes] == ["c0:module.decoder.layers.0", "c0:module.decoder.layers.1", "c0:outer"]
    seen = []
    orig = GDNLike.forward

    def spy(self, x):
        seen.append((float(self.a.weight[0, 0]), float(self.norm.weight[0, 0])))
        return orig(self, x)
    GDNLike.forward = spy
    try:
        chunk(torch.ones(3, 2))
    finally:
        GDNLike.forward = orig
    assert seen == [(2.0, 2.0), (2.0, 2.0)]
    assert calls[0] == "c0:outer" and calls[1:] == ["c0:module.decoder.layers.0", "c0:module.decoder.layers.1"]
    assert mm.install_param_gather_scopes([chunk]) == scopes   # once per chunk


def test_scopes_idle_while_the_ddp_hooks_are_off():
    chunk = Chunk(split=True)
    mm.install_param_gather_scopes([chunk])
    for h in chunk.remove_forward_pre_hook_handles.values():
        h.remove()
    chunk.remove_forward_pre_hook_handles = {}
    chunk(torch.ones(3, 2))
    assert chunk.log == []


def test_chunk_without_decoder_layers_refused():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Proj()
            self.param_to_bucket_group = {}
    with pytest.raises(ValueError, match="no decoder layers"):
        mm.install_param_gather_scopes([Bare()])


def test_train_installs_scopes_with_the_flag():
    import inspect
    src = inspect.getsource(mm.train)
    i = src.index("_refuse_trainable_vision(model)")
    assert src.index("install_param_gather_scopes(model)") > i


def test_parameter_shared_between_layers_refused():
    chunk = Chunk(split=True)
    layers = chunk.module.decoder.layers
    layers[1].mixer.a = layers[0].mixer.a   # one module (and its Parameter) in two layers
    with pytest.raises(ValueError, match="several scopes"):
        mm.install_param_gather_scopes([chunk])


def test_parameter_shared_between_a_layer_and_the_model_refused():
    chunk = Chunk(split=True)
    chunk.module.head = chunk.module.decoder.layers[0].mixer.norm   # an outer reader of a layer's weight
    with pytest.raises(ValueError, match="several scopes"):
        mm.install_param_gather_scopes([chunk])


def test_several_chunks_refused():
    with pytest.raises(ValueError, match="one model chunk"):
        mm.install_param_gather_scopes([Chunk(split=True), Chunk(split=True)])


def test_read_then_scope_is_not_detectable_by_the_digest():
    """RECHECK2 R4: a reader that runs before the scope barrier and the barrier afterwards (the gather then
    completes normally) leaves correct contents at the barrier: a digest cannot see the earlier read. It is prevented by
    the topology rule (every reader inside the layer that owns the parameter, the layer pre-hook first), not detected."""
    chunk = Chunk(split=True)
    mm.install_param_gather_scopes([chunk])
    m = chunk.module.decoder.layers[0].mixer
    early = float(m.a.weight[0, 0])                       # a read outside / before the layer
    mm._qwen38_finish_scope(chunk, "c0:module.decoder.layers.0", [chunk.param_to_bucket_group[m.a.weight]])
    assert early == 1.0 and float(m.a.weight[0, 0]) == 2.0
