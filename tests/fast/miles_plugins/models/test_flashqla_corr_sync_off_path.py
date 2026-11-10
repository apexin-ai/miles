"""MILES_FLASHQLA_CORR_SYNC off (unset or an off value): the GDN kernel call path is apex/main's.

Base = apex/main 2d441bfa (= 09a5adcf + docs). The only change this branch makes to linear_attn.py is the two-statement
hook at the top of gdn_kernel (import of flashqla_corr_sync.apply_for_backend and its call). These tests prove that
(1) at source level, linear_attn.py minus exactly those two statements is apex/main's linear_attn.py (AST equal,
pinned by sha256), and (2) at run time, with the switch off, the patched gdn_kernel returns the same object as
apex/main's gdn_kernel for every backend, raises the same errors, imports nothing beyond flashqla_corr_sync itself
(no FlashQLA, no fla backend module), and leaves FlashQLA's cp_context and fla's FlashQLABackend untouched.
"""
import ast
import hashlib
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from miles_plugins.models import flashqla_corr_sync as fcs

BASE_REF = "apex/main 2d441bfa"
BASE_GDN_KERNEL_AST_SHA256 = "967141940051fa6c42e7d8525c041244fdcc8d9befe2b93f1890c5945f041808"
BASE_MODULE_AST_SHA256 = "cacefc7ddef116d344a9a40d001ad10cd5b0f6757c787fbf27ff88d984f741b2"
HOOK_MODULE = "miles_plugins.models.flashqla_corr_sync"
OFF_VALUES = (None, "", "0", "off", "none", "false", "OFF", " Off ")


def _is_hook(stmt) -> bool:
    if isinstance(stmt, ast.ImportFrom):
        return (stmt.module == HOOK_MODULE and len(stmt.names) == 1
                and stmt.names[0].name == "apply_for_backend" and stmt.names[0].asname == "apply_flashqla_corr_sync")
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        c = stmt.value
        return (isinstance(c.func, ast.Name) and c.func.id == "apply_flashqla_corr_sync" and not c.keywords
                and len(c.args) == 1 and isinstance(c.args[0], ast.Name) and c.args[0].id == "backend")
    return False


def _sources():
    path = Path(fcs.__file__).with_name("linear_attn.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gdn_kernel")
    return path, tree, node


def _stripped(node):
    """gdn_kernel without the hook; asserts the hook is exactly its first two statements."""
    assert [_is_hook(s) for s in node.body[:2]] == [True, True], "hook must be the first two statements"
    assert not any(_is_hook(s) for s in node.body[2:]), "no other hook statement allowed"
    new = ast.FunctionDef(name=node.name, args=node.args, body=node.body[2:], decorator_list=node.decorator_list,
                          returns=node.returns, type_comment=node.type_comment,
                          **({"type_params": node.type_params} if hasattr(node, "type_params") else {}))
    return ast.copy_location(new, node)


def _sha(n) -> str:
    return hashlib.sha256(ast.dump(n).encode()).hexdigest()


def test_source_equals_apex_main_except_the_two_hook_statements():
    _, tree, node = _sources()
    base_fn = _stripped(node)
    assert _sha(base_fn) == BASE_GDN_KERNEL_AST_SHA256, f"gdn_kernel minus the hook differs from {BASE_REF}"
    tree.body = [base_fn if n is node else n for n in tree.body]
    assert _sha(tree) == BASE_MODULE_AST_SHA256, f"linear_attn.py minus the hook differs from {BASE_REF}"


def _compile(fn_node, path):
    ns = {"__name__": "gdn_kernel_variant"}
    mod = ast.fix_missing_locations(ast.Module(body=[fn_node], type_ignores=[]))
    exec(compile(mod, str(path), "exec"), ns)
    return ns["gdn_kernel"]


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def world(monkeypatch):
    """Fake fla (kernel + FlashQLA backend class) and fake flash_qla (kernel + cp_context), state reset."""
    monkeypatch.setattr(fcs, "_applied", None)
    monkeypatch.setattr(fcs, "_fla_hooked", False)
    monkeypatch.setattr(fcs, "_blackwell_device", lambda: True)

    def fla_kernel(*a, **k):
        return "fla"

    def flashqla_kernel(*a, **k):
        return "flashqla"

    def original_correction(**kw):
        return "original"

    class FlashQLABackend:
        def chunk_gated_delta_rule(self, *a, **k):
            return "backend"

    mods = {"fla": _module("fla"), "fla.ops": _module("fla.ops"),
            "fla.ops.gated_delta_rule": _module("fla.ops.gated_delta_rule", chunk_gated_delta_rule=fla_kernel),
            "fla.ops.gated_delta_rule.backends": _module("fla.ops.gated_delta_rule.backends"),
            fcs.FLA_FLASHQLA_BACKEND_MODULE: _module(fcs.FLA_FLASHQLA_BACKEND_MODULE, FlashQLABackend=FlashQLABackend),
            "flash_qla": _module("flash_qla", chunk_gated_delta_rule=flashqla_kernel),
            fcs.CP_CONTEXT_MODULE: _module(fcs.CP_CONTEXT_MODULE, correct_terminal_states=original_correction)}
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    return types.SimpleNamespace(fla_kernel=fla_kernel, flashqla_kernel=flashqla_kernel, cls=FlashQLABackend,
                                 method=FlashQLABackend.chunk_gated_delta_rule, original=original_correction,
                                 cpx=mods[fcs.CP_CONTEXT_MODULE], mp=monkeypatch)


def _call(fn, backend):
    before = set(sys.modules)
    try:
        out = ("ok", fn(backend))
    except Exception as e:
        out = ("raise", type(e), str(e))
    return out, set(sys.modules) - before


@pytest.mark.parametrize("value", OFF_VALUES)
@pytest.mark.parametrize("backend", ["fla", "flashqla", "torch", "kda"])
def test_off_runtime_path_identical_to_apex_main(world, value, backend):
    path, _, node = _sources()
    patched = _compile(node, path)
    base = _compile(_stripped(node), path)
    if value is None:
        world.mp.delenv(fcs.ENV, raising=False)
    else:
        world.mp.setenv(fcs.ENV, value)
    rb, db = _call(base, backend)
    rp, dp = _call(patched, backend)
    assert rp == rb, (rp, rb)                                   # same object / same exception type and message
    assert dp - db <= {HOOK_MODULE}, dp - db                     # nothing else imported (no FlashQLA, no backend)
    assert world.cpx.correct_terminal_states is world.original   # FlashQLA's correction untouched
    assert world.cls.chunk_gated_delta_rule is world.method      # fla's FlashQLA backend untouched
    assert fcs._applied is None and not fcs._fla_hooked
    if backend == "fla":
        assert rp == ("ok", world.fla_kernel)
    if backend == "flashqla":
        assert rp == ("ok", world.flashqla_kernel)


def test_off_apply_for_backend_returns_none_and_reads_only_the_switch(world):
    world.mp.delenv(fcs.ENV, raising=False)
    before = set(sys.modules)
    for backend in ("fla", "flashqla", "torch"):
        assert fcs.apply_for_backend(backend) is None
    assert set(sys.modules) == before
    assert world.cpx.correct_terminal_states is world.original and world.cls.chunk_gated_delta_rule is world.method


def test_on_value_is_not_the_off_path(world):
    """Positive control: with the switch on, the fla backend IS hooked (so the off tests can see a difference)."""
    world.mp.setenv(fcs.ENV, "hdirect")
    path, _, node = _sources()
    _compile(node, path)("fla")
    assert fcs._fla_hooked and world.cls.chunk_gated_delta_rule is not world.method
