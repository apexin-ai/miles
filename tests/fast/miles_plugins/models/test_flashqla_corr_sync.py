"""CPU tests for miles_plugins.models.flashqla_corr_sync and its gdn_kernel hook (no GPU needed)."""

import hashlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from miles_plugins.models import flashqla_corr_sync as fcs

PKG = "flash_qla.ops.gated_delta_rule.chunk"
BLACKWELL_MOD = f"{PKG}.blackwell.cp_fwd"


def _installed(rel: str) -> Path | None:
    try:
        spec = importlib.util.find_spec("flash_qla")  # locates the package without executing it
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / rel
    return path if path.exists() else None


def _module(name, file=None, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    if file is not None:
        m.__file__ = str(file)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _real_fla_backend():
    try:
        return importlib.import_module(fcs.FLA_FLASHQLA_BACKEND_MODULE)
    except Exception:
        return None


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(fcs, "_applied", None)
    monkeypatch.setattr(fcs, "_fla_hooked", False)
    monkeypatch.setattr(fcs, "_blackwell_device", lambda: True)
    for var in (fcs.ENV, "FLA_FLASH_QLA", "FLA_DISABLE_BACKEND_DISPATCH"):
        monkeypatch.delenv(var, raising=False)
    q = _real_fla_backend()
    if q is not None:  # restore fla's class method after tests that install the hook
        monkeypatch.setattr(q.FlashQLABackend, "chunk_gated_delta_rule", q.FlashQLABackend.chunk_gated_delta_rule)
    return monkeypatch


def install_fake_flashqla(monkeypatch, tmp_path, cp_fwd_mod=BLACKWELL_MOD, review=True,
                          cp_fwd_file=None, cp_context_file=None):
    """Fake flash_qla package whose cp_context/cp_fwd sources are files on disk (optionally reviewed)."""
    def original_correct_terminal_states(**kw):
        return "original-backward"

    def original_correct_initial_states(**kw):
        return "original-forward"

    original_correct_terminal_states.__module__ = cp_fwd_mod
    if cp_fwd_file is None:
        cp_fwd_file = tmp_path / "cp_fwd_fake.py"
        cp_fwd_file.write_text("# fake cp_fwd\n")
    if cp_context_file is None:
        cp_context_file = tmp_path / "cp_context_fake.py"
        cp_context_file.write_text("# fake cp_context\n")
    if review:
        monkeypatch.setitem(fcs.REVIEWED_CP_FWD_SHA256, hashlib.sha256(Path(cp_fwd_file).read_bytes()).hexdigest(), "t")
        monkeypatch.setitem(fcs.REVIEWED_CP_CONTEXT_SHA256,
                            hashlib.sha256(Path(cp_context_file).read_bytes()).hexdigest(), "t")
    cp_fwd = _module(cp_fwd_mod, cp_fwd_file, correct_terminal_states=original_correct_terminal_states)
    cp_context = _module(f"{PKG}.cp_context", cp_context_file,
                         correct_terminal_states=original_correct_terminal_states,
                         correct_initial_states=original_correct_initial_states)
    mods = {"flash_qla": _module("flash_qla", chunk_gated_delta_rule=lambda *a, **k: "flashqla-kernel"),
            "flash_qla.ops": _module("flash_qla.ops"),
            "flash_qla.ops.gated_delta_rule": _module("flash_qla.ops.gated_delta_rule"),
            PKG: _module(PKG), f"{PKG}.cp_context": cp_context, cp_fwd_mod: cp_fwd}
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    return cp_context, cp_fwd, original_correct_terminal_states, original_correct_initial_states


def _reference_fp64(raw_dht, buf, mt, fb, r2c):
    S, H = fb.shape
    out = torch.empty(buf.shape, dtype=torch.float64)
    for b in range(len(r2c) - 1):
        start, end = int(r2c[b]), int(r2c[b + 1])
        for h in range(H):
            st = raw_dht[b, h].double() if raw_dht is not None else torch.zeros(buf.shape[2:], dtype=torch.float64)
            for idx in range(end - 1, start, -1):
                out[idx, h] = st
                new = buf[idx, h].double()
                if fb[idx, h]:
                    new = new + mt[idx, h].double().T @ st
                st = new
            out[start, h] = st
    return out


class TestMode:
    def test_default_off_does_not_import_flash_qla(self, env):
        env.delitem(sys.modules, "flash_qla", raising=False)
        assert fcs.maybe_apply() is None
        assert fcs.apply_for_backend("fla") is None
        assert "flash_qla" not in sys.modules

    @pytest.mark.parametrize("value", ["", "0", "off", "OFF", "none"])
    def test_off_values(self, env, value):
        env.setenv(fcs.ENV, value)
        assert fcs.requested_mode() is None

    def test_invalid_value_raises(self, env):
        env.setenv(fcs.ENV, "stage1")
        with pytest.raises(ValueError):
            fcs.requested_mode()


class TestApplyState:
    @pytest.mark.parametrize("after", ["off", None, "stages1"])
    def test_cannot_change_or_disable_after_torch(self, env, tmp_path, after):
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path)
        env.setenv(fcs.ENV, "torch")
        assert fcs.maybe_apply() == "torch"
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch
        assert fcs.maybe_apply() == "torch"  # idempotent
        if after is None:
            env.delenv(fcs.ENV)
        else:
            env.setenv(fcs.ENV, after)
        with pytest.raises(RuntimeError, match="cannot|not supported"):
            fcs.maybe_apply()
        with pytest.raises(RuntimeError):
            fcs.apply_for_backend("fla")
        with pytest.raises(RuntimeError):
            fcs.apply_for_backend("flashqla")
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch

    @pytest.mark.parametrize("mode", ["torch", "stages1"])
    def test_unreviewed_sources_rejected_in_both_modes(self, env, tmp_path, mode):
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path, review=False)
        env.setenv(fcs.ENV, mode)
        with pytest.raises(RuntimeError, match="not reviewed"):
            fcs.maybe_apply()
        assert cpx.correct_terminal_states is original and fcs._applied is None

    def test_rejects_if_already_patched_by_someone_else(self, env, tmp_path):
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path)
        cpx.correct_terminal_states = lambda **kw: "other"
        cpx.correct_terminal_states.__module__ = BLACKWELL_MOD
        env.setenv(fcs.ENV, "torch")
        with pytest.raises(RuntimeError, match="not the original"):
            fcs.maybe_apply()

    def test_non_blackwell_device_checked_before_flashqla_import(self, env, tmp_path):
        install_fake_flashqla(env, tmp_path)
        env.setattr(fcs, "_blackwell_device", lambda: False)
        imported = []
        real_import = fcs.importlib.import_module
        env.setattr(fcs.importlib, "import_module", lambda n, *a, **k: imported.append(n) or real_import(n, *a, **k))
        env.setenv(fcs.ENV, "stages1")
        with pytest.raises(RuntimeError, match="Blackwell"):
            fcs.maybe_apply(strict_arch=True)
        assert fcs.maybe_apply(strict_arch=False) is None
        assert fcs.CP_CONTEXT_MODULE not in imported and fcs._applied is None

    def test_non_blackwell(self, env, tmp_path):
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path, cp_fwd_mod=f"{PKG}.hopper.cp_fwd")
        env.setenv(fcs.ENV, "torch")
        with pytest.raises(RuntimeError, match="Blackwell"):
            fcs.maybe_apply(strict_arch=True)
        assert fcs.maybe_apply(strict_arch=False) is None
        assert cpx.correct_terminal_states is original and fcs._applied is None


class TestRealLoader:
    def test_stages1_installs_patched_module_from_real_source(self, env, tmp_path):
        cp_fwd_file = _installed("ops/gated_delta_rule/chunk/blackwell/cp_fwd.py")
        cp_context_file = _installed("ops/gated_delta_rule/chunk/cp_context.py")
        if cp_fwd_file is None or cp_context_file is None or importlib.util.find_spec("tilelang") is None:
            pytest.skip("flash_qla sources and tilelang required")
        cpx, cp_fwd, original, original_fwd = install_fake_flashqla(
            env, tmp_path, review=False, cp_fwd_file=cp_fwd_file, cp_context_file=cp_context_file)
        env.setenv(fcs.ENV, "stages1")
        env.setenv("MILES_FLASHQLA_PATCH_DIR", str(tmp_path / "cache"))
        assert fcs.maybe_apply() == "stages1"
        new = cpx.correct_terminal_states
        assert new is not original
        assert new.__module__.startswith("miles_flashqla_cp_fwd_stages1_")
        loaded = sys.modules.get(new.__module__) or new.__globals__
        src_file = Path(new.__globals__["__file__"])
        assert src_file.parent == tmp_path / "cache" / "miles_flashqla_corr_sync"
        text = src_file.read_text()
        assert text.count("T.Pipelined(num_iters - 1, num_stages=1)") == 1
        assert "num_stages=2)" not in text
        assert new.__globals__["tilelang_correct_h0"] is not None
        assert cpx.correct_initial_states is original_fwd  # forward untouched
        assert fcs.maybe_apply() == "stages1" and cpx.correct_terminal_states is new  # no reinstall


class TestLoaderContentCheck:
    def test_stale_cache_file_is_rejected(self, env, tmp_path):
        cp_fwd_file = _installed("ops/gated_delta_rule/chunk/blackwell/cp_fwd.py")
        if cp_fwd_file is None:
            pytest.skip("flash_qla sources required")
        patched = fcs.patch_cp_fwd_source(cp_fwd_file.read_text())
        digest = hashlib.sha256(patched.encode()).hexdigest()[:16]
        cache = tmp_path / "cache" / "miles_flashqla_corr_sync"
        cache.mkdir(parents=True)
        (cache / f"cp_fwd_stages1_{digest}.py").write_text("x = 1\n")   # stale / foreign content
        env.setenv("MILES_FLASHQLA_PATCH_DIR", str(tmp_path / "cache"))
        env.setattr(fcs.os, "replace", lambda a, b: None)              # publish step fails silently
        with pytest.raises(RuntimeError, match="does not hold the patched source"):
            fcs._load_stages1_module(types.SimpleNamespace(__file__=str(cp_fwd_file)))


class TestTorchScan:
    @pytest.mark.parametrize("with_raw", [False, True])
    def test_matches_fp64_reference_and_copies_exactly(self, with_raw):
        g = torch.Generator().manual_seed(0)
        r2c = torch.tensor([0, 23, 24, 26], dtype=torch.int32)
        S, H, K = 26, 3, 16
        buf = torch.randn(S, H, K, K, generator=g)
        mt = torch.randn(S, H, K, K, generator=g) * 0.1
        fb = torch.rand(S, H, generator=g) < 0.5
        fb[[0, 23, 24]] = False
        # unwritten mt blocks hold arbitrary data (NaN or finite garbage) and must not leak
        nan_block = (torch.rand(S, H, generator=g) < 0.5)[:, :, None, None]   # whole blocks NaN or finite
        garbage = torch.where(nan_block, torch.full((S, H, K, K), float("nan")),
                              torch.randn(S, H, K, K, generator=g) * 100)
        mt = torch.where(fb[:, :, None, None], mt, garbage)
        raw = torch.randn(3, H, K, K, generator=g) if with_raw else None
        out = fcs.correct_terminal_states_torch(raw, buf, mt, fb, r2c)
        ref = _reference_fp64(raw, buf, mt, fb, r2c)
        assert torch.isfinite(out).all()
        assert (out.double() - ref).abs().max().item() < 1e-4
        if not with_raw:
            for h in range(H):
                clean = True
                for idx in range(21, -1, -1):
                    clean = clean and not bool(fb[idx + 1, h])
                    if clean:
                        assert torch.equal(out[idx, h], buf[idx + 1, h])

    def test_state_v_first_rejected(self):
        z = torch.zeros(2, 1, 4, 4)
        with pytest.raises(NotImplementedError):
            fcs.correct_terminal_states_torch(None, z, z, torch.zeros(2, 1, dtype=torch.bool),
                                              torch.tensor([0, 2]), state_v_first=True)


class TestPatchSource:
    def test_rejects_unreviewed_source(self):
        with pytest.raises(RuntimeError, match="not reviewed"):
            fcs.patch_cp_fwd_source("x = 1\n")

    def test_installed_source_patched_once(self):
        path = _installed("ops/gated_delta_rule/chunk/blackwell/cp_fwd.py")
        if path is None:
            pytest.skip("flash_qla not installed")
        text = path.read_text()
        patched = fcs.patch_cp_fwd_source(text)
        assert patched.count("num_stages=1):  # miles flashqla_corr_sync") == 1
        assert "T.Pipelined(num_iters - 1, num_stages=2)" not in patched
        assert len(patched.splitlines()) == len(text.splitlines())


def _linear_attn():
    """The real gdn_kernel source from linear_attn.py, executed on its own.

    Importing linear_attn needs megatron/CUDA libraries; gdn_kernel itself only uses function-local
    imports, so compiling exactly its AST node tests the shipped code without those dependencies."""
    import ast

    path = Path(fcs.__file__).with_name("linear_attn.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gdn_kernel")
    ns = {"__name__": "linear_attn_gdn_kernel_under_test"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return types.SimpleNamespace(gdn_kernel=ns["gdn_kernel"])


def _real_dispatch(monkeypatch, disabled_at_import):
    """fla's real dispatch() and BaseBackend, loaded from the installed source with the given import-time
    FLA_DISABLE_BACKEND_DISPATCH, plus a registry holding fla's real FlashQLABackend."""
    q = _real_fla_backend()
    if q is None:
        pytest.skip("fla FlashQLA backend not importable")
    src = Path(importlib.util.find_spec("fla.ops.backends").origin)
    monkeypatch.setattr(torch.compiler, "disable", lambda fn: fn)
    if disabled_at_import:
        monkeypatch.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1")
    spec = importlib.util.spec_from_file_location("m039_fla_backends_copy", src)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    op = f"m039_cpu_gdr_{disabled_at_import}"
    reg = b.BackendRegistry(op)
    reg.register(q.FlashQLABackend())
    b.BackendRegistry._initialized.add(op)

    @b.dispatch(op)
    def chunk_gated_delta_rule(q, k, v, g, beta, **kwargs):
        return "fla-native"

    return q, chunk_gated_delta_rule


def _gdr_args(dtype=torch.bfloat16, K=128):
    t = lambda *s: torch.zeros(*s, dtype=dtype)
    return (t(1, 8, 1, K), t(1, 8, 1, K), t(1, 8, 1, 128), torch.zeros(1, 8, 1), torch.zeros(1, 8, 1, dtype=dtype))


class TestRealFlaDispatch:
    def _arm(self, env, tmp_path, disabled_at_import=False, sm100=True, hopper=False):
        q, fn = _real_dispatch(env, disabled_at_import)
        env.setattr(q, "IS_NVIDIA_SM100", sm100)
        env.setattr(q, "IS_NVIDIA_HOPPER", hopper)
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path)
        env.setenv(fcs.ENV, "torch")
        la = _linear_attn()
        la.gdn_kernel("fla")  # installs the hook; must not import flash_qla code paths beyond the fake
        return fn, cpx, original

    def test_dispatched_call_applies_patch(self, env, tmp_path):
        fn, cpx, original = self._arm(env, tmp_path)
        assert fcs._applied is None and cpx.correct_terminal_states is original  # lazy until dispatched
        assert fn(*_gdr_args()) == "flashqla-kernel"
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch

    @pytest.mark.parametrize("case", ["inter_card_cp", "k64", "fp32", "fla_flash_qla_0"])
    def test_verifier_or_switch_rejection_leaves_kernel_untouched(self, env, tmp_path, case):
        fn, cpx, original = self._arm(env, tmp_path)
        kwargs = {}
        args = _gdr_args()
        if case == "inter_card_cp":
            kwargs["cp_context"] = object()
        elif case == "k64":
            args = _gdr_args(K=64)
        elif case == "fp32":
            args = _gdr_args(dtype=torch.float32)
        else:
            env.setenv("FLA_FLASH_QLA", "0")
        assert fn(*args, **kwargs) == "fla-native"
        assert cpx.correct_terminal_states is original and fcs._applied is None

    def test_sm80_never_touches_flashqla(self, env, tmp_path):
        fn, cpx, original = self._arm(env, tmp_path, sm100=False, hopper=False)
        env.setattr(fcs, "_blackwell_device", lambda: False)
        assert fn(*_gdr_args()) == "fla-native"
        assert cpx.correct_terminal_states is original and fcs._applied is None

    def test_hopper_flashqla_path_skips_patch(self, env, tmp_path):
        fn, cpx, original = self._arm(env, tmp_path, sm100=False, hopper=True)
        env.setattr(fcs, "_blackwell_device", lambda: False)
        assert fn(*_gdr_args()) == "flashqla-kernel"
        assert cpx.correct_terminal_states is original and fcs._applied is None

    def test_dispatch_disabled_at_fla_import(self, env, tmp_path):
        fn, cpx, original = self._arm(env, tmp_path, disabled_at_import=True)
        assert fn(*_gdr_args()) == "fla-native"
        assert cpx.correct_terminal_states is original

    def test_disable_env_after_fla_import_does_not_stop_dispatch(self, env, tmp_path):
        fn, cpx, original = self._arm(env, tmp_path)
        env.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1")  # fla fixed its state at import, as here
        assert fn(*_gdr_args()) == "flashqla-kernel"
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch


class TestGdnKernelHook:
    def _fake_fla(self, monkeypatch):
        def fla_kernel(*a, **k):
            return "fla-kernel"

        for n in ["fla", "fla.ops", "fla.ops.gated_delta_rule"]:
            monkeypatch.setitem(sys.modules, n, _module(n))
        sys.modules["fla.ops.gated_delta_rule"].chunk_gated_delta_rule = fla_kernel
        return fla_kernel

    def test_off_changes_nothing_and_imports_no_flashqla(self, env, tmp_path):
        la = _linear_attn()
        fla_kernel = self._fake_fla(env)
        env.delitem(sys.modules, "flash_qla", raising=False)
        before = set(sys.modules)
        assert la.gdn_kernel("fla") is fla_kernel
        assert not any(m.startswith("flash_qla") for m in set(sys.modules) - before)
        cpx, _, original, _ = install_fake_flashqla(env, tmp_path)
        for backend in ("fla", "flashqla"):
            la.gdn_kernel(backend)
        assert cpx.correct_terminal_states is original and fcs._applied is None and not fcs._fla_hooked

    def test_flashqla_backend_patches(self, env, tmp_path):
        la = _linear_attn()
        cpx, _, original, original_fwd = install_fake_flashqla(env, tmp_path)
        env.setenv(fcs.ENV, "torch")
        k = la.gdn_kernel("flashqla")
        assert k() == "flashqla-kernel"
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch
        assert cpx.correct_initial_states is original_fwd
        la.gdn_kernel("flashqla")
        assert fcs._applied == "torch"

    def test_fla_hook_does_not_import_flashqla(self, env):
        if _real_fla_backend() is None:
            pytest.skip("fla FlashQLA backend not importable")
        la = _linear_attn()
        env.setitem(sys.modules, "flash_qla", None)  # importing FlashQLA would raise
        env.setenv(fcs.ENV, "stages1")
        la.gdn_kernel("fla")
        assert fcs._fla_hooked and fcs._applied is None

    def test_mode_change_after_apply_raises_through_gdn_kernel(self, env, tmp_path):
        la = _linear_attn()
        install_fake_flashqla(env, tmp_path)
        env.setenv(fcs.ENV, "torch")
        la.gdn_kernel("flashqla")
        env.delenv(fcs.ENV)
        for backend in ("fla", "flashqla"):
            with pytest.raises(RuntimeError):
                la.gdn_kernel(backend)


@pytest.mark.skipif(os.environ.get("MILES_TEST_TILELANG_LOWERING") != "1",
                    reason="set MILES_TEST_TILELANG_LOWERING=1 (needs tilelang + flash_qla source, ~1 min CPU)")
def test_stages1_lowers_without_double_buffering(tmp_path):
    """Lower original and patched tilelang_correct_h0 (backward variant) to CUDA on CPU."""
    path = _installed("ops/gated_delta_rule/chunk/blackwell/cp_fwd.py")
    if path is None:
        pytest.skip("flash_qla not installed")
    script = tmp_path / "lower.py"
    script.write_text(
        "import importlib.util, re, sys\n"
        "import tilelang, tvm\n"
        "src, out = sys.argv[1], sys.argv[2]\n"
        "t = open(src).read()\n"
        "m = re.search(r'@tilelang\\.jit\\((.*?)\\)\\ndef tilelang_correct_h0\\(', t, re.S)\n"
        "t = t[:m.start()] + 'def tilelang_correct_h0(' + t[m.end():]\n"
        "p = out + '.py'; open(p, 'w').write(t)\n"
        "s = importlib.util.spec_from_file_location('k', p); k = importlib.util.module_from_spec(s); s.loader.exec_module(k)\n"
        "f = k.tilelang_correct_h0(H=12, DK=128, DV=128, res_dtype='float32', accum_dtype='float32',"
        " buffer_dtype='float32', seqlen_dtype='int32', mask_dtype='bool', use_raw_h0=False,"
        " state_v_first=False, reverse=True, transpose_m=True)\n"
        "tg = tvm.target.Target({'kind': 'cuda', 'arch': 'sm_103a'})\n"
        "with tvm.transform.PassContext(opt_level=3), tg:\n"
        "    open(out, 'w').write(tilelang.lower(f, target=tg).kernel_source)\n"
    )
    srcs = {"orig": path.read_text(), "s1": fcs.patch_cp_fwd_source(path.read_text())}
    cu = {}
    for name, text in srcs.items():
        src = tmp_path / f"cp_fwd_{name}.py"
        src.write_text(text)
        out = tmp_path / f"{name}.cu"
        subprocess.run([sys.executable, "-B", str(script), str(src), str(out)], check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True)
        cu[name] = out.read_text()
    assert "(i_s & 1) * 4096" in cu["orig"]
    assert "(i_s & 1) * 4096" not in cu["s1"]
    assert "mbarrier_mem[4]" in cu["s1"]
