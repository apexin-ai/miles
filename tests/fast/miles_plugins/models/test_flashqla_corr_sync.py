"""CPU tests for miles_plugins.models.flashqla_corr_sync (no GPU, no flash_qla import needed)."""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from miles_plugins.models import flashqla_corr_sync as fcs

BLACKWELL_MOD = "flash_qla.ops.gated_delta_rule.chunk.blackwell.cp_fwd"


def _installed_cp_fwd() -> Path | None:
    spec = importlib.util.find_spec("flash_qla")  # locates the package without executing it
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "ops/gated_delta_rule/chunk/blackwell/cp_fwd.py"
    return path if path.exists() else None


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(fcs, "_applied", None)
    monkeypatch.delenv(fcs.ENV, raising=False)
    return monkeypatch


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
    def test_default_off_does_not_import_flash_qla(self, fresh):
        before = "flash_qla" in sys.modules
        assert fcs.maybe_apply() is None
        assert ("flash_qla" in sys.modules) == before

    @pytest.mark.parametrize("value", ["", "0", "off", "OFF", "none"])
    def test_off_values(self, fresh, value):
        fresh.setenv(fcs.ENV, value)
        assert fcs.requested_mode() is None

    def test_invalid_value_raises(self, fresh):
        fresh.setenv(fcs.ENV, "stage1")
        with pytest.raises(ValueError):
            fcs.requested_mode()


class TestTorchScan:
    @pytest.mark.parametrize("with_raw", [False, True])
    def test_matches_fp64_reference_and_copies_exactly(self, with_raw):
        g = torch.Generator().manual_seed(0)
        r2c = torch.tensor([0, 23, 24, 26], dtype=torch.int32)
        S, H, K = 26, 3, 16
        buf = torch.randn(S, H, K, K, generator=g)
        mt = torch.randn(S, H, K, K, generator=g) * 0.1
        fb = torch.rand(S, H, generator=g) < 0.5
        fb[[0, 23, 24]] = False                 # first segment of each raw sequence never corrects
        mt[~fb] = float("nan")                  # unwritten mt blocks must not leak
        raw = torch.randn(3, H, K, K, generator=g) if with_raw else None
        out = fcs.correct_terminal_states_torch(raw, buf, mt, fb, r2c)
        ref = _reference_fp64(raw, buf, mt, fb, r2c)
        assert torch.isfinite(out).all()
        assert (out.double() - ref).abs().max().item() < 1e-3
        if not with_raw:
            # cp[idx] is a plain copy of buf[idx+1] when no fallback occurs in idx+1 .. end-1
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
        path = _installed_cp_fwd()
        if path is None:
            pytest.skip("flash_qla not installed")
        text = path.read_text()
        patched = fcs.patch_cp_fwd_source(text)
        assert patched.count("num_stages=1):  # miles flashqla_corr_sync") == 1
        assert "T.Pipelined(num_iters - 1, num_stages=2)" not in patched
        assert len(patched.splitlines()) == len(text.splitlines())


class TestApplyWithFakeModules:
    def _install_fake(self, monkeypatch, module_name=BLACKWELL_MOD):
        def original(**kw):
            return "original"

        original.__module__ = module_name
        names = ["flash_qla", "flash_qla.ops", "flash_qla.ops.gated_delta_rule",
                 "flash_qla.ops.gated_delta_rule.chunk", "flash_qla.ops.gated_delta_rule.chunk.cp_context"]
        mods = {n: types.ModuleType(n) for n in names}
        mods[names[-1]].correct_terminal_states = original
        for i in range(len(names) - 1):
            setattr(mods[names[i]], names[i + 1].rsplit(".", 1)[1], mods[names[i + 1]])
        for n, m in mods.items():
            monkeypatch.setitem(sys.modules, n, m)
        return mods[names[-1]], original

    def test_torch_mode_replaces_backward_correction_once(self, fresh):
        cpx, original = self._install_fake(fresh)
        fresh.setenv(fcs.ENV, "torch")
        assert fcs.maybe_apply() == "torch"
        assert cpx.correct_terminal_states is fcs.correct_terminal_states_torch
        assert fcs.maybe_apply() == "torch"          # idempotent
        fresh.setenv(fcs.ENV, "stages1")
        with pytest.raises(RuntimeError, match="changed"):
            fcs.maybe_apply()

    def test_non_blackwell_path_refused(self, fresh):
        cpx, original = self._install_fake(fresh, "flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd")
        fresh.setenv(fcs.ENV, "torch")
        with pytest.raises(RuntimeError, match="Blackwell"):
            fcs.maybe_apply()
        assert cpx.correct_terminal_states is original


@pytest.mark.skipif(os.environ.get("MILES_TEST_TILELANG_LOWERING") != "1",
                    reason="set MILES_TEST_TILELANG_LOWERING=1 (needs tilelang + flash_qla source, ~1 min CPU)")
def test_stages1_lowers_without_double_buffering(tmp_path):
    """Lower original and patched tilelang_correct_h0 (backward variant) to CUDA on CPU."""
    path = _installed_cp_fwd()
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
