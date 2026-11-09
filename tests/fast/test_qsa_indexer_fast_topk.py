"""``--qwen38-indexer-fast-topk``: the Triton top-k (``ops/qsa_topk.py``) gives ``select_topk_blocks``' block ids.

The kernel runs here under Triton's CPU interpreter (``TRITON_INTERPRET=1``, set in a subprocess before
Triton is imported) on cases built to hit every branch: exact ties at the k-th place (more than the
scratch row holds, too), ``-0.0`` next to ``+0.0``, ``-inf`` masks (rows with fewer candidates than k,
or none), keys that share their high bits (all radix passes), widths that are not a multiple of the
chunk, ``kk == width``. Every case must give the reference ids bit for bit:
``select_topk_blocks(logits, kk)`` then ``torch.where(values == -inf, -1, idx + col_lo)``.

The indexer with the flag on is checked the same way end to end (``iter_selected_blocks`` and
``qsa_selected_blocks``, packed and TP-sharded rows), and its tile loop must not read the device. On
CPU tensors without the interpreter the flag falls back to the reference path.

``tests/fast-gpu/test_qwen3_8_next_indexer_fast_topk.py`` runs the same cases on a GPU.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from miles_plugins.models.qwen3_8_next.ops import attention, qsa_indexer, qsa_topk  # noqa: E402

# the interpreter cases take about 60 s on a 2-CPU pod
register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["miles-plugin"])

REPO_ROOT = Path(__file__).resolve().parents[2]
NEG_INF = float("-inf")


def reference_block_ids(logits, kk, col_lo=0):
    """What ``iter_selected_blocks`` writes into ``block_idx[:, :kk]`` without the flag."""
    values, idx = qsa_indexer.select_topk_blocks(logits.clone(), kk)
    return torch.where(values == NEG_INF, -1, idx + col_lo)


def adversarial_cases(seed=0, scale=1):
    """``[(name, logits, kk, col_lo, kernel kwargs)]``; ``scale`` > 1 widens the rows (GPU runs).

    Small ``cap`` / ``digit`` / ``block`` force the radix passes, the many-ties branch and several chunks
    on CPU-sized rows. NaN appears once per row only: with two, neither path has a fixed answer.
    """
    g = torch.Generator().manual_seed(seed)
    w = lambda n: n * scale  # noqa: E731
    small = dict(block=64, num_warps=4, digit=5, cap=32)
    wide = dict(block=64, num_warps=4)
    cases = []

    def add(name, logits, kk, col_lo=0, **kw):
        cases.append((name, logits.contiguous(), kk, col_lo, kw))

    add("normal", torch.randn(7, w(300), generator=g), 20, 5, **wide)
    add("normal-cap32", torch.randn(6, w(700), generator=g), 20, 3, **small)
    add("few-levels", torch.randint(0, 4, (7, w(300)), generator=g).float(), 33, **wide)
    add("few-levels-cap32", torch.randint(0, 3, (6, w(700)), generator=g).float(), 20, **small)
    # more equal scores at the k-th place than the scratch row holds
    many_ties = torch.full((3, w(500)), 2.0)
    many_ties[:, ::7] = 3.0
    add("ties-over-cap", many_ties, 30, 11, **small)
    add("all-equal", torch.ones(4, w(200)), 16, **small)
    # -inf masks: causal triangle, a row with no candidate, rows with fewer candidates than kk
    masked = torch.rand(5, w(90), generator=g)
    masked[:, w(40) :] = NEG_INF
    masked[2] = NEG_INF
    masked[3, 5:] = NEG_INF
    add("masked", masked, 50, 7, **wide)
    tri = torch.rand(64, w(128), generator=g)
    cols = torch.arange(w(128))
    tri = tri.masked_fill(cols.unsqueeze(0) >= (torch.arange(64) * 2 * scale).unsqueeze(1), NEG_INF)
    add("causal-triangle", tri, 16, 100, **small)
    # keys that share the high bits: 1 + m * 2^-23 (one exponent), and two exponents
    add("one-exponent", 1 + torch.randint(0, 6, (6, w(257)), generator=g).float() * 2**-23, 17, **wide)
    mant = 1 + torch.randint(0, 2**20, (6, w(700)), generator=g).float() * 2**-23
    add("two-exponents-cap32", mant * torch.randint(1, 3, (6, w(700)), generator=g).float(), 25, **small)
    # signed zeros compare equal: ties among them keep the lower column
    zeros = torch.zeros(4, w(200))
    zeros[0, ::3] = -0.0
    zeros[1] = -0.0
    zeros[2, 1::2] = -0.0
    zeros[3, : w(50)] = 1.0
    add("signed-zeros", zeros, 10, **wide)
    neg = -torch.rand(5, w(400), generator=g)
    neg[0, : w(100)] = -0.0
    neg[0, w(50) : w(150)] = 0.0
    neg[2, 3::5] = NEG_INF
    add("negative-and-zeros-cap32", neg, 32, **small)
    add("kk-equals-width", torch.randn(3, w(64), generator=g), w(64), **wide)
    add("kk-one", torch.randint(0, 2, (5, w(130)), generator=g).float(), 1, **small)
    add("width-under-block", torch.randn(4, 37, generator=g), 9, **wide)
    big = torch.randn(4, w(300), generator=g) * 1e30
    big[:, ::4] = 3.4e38
    add("huge-values", big, 40, **small)
    tiny = torch.randn(4, w(300), generator=g) * 1e-40  # subnormals
    add("subnormals", tiny, 40, **small)
    # one NaN per row, below and above the k-th place: NaN ranks first, as in torch.topk / torch.sort
    nan = torch.rand(4, w(300), generator=g)
    nan[torch.arange(4), torch.tensor([0, 17, w(150), w(299)])] = float("nan")
    add("one-nan-per-row", nan, 12, **small)
    # the indexer's own form: a ReLU sum of 4 head scores over sqrt(d), bf16 operands
    q = torch.randn(48, 4, 32, generator=g).bfloat16().float()
    k = torch.randn(w(256), 32, generator=g).bfloat16().float()
    relu = torch.einsum("mhd,nd->mnh", q, k).relu_().sum(-1) / math.sqrt(32)
    add("relu-sum", relu, 24, 2, **wide)
    add("relu-sum-cap32", relu, 24, 2, **small)
    return cases


_INTERPRETER_SCRIPT = r"""
import json, sys
import torch
sys.path.insert(0, sys.argv[1])
from tests.fast.test_qsa_indexer_fast_topk import adversarial_cases, reference_block_ids
from miles_plugins.models.qwen3_8_next.ops.qsa_topk import select_topk_block_ids
out = []
for name, logits, kk, col_lo, kw in adversarial_cases(int(sys.argv[2])):
    want = reference_block_ids(logits, kk, col_lo)
    got = select_topk_block_ids(logits, kk, col_lo, **kw)
    bad = (got != want).any(1).nonzero().flatten().tolist()
    out.append({"name": name, "equal": bool(torch.equal(got, want)), "bad_rows": bad[:5],
                "got": got[bad[0]].tolist()[:12] if bad else None, "want": want[bad[0]].tolist()[:12] if bad else None})
print("RESULT " + json.dumps(out))
"""


def _run_interpreted(script, *args, timeout=900):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script, str(REPO_ROOT), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, f"rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("seed", [0, 1])
def test_kernel_gives_the_reference_ids(seed):
    results = _run_interpreted(_INTERPRETER_SCRIPT, seed)
    assert len(results) == len(adversarial_cases(seed))
    bad = [r for r in results if not r["equal"]]
    assert not bad, bad


def test_cases_reach_every_branch():
    """The cases do what the docstring says: ties over the scratch row, empty rows, radix passes."""
    names = {name: (logits, kk, kw) for name, logits, kk, _, kw in adversarial_cases(0)}
    logits, kk, kw = names["ties-over-cap"]
    kth = logits.sort(dim=1, descending=True).values[:, kk - 1 : kk]
    assert ((logits == kth).sum(1) > kw["cap"]).all()
    assert (names["masked"][0] == NEG_INF).all(1).any()
    for name in ("normal-cap32", "two-exponents-cap32", "relu-sum-cap32"):
        logits, kk, kw = names[name]
        assert logits.shape[1] > kw["cap"] and kk < kw["cap"]
    assert (torch.signbit(names["signed-zeros"][0]) & (names["signed-zeros"][0] == 0)).any()


def test_wrapper_rejects_bad_arguments():
    x = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="kk"):
        qsa_topk.select_topk_block_ids(x, 0)
    with pytest.raises(ValueError, match="kk"):
        qsa_topk.select_topk_block_ids(x, 9)
    # CPU tensors outside the interpreter are not the kernel's
    if os.environ.get("TRITON_INTERPRET") != "1":
        assert not qsa_topk.fast_topk_supported(x)
        with pytest.raises(ValueError, match="CUDA"):
            qsa_topk.select_topk_block_ids(x, 2)
    assert not qsa_topk.fast_topk_supported(torch.zeros(2, 8, dtype=torch.float64))
    assert not qsa_topk.fast_topk_supported(torch.zeros(8, 2).t())


# ---- the indexer with the flag ----

HIDDEN = 64
RATIO = 4


class _DuplicatedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))

    def forward(self, x):
        return x @ self.weight.t(), None


def build_indexer(fast_topk, score_dtype="fp32", tile_bytes=1 << 16, budget=64, head_dim=32, monkeypatch=None):
    if monkeypatch is None:  # the interpreter subprocess
        qsa_indexer.TELinear = _DuplicatedLinear
    else:
        monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        init_method=None,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=4,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=head_dim,
        qwen3_8_next_indexer_budget=budget,
        qwen3_8_next_indexer_compress_ratio=RATIO,
        qwen3_8_next_indexer_tile_bytes=tile_bytes,
        qwen3_8_next_indexer_score_dtype=score_dtype,
        qwen3_8_next_indexer_fast_topk=fast_topk,
    )
    torch.manual_seed(0)
    indexer = qsa_indexer.Qwen38NextQSAIndexer(config, layer_number=1)
    with torch.no_grad():
        indexer.q_layernorm.normal_(0, 0.1)
        indexer.k_layernorm.normal_(0, 0.1)
    return indexer


def indexer_inputs(lens, seed=0, quantize=False):
    g = torch.Generator().manual_seed(seed)
    total = sum(lens)
    hidden = torch.randn(total, HIDDEN, generator=g)
    if quantize:
        # coarse hidden states: many equal block scores, so the tie rule decides
        hidden = hidden.round()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    positions = attention.Qwen38NextAttention._packed_positions(cu, total)
    rope = torch.randn(max(lens), 1, 1, 16, generator=g)
    return hidden, positions, rope, cu


_E2E_SCRIPT = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
from tests.fast.test_qsa_indexer_fast_topk import build_indexer, indexer_inputs
from miles_plugins.models.qwen3_8_next.ops import attention, qsa_indexer
out = []
for lens, quant, shard in [([300], False, None), ([97, 160, 41, 2], True, None), ([257, 300], False, (1, 4)),
                           ([200, 200], True, (3, 4))]:
    hidden, positions, rope, cu = indexer_inputs(lens, quantize=quant)
    cu_host = tuple(cu.tolist())
    res = {}
    for fast in (False, True):
        indexer = build_indexer(fast)
        with torch.no_grad():
            got = []
            for start, end, ids in indexer.iter_selected_blocks(hidden, positions, rope, cu, row_shard=shard,
                                                                cu_seqlens_host=cu_host):
                got.append(ids.clone())
            res[fast] = torch.cat(got)
            res[(fast, "tokens")] = indexer(hidden, positions, rope, cu)
    out.append({"lens": lens, "shard": shard, "blocks_equal": bool(torch.equal(res[False], res[True])),
                "tokens_equal": bool(torch.equal(res[(False, "tokens")], res[(True, "tokens")])),
                "selected": int((res[False] >= 0).sum())})
# the tile loop with the flag reads nothing back from the device
hidden, positions, rope, cu = indexer_inputs([257, 300])
indexer = build_indexer(True)
calls = []
qsa_indexer.select_topk_blocks = lambda *a, **k: calls.append("select_topk_blocks")
real = {n: getattr(torch.Tensor, n) for n in ("tolist", "item", "__bool__")}
seen = []
in_kernel = [False]
kernel = qsa_indexer.select_topk_block_ids
def kernel_call(*a, **k):
    in_kernel[0] = True
    try:
        return kernel(*a, **k)
    finally:
        in_kernel[0] = False
qsa_indexer.select_topk_block_ids = kernel_call
def spy(name):
    def f(self, *a, **k):
        if not in_kernel[0]:
            seen.append(name)
        return real[name](self, *a, **k)
    return f
gen = indexer.iter_selected_blocks(hidden, positions, rope, cu, row_shard=(0, 4), cu_seqlens_host=tuple(cu.tolist()))
with torch.no_grad():
    first = next(gen)  # project + layout run before the first tile
    for n in real:
        setattr(torch.Tensor, n, spy(n))
    try:
        rest = list(gen)
    finally:
        for n, f in real.items():
            setattr(torch.Tensor, n, f)
print("RESULT " + json.dumps({"cases": out, "reference_calls": calls, "tiles": 1 + len(rest), "reads": seen}))
"""


def test_indexer_with_the_flag_selects_the_same_blocks_and_reads_nothing_back():
    """iter_selected_blocks / forward with the flag == without it (packed, TP rows, ties); no device read
    in the tile loop. Under the interpreter, Triton itself may read tensors; the spy counts only reads made
    while no kernel runs, so a read here is the indexer's own."""
    result = _run_interpreted(_E2E_SCRIPT)
    for case in result["cases"]:
        assert case["blocks_equal"] and case["tokens_equal"], case
        assert case["selected"] > 0
    assert result["reference_calls"] == []
    assert result["tiles"] > 1
    assert result["reads"] == [], result["reads"]


def test_cpu_tensors_fall_back_to_the_reference(monkeypatch):
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("the interpreter runs the kernel on CPU tensors")
    hidden, positions, rope, cu = indexer_inputs([97, 160, 41, 2], quantize=True)
    with torch.no_grad():
        want, _ = attention.qsa_selected_blocks(
            build_indexer(False, monkeypatch=monkeypatch), hidden, positions, rope, cu
        )
        got, _ = attention.qsa_selected_blocks(
            build_indexer(True, monkeypatch=monkeypatch), hidden, positions, rope, cu
        )
    assert torch.equal(got, want)


def test_flag_default_and_spec(monkeypatch):
    from miles.utils.args.configs.custom_megatron_plugins import Qwen38VisionMegatronPluginsConfig
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    assert Qwen38VisionMegatronPluginsConfig.model_fields["qwen38_indexer_fast_topk"].default is False
    assert build_indexer(True, monkeypatch=monkeypatch).fast_topk is True
    old = build_indexer(False, monkeypatch=monkeypatch)
    del old.config.qwen3_8_next_indexer_fast_topk
    assert qsa_indexer.Qwen38NextQSAIndexer(old.config, layer_number=1).fast_topk is False

    for flag in (True, False):
        config = SimpleNamespace(virtual_pipeline_model_parallel_size=2)
        args = SimpleNamespace(qwen38_indexer_fast_topk=flag, hf_checkpoint="/x")
        args.backend = SimpleNamespace(position_embedding_type="rope")
        monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
        monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
        monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
        with pytest.raises(NotImplementedError, match="interleaved pipeline"):
            qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
        assert config.qwen3_8_next_indexer_fast_topk is flag
