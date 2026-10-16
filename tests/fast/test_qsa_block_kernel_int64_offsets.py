"""Every row-sized pointer offset in the packed QSA kernel and its index widens to int64 first.

At 262K packed tokens a row index times a row stride can pass 2^31 (the old bitmap kernel
faulted at ~92.7K tokens this way). The widening must happen before the multiply
(``query.to(tl.int64) * stride``); converting the product is too late. The kernels only run on
GPU, so this pins it in the source; the 262K packed forward / backward against fp64 runs in
``tests/fast-gpu/test_qwen3_8_next_qsa_long_packed.py``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = {
    "block_attention": ROOT / "miles/kernels/attention/qsa/block_attention.py",
    "block_index": ROOT / "miles/kernels/attention/qsa/block_index.py",
}
# a multiply by one of these scales a query, key, block or tile index into an address
_ROW_SCALE = re.compile(r"\s\*\s(stride_\w+|q_words|k_words|words|width|HQ)\b")
# head indices are below the head count, so head strides stay small
_SMALL = {"head", "kv_head"}


def _jit_functions(src: str) -> dict[str, list[str]]:
    """``{name: body lines}`` of every ``@triton.jit`` function."""
    out, name, body = {}, None, []
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("def ") and i and lines[i - 1].strip() == "@triton.jit":
            name, body = line[4:].split("(")[0], []
            out[name] = body
        elif line and not line[0].isspace() and not line.startswith(")"):
            # a multi-line signature closes with an unindented ")"; anything else ends the function
            name = None
        elif name is not None:
            body.append(line)
    return out


def _left_operand(text: str, end: int) -> str:
    i, depth = end, 0
    while i > 0:
        c = text[i - 1]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and (c.isspace() or c in "+-,=*"):
            break
        i -= 1
    return text[i:end]


def narrow_row_offsets(src: str) -> tuple[int, list[str]]:
    """(row-scaled multiplies checked, the ones whose left operand is not int64)."""
    checked, bad = 0, []
    for name, body in _jit_functions(src).items():
        widened = {m.group(1) for line in body if (m := re.match(r"\s*(\w+) = .*\.to\(tl\.int64\)\s*$", line))}
        for line in body:
            for m in _ROW_SCALE.finditer(line):
                operand = _left_operand(line, m.start())
                base = operand.split("[")[0].split(".")[0].strip("()")
                if base in _SMALL:
                    continue
                checked += 1
                if not (operand.endswith(".to(tl.int64)") or base in widened or base.endswith("64")):
                    bad.append(f"{name}: {line.strip()}")
    return checked, bad


@pytest.mark.parametrize("kernel,minimum", [("block_attention", 20), ("block_index", 8)])
def test_every_row_offset_widens_before_the_stride(kernel, minimum):
    checked, bad = narrow_row_offsets(KERNELS[kernel].read_text())
    assert checked >= minimum, f"only {checked} row offsets found; the pattern no longer matches the kernel"
    assert not bad, bad


def test_the_check_catches_a_narrow_row_offset():
    """The old bitmap kernel's fault, and a widening applied after the multiply."""
    src = (
        "@triton.jit\n"
        "def _k(SEL, Q, offs_q, row, stride_st, stride_qs):\n"
        "    a = tl.load(SEL + offs_q[:, None] * stride_st)\n"
        "    b = tl.load(Q + (row * stride_qs).to(tl.int64))\n"
        "    row64 = row.to(tl.int64)\n"
        "    c = tl.load(Q + row64 * stride_qs)\n"
        "\n"
        "\n"
        "@triton.jit\n"
        "def _multi_line(\n"
        "    Q,\n"
        "    row,\n"
        "    stride_qs,\n"
        "):\n"
        "    d = tl.load(Q + row * stride_qs)\n"
    )
    checked, bad = narrow_row_offsets(src)
    assert checked == 4
    assert len(bad) == 3 and "offs_q" in bad[0] and "(row * stride_qs)" in bad[1]
    assert bad[2].startswith("_multi_line: "), "the body after a multi-line signature was not read"
