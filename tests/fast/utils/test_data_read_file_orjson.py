"""read_file parses JSONL through orjson with a json.loads fallback (_json_loads): the same Python objects and the
same skipped lines as json.loads. Ported from the Qwen 35B training repo (829a424, test_miles_data_load.py).

MILES_REAL_JSONL=<a real pre-tokenized train.jsonl> also compares Datasets built from 400 rows spread over that file.
"""

import json
import math
import os
import random
import re
from pathlib import Path

import pytest

from miles.utils import data

pytestmark = pytest.mark.skipif(data.orjson is None, reason="orjson is not installed")


def same(a, b) -> bool:
    """Equal values of equal types all the way down (NaN equals NaN; 1 is not True, 1.0 is not 1)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    return a == b


def random_json(rng: random.Random, depth: int = 0):
    kind = rng.randrange(9 if depth < 4 else 6)
    if kind == 0:
        return rng.randrange(-(2**40), 2**40)
    if kind == 1:
        return rng.random() * 10 ** rng.randrange(-5, 6)
    if kind == 2:
        return "".join(rng.choice('abé中\U0001f600"\\\n') for _ in range(rng.randrange(6)))
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return [rng.randrange(248320) for _ in range(rng.randrange(50))]  # token ids / labels
    if kind == 5:
        return [-100] * rng.randrange(5) + [rng.randrange(10)]
    if kind in (6, 7):
        return {f"k{i}": random_json(rng, depth + 1) for i in range(rng.randrange(5))}
    return [random_json(rng, depth + 1) for _ in range(rng.randrange(5))]


EDGE_LINES = [
    '{"input_ids": [1, 2, 248319], "meta": {"labels": [-100, 2, 248319], "record_id": "a"}}',
    '{"a": "\\u4e2d\\u6587 \\ud83d\\ude00 tab\\t quote\\" slash\\\\/ nul\\u0000"}',
    '{"x": "\\ud800"}',  # lone surrogate: orjson refuses, json returns it
    '{"x": NaN, "y": Infinity, "z": -Infinity}',  # not JSON: orjson refuses, json accepts
    '{"big": 123456789012345678901234567890, "neg": -18446744073709551617}',  # beyond 64 bits: orjson -> float
    '{"edge": [18446744073709551615, 18446744073709551616, -9223372036854775808, -9223372036854775809]}',
    '{"ids": [1, 22, 333], "s": "1234567890123456789012 in a string"}',
    '{"f": 1.2345678901234567890123, "g": 12345678901234567890.5, "h": 1234567890123456789e-5}',
    '{"f": [0.1, -0.0, 1e-320, 1.7976931348623157e308, 1e400, 5e-324, 2.5e-07]}',
    '{"dup": 1, "dup": 2}',  # duplicate key: the last value wins
    '{"deep": [[[[[[[[[[{"a": []}]]]]]]]]]], "empty": {}, "e2": [], "s": ""}',
    '[1, "top-level list", null, true, false]',
    '"just a string"',
    "12",
    ' {"padded": 1} ',
]


@pytest.mark.parametrize("line", EDGE_LINES)
def test_json_loads_equals_json(line: str) -> None:
    assert same(data._json_loads(line), json.loads(line))


@pytest.mark.parametrize(
    "line", ['{"a": 1', '{"a": 1}}', "", '{"a": 01}', '{﻿"a": 1}', '﻿{"a": 1}', "{'a': 1}", '{"a": 1,}']
)
def test_json_loads_raises_like_json(line: str) -> None:
    with pytest.raises(json.JSONDecodeError) as want:
        json.loads(line)
    with pytest.raises(json.JSONDecodeError) as got:
        data._json_loads(line)
    assert type(got.value) is json.JSONDecodeError and str(got.value) == str(want.value)


def test_long_integers_never_reach_orjson(monkeypatch: pytest.MonkeyPatch) -> None:
    """orjson returns a float for an integer outside 64 bits (no error), so such lines must not reach it."""
    calls = []
    real = data.orjson.loads
    monkeypatch.setattr(data.orjson, "loads", lambda line: calls.append(line) or real(line))
    assert data._json_loads('{"n": 18446744073709551616}') == {"n": 18446744073709551616} and not calls
    assert data._json_loads('{"n": [248319, -100]}') == {"n": [248319, -100]} and len(calls) == 1


@pytest.mark.parametrize("seed", range(20))
def test_random_numbers_equal_json(seed: int) -> None:
    rng = random.Random(seed)
    nums = [
        rng.choice(
            [
                rng.randrange(-(2**70), 2**70),
                rng.randrange(-(10**6), 10**6),
                rng.random() * 10 ** rng.randrange(-40, 40),
                -rng.random(),
                float(rng.randrange(10**18)),
            ]
        )
        for _ in range(2000)
    ]
    line = json.dumps({"nums": nums})
    assert same(data._json_loads(line), json.loads(line))


@pytest.mark.parametrize("seed", range(50))
def test_long_digit_run_is_the_regex(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        line = "".join(rng.choice("0123456789" * 6 + ', []"ab٣中.-e') for _ in range(rng.randrange(80)))
        assert data._has_long_digit_run(line) == bool(re.search(r"[0-9]{19}", line)), line
    assert data._has_long_digit_run("x" + "9" * 19) and not data._has_long_digit_run("9" * 18 + "," + "9" * 18)
    assert not data._has_long_digit_run("٣" * 30)  # non-ASCII digits are not JSON number digits


def _old_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """miles before this change: json.loads."""
    monkeypatch.setattr(data, "_json_loads", json.loads)


def test_read_file_skips_the_same_bad_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(['{"a": 1}', "", "   ", '{"a": broken', '{"x": NaN}', '{"a": 2}']) + "\n")
    new = list(data.read_file(str(path)))
    new_out = capsys.readouterr().out
    _old_loader(monkeypatch)
    old = list(data.read_file(str(path)))
    assert same(new, old) and len(new) == 3 and new_out == capsys.readouterr().out
    assert "JSON decode error at line 3" in new_out


def test_read_file_keeps_the_row_slice(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(10)))
    assert [row["i"] for row in data.read_file(f"{path}@[2:5]")] == [2, 3, 4]


def _dataset(path: Path) -> data.Dataset:
    """The Dataset a pre-tokenized SFT run builds (input_key input_ids, metadata_key meta, no template / processor)."""
    return data.Dataset(
        str(path),
        tokenizer=None,
        processor=None,
        max_length=None,
        prompt_key="input_ids",
        multimodal_keys=None,
        label_key=None,
        metadata_key="meta",
        tool_key="tools",
        seed=42,
    )


def _assert_same_datasets(path: Path, monkeypatch: pytest.MonkeyPatch) -> data.Dataset:
    new = _dataset(path)
    with monkeypatch.context() as patch:
        _old_loader(patch)
        old = _dataset(path)
    assert len(new) == len(old) > 0
    for a, b in zip(new.origin_samples, old.origin_samples, strict=True):
        assert same(vars(a), vars(b)), (a.metadata.get("record_id"), b.metadata.get("record_id"))
    return new


def test_dataset_same_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(7)
    rows = []
    for i in range(300):
        n = rng.choice([1, 5, 300, 5000])
        ids = [rng.randrange(248320) for _ in range(n)]
        meta = {
            "labels": [-100 if rng.random() < 0.3 else t for t in ids],
            "record_id": f"r{i}",
            "images": [{"path": f"/img/{i}.png", "sha256": "ab" * 32, "grid_thw": [1, 32, 48]}],
            "extra": random_json(rng),
        }
        rows.append(json.dumps({"input_ids": ids, "meta": meta}, ensure_ascii=rng.random() < 0.5))
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(rows) + "\n")
    dataset = _assert_same_datasets(path, monkeypatch)
    assert [s.metadata["record_id"] for s in dataset.origin_samples] == [f"r{i}" for i in range(300)]


@pytest.mark.skipif(not os.environ.get("MILES_REAL_JSONL"), reason="set MILES_REAL_JSONL to a real train.jsonl")
def test_dataset_same_samples_on_real_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(os.environ["MILES_REAL_JSONL"])
    size, picked = source.stat().st_size, []
    with open(source, "rb") as stream:
        for i in range(400):
            stream.seek(size * i // 400)
            if i:
                stream.readline()
            if line := stream.readline():
                picked.append(line)
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"".join(dict.fromkeys(picked)))
    dataset = _assert_same_datasets(path, monkeypatch)
    assert len(dataset) == len(dict.fromkeys(picked))
