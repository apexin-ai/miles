"""RolloutDataSource.get_samples copies each dataset row with copy_dataset_sample instead of copy.deepcopy: the same
samples, nothing shared with the dataset, and the same batches over epochs."""

import copy
import json
import math
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from miles.rollout import data_source as data_source_module
from miles.rollout.data_source import RolloutDataSource, copy_dataset_sample
from miles.utils.types import Sample, SampleLineage


def same(a, b) -> bool:
    """Equal values of equal types all the way down (NaN equals NaN; 1 is not True)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Sample):
        return same(vars(a), vars(b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    return a == b


def containers(value, out: list | None = None) -> list:
    """Every mutable container reachable from value, in traversal order (repeats included)."""
    out = [] if out is None else out
    if isinstance(value, Sample):
        containers(vars(value), out)
    elif isinstance(value, (dict, list, np.ndarray, set)):
        out.append(value)
        if isinstance(value, dict):
            for item in value.values():
                containers(item, out)
        elif isinstance(value, list):
            for item in value:
                containers(item, out)
    elif isinstance(value, tuple):
        for item in value:
            containers(item, out)
    return out


def alias_pattern(value) -> list[int]:
    """Which reachable containers are the same object, as first-seen positions."""
    first: dict[int, int] = {}
    return [first.setdefault(id(item), index) for index, item in enumerate(containers(value))]


def random_json(rng: random.Random, depth: int = 0):
    kind = rng.randrange(9 if depth < 4 else 6)
    if kind == 0:
        return rng.randrange(-(2**70), 2**70)
    if kind == 1:
        return rng.choice([rng.random() * 10 ** rng.randrange(-5, 6), float("nan"), -0.0])
    if kind == 2:
        return "".join(rng.choice('abé中\U0001f600"\\\n') for _ in range(rng.randrange(6)))
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return [rng.randrange(248320) for _ in range(rng.randrange(50))]
    if kind == 5:
        return [-100] * rng.randrange(5) + [rng.randrange(10), True, None, "s", 1.5]
    if kind in (6, 7):
        return {f"k{i}": random_json(rng, depth + 1) for i in range(rng.randrange(5))}
    return [random_json(rng, depth + 1) for _ in range(rng.randrange(5))]


def random_sample(rng: random.Random) -> Sample:
    ids = [rng.randrange(248320) for _ in range(rng.randrange(1, 300))]
    sample = Sample(
        index=rng.randrange(100),
        prompt=ids if rng.random() < 0.7 else [{"role": "user", "content": random_json(rng)}],
        label=rng.choice([None, "answer"]),
        metadata={
            "labels": [-100 if rng.random() < 0.3 else t for t in ids],
            "record_id": f"r{rng.randrange(1000)}",
            "images": [{"path": f"/img/{i}.png", "sha256": "ab" * 32} for i in range(rng.randrange(3))],
            "extra": random_json(rng),
        },
    )
    if rng.random() < 0.3:
        sample.tokens = sample.prompt  # aliased fields: the copy keeps them one list, as deepcopy does
    if rng.random() < 0.3:
        sample.metadata["again"] = sample.metadata["labels"]
    if rng.random() < 0.3:
        sample.metadata["tuple"] = (1, [2, 3], {"x": [4]})
        sample.metadata[(1, 2)] = "tuple key"
    if rng.random() < 0.3:
        sample.rollout_routed_experts = np.arange(12, dtype=np.int32).reshape(2, 3, 2)
        sample.lineage = SampleLineage(source_sample_index=1, output_index=0, output_count=2)
        sample.multimodal_inputs = {"images": [np.zeros(3)]}
    return sample


class TestCopyDatasetSample:
    @pytest.mark.parametrize("seed", range(200))
    def test_equals_deepcopy(self, seed: int) -> None:
        source = random_sample(random.Random(seed))
        before = pickle.dumps(source)

        got, want = copy_dataset_sample(source), copy.deepcopy(source)

        assert same(got, want) and same(got, source)
        assert alias_pattern(got) == alias_pattern(want) == alias_pattern(source)
        assert not {id(item) for item in containers(got)} & {id(item) for item in containers(source)}
        assert pickle.dumps(source) == before

    def test_mutating_the_copy_leaves_the_row_alone(self) -> None:
        source = random_sample(random.Random(1))
        source.prompt = [1, 2, 3]
        source.metadata["images"] = [{"path": "/img/0.png", "sha256": "ab" * 32}]
        before = pickle.dumps(source)

        copied = copy_dataset_sample(source)
        copied.prompt.append(1)
        copied.tokens.append(2)
        copied.metadata["labels"][0] = 7
        copied.metadata["images"][0]["path"] = "/elsewhere.png"
        copied.metadata["new"] = 1
        copied.reward = 0

        assert pickle.dumps(source) == before

    def test_a_subclass_goes_to_deepcopy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Custom(Sample):
            pass

        calls = []
        real = copy.deepcopy
        monkeypatch.setattr(data_source_module.copy, "deepcopy", lambda value, *a: calls.append(value) or real(value))
        source = Custom(prompt=[1, 2])

        assert same(vars(copy_dataset_sample(source)), vars(source)) and calls == [source]

    def test_token_lists_skip_deepcopy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the change: no list or dict of an all-JSON row reaches deepcopy (only the small non-JSON
        fields: the status enum, the spec / prefix cache info dataclasses)."""
        real = copy.deepcopy
        handed = []
        monkeypatch.setattr(
            data_source_module.copy, "deepcopy", lambda value, *a: handed.append(type(value)) or real(value, *a)
        )
        source = Sample(prompt=list(range(1000)), metadata={"labels": [-100] * 1000, "images": [{"path": "p"}]})

        assert same(copy_dataset_sample(source), source)
        assert Sample.Status in handed and not {list, dict} & set(handed)


def _write_rows(path: Path, n: int, seed: int = 3) -> None:
    rng = random.Random(seed)
    with path.open("w") as f:
        for i in range(n):
            ids = [rng.randrange(248320) for _ in range(rng.randrange(2, 40))]
            meta = {"labels": [-100] + ids[1:], "record_id": f"r{i}", "images": [{"path": f"/i/{i}", "sha256": "0"}]}
            f.write(json.dumps({"input_ids": ids, "meta": meta}) + "\n")


def _source(path: Path, n_samples_per_prompt: int) -> RolloutDataSource:
    args = SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(path),
        rollout_max_prompt_len=None,
        input_key="input_ids",
        label_key=None,
        metadata_key="meta",
        tool_key=None,
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=True,
        n_samples_per_prompt=n_samples_per_prompt,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data_source_module, "load_tokenizer", lambda *a, **k: None)
        patch.setattr(data_source_module, "load_processor", lambda *a, **k: None)
        return RolloutDataSource(args)


def _train_like(sample: Sample) -> None:
    """What a pre-tokenized SFT rollout and the train-data conversion do to a sample, and more."""
    sample.tokens = sample.prompt
    sample.loss_mask = [int(y != -100) for y in sample.metadata["labels"]]
    sample.response_length = len(sample.tokens) - 1
    sample.reward = 0
    sample.tokens.append(0)
    sample.metadata["labels"].append(-100)
    sample.metadata["images"][0]["path"] = "/moved"
    sample.metadata["record_id"] = "trained"


class TestGetSamplesOverEpochs:
    @pytest.mark.parametrize("n_samples_per_prompt", [1, 2])
    def test_batches_equal_deepcopy_and_the_dataset_is_never_mutated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_samples_per_prompt: int
    ) -> None:
        path = tmp_path / "rows.jsonl"
        _write_rows(path, 7)
        batches = {}
        for fast in (True, False):
            with monkeypatch.context() as patch:
                if not fast:
                    patch.setattr(data_source_module, "copy_dataset_sample", copy.deepcopy)
                source = _source(path, n_samples_per_prompt)
                dataset_before = pickle.dumps(source.dataset.origin_samples)
                got = []
                for _ in range(5):  # 15 rows: two epochs and a batch across each boundary
                    batch = source.get_samples(3)
                    got.append(pickle.dumps(batch))
                    for group in batch:
                        for sample in group:
                            _train_like(sample)
                assert source.epoch_id == 2
                assert pickle.dumps(source.dataset.origin_samples) == dataset_before
                batches[fast] = got

        assert batches[True] == batches[False]
