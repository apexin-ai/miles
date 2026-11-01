"""train.py skips the executor's per-rollout deepcopy of each generated batch (skip_output_snapshot_copies): its loop
saves only after every rollout it fetched has trained, so each snapshot was discarded unread. Nothing it saves or
resumes from changes, and a save that would still hold an uncopied batch refuses instead of losing it."""

from pathlib import Path

import pytest
import train as train_driver
from tests.fast.ray.rollout.test_rollout_executor import _make_executor
from tests.fast.test_train import _install_driver_fakes, _make_args, _startup_info

from miles.ray.rollout import output_snapshotter as output_snapshotter_module
from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import compute_rollout_checkpoint_dir
from miles.utils.async_utils import with_disposer
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.types import Sample

SAVE_STEPS = (1, 3)
NUM_ROLLOUT = 6


def _snapshotter(*, skip: bool) -> _RolloutExecutorOutputSnapshotter:
    snapshotter = _RolloutExecutorOutputSnapshotter(
        args=_make_args(ci_test=False, ci_inject_missing_prefetched_batch_bug=False, use_output_store=False)
    )
    if skip:
        snapshotter.skip_capture_copies()
    return snapshotter


class _Rows:
    """Each rollout hands out the next two rows, with token lists and metadata like a pre-tokenized SFT row."""

    def __init__(self, next_index: int = 0) -> None:
        self.next_index = next_index

    async def __call__(self, *, rollout_id: int, trainer_model_id: str | None) -> tuple[list[Sample], dict]:
        data = []
        for _ in range(2):
            index = self.next_index
            self.next_index += 1
            tokens = [index, index + 1, index + 2]
            data.append(
                Sample(
                    index=index,
                    group_index=index,
                    prompt=tokens,
                    tokens=tokens,
                    response_length=2,
                    loss_mask=[1, 1],
                    reward=0,
                    metadata={"labels": [-100, index + 1, index + 2], "record_id": f"r{index}"},
                    status=Sample.Status.COMPLETED,
                )
            )
        return data, {"rollout_id": rollout_id}


class _InlineStore:
    def put(self, *, value, value_spec):
        return _MooncakeStoreObjectRef(payload=value)


def _trained(pack) -> list:
    return [ref.payload for ref in pack.data_ref]


def _executor(root: Path, next_index: int = 0):
    executor = _make_executor(root, None)
    executor._generate_rollout_data = _Rows(next_index)
    return executor


@pytest.fixture(autouse=True)
def _stub_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rollout_executor_module,
        "convert_samples_to_train_data",
        lambda _args, data, **kwargs: {
            "sample_indices": [sample.index for sample in data],
            "tokens": [list(sample.tokens) for sample in data],
        },
    )
    monkeypatch.setattr(
        rollout_executor_module, "split_train_data_by_dp", lambda _args, train_data, _config: [train_data]
    )
    monkeypatch.setattr(rollout_executor_module.object_store, "get_instance", _InlineStore)


def _checkpoint_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


async def _sync_run(root: Path, *, skip: bool, start: int = 0, load_from: int | None = None) -> list:
    """What train.py does with the executor: get, train, release, and save on the save steps."""
    executor = _executor(root, next_index=2 * start)
    if load_from is not None:
        await executor.load(load_from, load=str(root))
    if skip:
        await executor.skip_output_snapshot_copies()
    trained = []
    for rollout_id in range(start, NUM_ROLLOUT):
        pack = await executor.get(rollout_id)
        trained.append(_trained(pack))
        await executor.release(rollout_id)
        if rollout_id in SAVE_STEPS:
            await executor.save(rollout_id)
    return trained


class TestSkipCaptureCopies:
    def test_a_skipped_capture_makes_no_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        snapshotter = _snapshotter(skip=True)

        def no_deepcopy(*_args, **_kwargs):
            raise AssertionError("deepcopy called")

        monkeypatch.setattr(output_snapshotter_module.copy, "deepcopy", no_deepcopy)
        snapshotter.capture(trainer_model_id=None, rollout_id=0, data=[Sample(index=1)], metadata={"m": [1]})

    def test_the_default_still_copies(self) -> None:
        snapshotter = _snapshotter(skip=False)
        data = [Sample(index=1, tokens=[1, 2])]
        snapshotter.capture(trainer_model_id=None, rollout_id=0, data=data, metadata={})
        data[0].tokens.append(3)

        entry = next(iter(snapshotter._snapshots.values()))
        assert entry.data[0].tokens == [1, 2]

    def test_the_duplicate_and_replay_guards_still_hold(self) -> None:
        snapshotter = _snapshotter(skip=True)
        snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[], metadata={})

        with pytest.raises(AssertionError, match="captured before"):
            snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[], metadata={})
        with pytest.raises(AssertionError, match="must not be replayed"):
            snapshotter.get(trainer_model_id=None, rollout_id=2)

    def test_a_save_that_still_holds_an_uncopied_batch_refuses(self, tmp_path: Path) -> None:
        """A driver that fetched past the save (train_async.py, a prefetch beyond a save step) must not lose it."""
        snapshotter = _snapshotter(skip=True)
        snapshotter.capture(trainer_model_id=None, rollout_id=4, data=[Sample(index=9)], metadata={})
        snapshotter.discard_completed(completed_rollout_ids={None: 3})

        with pytest.raises(AssertionError, match=r"rollouts \[.*rollout_id=4.*\] were captured without a copy"):
            snapshotter.save(tmp_path)
        assert not (tmp_path / "state.pt").exists()

    def test_a_trained_uncopied_batch_saves_like_a_copied_one(self, tmp_path: Path) -> None:
        saved = {}
        for skip in (False, True):
            snapshotter = _snapshotter(skip=skip)
            for rollout_id in range(5):
                snapshotter.capture(trainer_model_id=None, rollout_id=rollout_id, data=[Sample()], metadata={})
            snapshotter.discard_completed(completed_rollout_ids={None: 4})
            snapshotter.save(tmp_path / str(skip))
            saved[skip] = (tmp_path / str(skip) / "state.pt").read_bytes()

        assert saved[True] == saved[False]


class TestSyncExecutorRun:
    async def test_saved_state_and_batches_are_byte_identical(self, tmp_path: Path) -> None:
        runs = {}
        for skip in (False, True):
            root = tmp_path / ("skip" if skip else "copy")
            trained = await _sync_run(root, skip=skip)
            runs[skip] = (trained, _checkpoint_bytes(root))

        assert runs[True][0] == runs[False][0]
        for rollout_id in SAVE_STEPS:
            directory = compute_rollout_checkpoint_dir(Path("."), rollout_id=rollout_id)
            assert {f"{directory}/data_source/state.pt", f"{directory}/executor/state.pt"} <= set(runs[True][1])
        assert runs[True][1] == runs[False][1]

    async def test_a_resumed_run_trains_the_same_batches(self, tmp_path: Path) -> None:
        resumed = {}
        for skip in (False, True):
            root = tmp_path / ("skip" if skip else "copy")
            await _sync_run(root, skip=False)
            resumed[skip] = await _sync_run(root, skip=skip, start=SAVE_STEPS[0] + 1, load_from=SAVE_STEPS[0])

        assert resumed[True] == resumed[False]
        assert [shards[0]["sample_indices"] for shards in resumed[True]] == [[4, 5], [6, 7], [8, 9], [10, 11]]

    async def test_a_restored_untrained_batch_is_still_replayed(self, tmp_path: Path) -> None:
        """Skipping applies to captures only: a batch an async run saved untrained still replays after a resume."""
        executor = _executor(tmp_path)
        await executor.get(0)
        await executor.get(1)
        await executor.save(0)

        resumed = _executor(tmp_path, next_index=100)
        await resumed.load(0, load=str(tmp_path))
        await resumed.skip_output_snapshot_copies()

        assert _trained(await resumed.get(1))[0]["sample_indices"] == [2, 3]


class TestTrainDriver:
    async def test_train_py_skips_the_copies_before_its_first_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []
        args = _make_args(num_rollout=3, save_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        skipped_at = components.rollout_executor.snapshot_copies_skipped_at
        assert skipped_at is not None
        assert "generate_start:0" in events[skipped_at:]
        assert not [event for event in events[:skipped_at] if event.startswith("generate_start")]
