"""train.py --prefetch-rollout: the next rollouts' data is fetched while the current one trains, and nothing else
the run does or saves changes."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import train as train_driver
from tests.fast.fixtures.driver_fakes import FakeRemoteMethod
from tests.fast.test_train import _install_driver_fakes, _make_args, _startup_info

from miles.ray.rollout.rollout_prefetcher import RolloutPrefetcher, validate_prefetch_rollout_args
from miles.utils.async_utils import with_disposer
from miles.utils.data import RolloutDataPack
from miles.utils.misc import should_save

BATCH = 3
# Events whose position moves with prefetching; every other event keeps its order.
_FETCH_EVENTS = ("generate_start:", "generate_done:", "generate_cancelled:", "actor_train_begin:")


def _install(
    monkeypatch: pytest.MonkeyPatch,
    args: SimpleNamespace,
    events: list[str],
    *,
    cursor: int = 0,
    get_delays: dict[int, int] | None = None,
    get_errors: dict[int, BaseException] | None = None,
    get_gates: dict[int, asyncio.Event] | None = None,
    train_errors: dict[int, BaseException] | None = None,
    num_rollout_per_epoch: int = 100,
) -> SimpleNamespace:
    """The driver fakes, with an executor whose gets hand out consecutive sample indices from a data cursor (like
    miles' RolloutDataSource) and whose save records that cursor; training yields to the loop like a real step."""
    components = _install_driver_fakes(monkeypatch, args, events, num_rollout_per_epoch=num_rollout_per_epoch)
    state = SimpleNamespace(cursor=cursor, saved={}, trained={})

    async def get(rollout_id: int) -> RolloutDataPack:
        events.append(f"generate_start:{rollout_id}")
        try:
            for _ in range((get_delays or {}).get(rollout_id, 1)):
                await asyncio.sleep(0)
            if (gate := (get_gates or {}).get(rollout_id)) is not None:
                await gate.wait()
        except asyncio.CancelledError:
            events.append(f"generate_cancelled:{rollout_id}")
            raise
        if (error := (get_errors or {}).get(rollout_id)) is not None:
            raise error
        pack = RolloutDataPack(sample_indices=list(range(state.cursor, state.cursor + BATCH)))
        state.cursor += BATCH
        events.append(f"generate_done:{rollout_id}")
        return pack

    async def save(rollout_id: int) -> None:
        events.append(f"executor_save:{rollout_id}")
        state.saved[rollout_id] = state.cursor

    async def release(rollout_id: int) -> None:
        events.append(f"release:{rollout_id}")

    executor = components.rollout_executor
    executor.get, executor.save, executor.release = map(FakeRemoteMethod, (get, save, release))

    actor = components.actor_model
    fake_train = actor.train

    async def train(rollout_id: int, rollout_data: RolloutDataPack, external_data: Any = None) -> Any:
        events.append(f"actor_train_begin:{rollout_id}")
        state.trained[rollout_id] = rollout_data.sample_indices
        for _ in range(20):
            await asyncio.sleep(0)
        if (error := (train_errors or {}).get(rollout_id)) is not None:
            raise error
        return await fake_train(rollout_id, rollout_data, external_data)

    actor.train = train
    return state


async def _run(monkeypatch: pytest.MonkeyPatch, depth: int, **kwargs: Any) -> tuple[SimpleNamespace, list[str]]:
    install = {
        k: kwargs.pop(k)
        for k in ("cursor", "get_delays", "get_errors", "get_gates", "train_errors", "num_rollout_per_epoch")
        if k in kwargs
    }
    events: list[str] = []
    args = _make_args(prefetch_rollout=depth, **kwargs)
    state = _install(monkeypatch, args, events, **install)
    await with_disposer(train_driver.train, _startup_info(args))
    return state, events


def _ordered(events: list[str]) -> list[str]:
    return [event for event in events if not event.startswith(_FETCH_EVENTS)]


def _gets_started_before(events: list[str], event: str) -> list[int]:
    head = events[: events.index(event)]
    return [int(e.split(":")[1]) for e in head if e.startswith("generate_start:")]


class TestSameRunForAnyDepth:
    @pytest.mark.parametrize("depth", [1, 2, 3])
    async def test_batches_saves_and_call_order_match_no_prefetch(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        """Every rollout trains the batch it trains without prefetching; every save persists the cursor of exactly
        the trained rollouts; trains, releases, saves and weight updates keep their order."""
        run = dict(num_rollout=11, save_interval=4, eval_interval=3, skip_eval_before_train=True)
        base_state, base_events = await _run(monkeypatch, 0, **run)
        state, events = await _run(monkeypatch, depth, **run)

        assert state.trained == base_state.trained
        assert base_state.saved == {3: 12, 7: 24, 10: 33}
        assert state.saved == base_state.saved
        assert _ordered(events) == _ordered(base_events)
        for rollout_id in state.saved:
            # no get past the saved rollout had started when its data progress was persisted
            assert max(_gets_started_before(events, f"executor_save:{rollout_id}")) == rollout_id

    @pytest.mark.parametrize("depth", [1, 2])
    async def test_the_next_gets_run_while_the_current_rollout_trains(
        self, monkeypatch: pytest.MonkeyPatch, depth: int
    ):
        _, events = await _run(monkeypatch, depth, num_rollout=6)

        for rollout_id in range(6 - depth):
            begin, end = events.index(f"actor_train_begin:{rollout_id}"), events.index(f"actor_train:{rollout_id}")
            assert begin < events.index(f"generate_done:{rollout_id + depth}") < end

    async def test_without_prefetch_each_get_waits_for_the_previous_step(self, monkeypatch: pytest.MonkeyPatch):
        _, events = await _run(monkeypatch, 0, num_rollout=4)

        for rollout_id in range(1, 4):
            assert events.index(f"release:{rollout_id - 1}") < events.index(f"generate_start:{rollout_id}")

    async def test_gets_run_one_at_a_time_in_rollout_order(self, monkeypatch: pytest.MonkeyPatch):
        """A slow get must not let the next one overtake it: the data source hands out batches in call order."""
        state, events = await _run(monkeypatch, 3, num_rollout=6, get_delays={1: 40, 2: 1, 3: 25})

        gets = [e for e in events if e.startswith(("generate_start:", "generate_done:"))]
        assert gets == [f"generate_{what}:{i}" for i in range(6) for what in ("start", "done")]
        assert state.trained == {i: list(range(i * BATCH, (i + 1) * BATCH)) for i in range(6)}

    async def test_save_steps_fetch_nothing_ahead(self, monkeypatch: pytest.MonkeyPatch):
        _, events = await _run(monkeypatch, 2, num_rollout=8, save_interval=2)

        for rollout_id in (1, 3, 5, 7):
            assert max(_gets_started_before(events, f"executor_save:{rollout_id}")) == rollout_id


class TestRunEnd:
    @pytest.mark.parametrize("depth", [1, 3])
    async def test_no_get_past_the_last_rollout(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        state, events = await _run(monkeypatch, depth, num_rollout=4)

        assert [e for e in events if e.startswith("generate_start:")] == [f"generate_start:{i}" for i in range(4)]
        assert state.cursor == 4 * BATCH

    @pytest.mark.parametrize("depth", [1, 3])
    async def test_no_get_past_debug_exit_after_rollout(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        state, events = await _run(
            monkeypatch, depth, num_rollout=20, start_rollout_id=5, debug_exit_after_rollout=3, cursor=5 * BATCH
        )

        assert [e for e in events if e.startswith("generate_start:")] == [f"generate_start:{i}" for i in (5, 6, 7)]
        assert sorted(state.trained) == [5, 6, 7]
        assert state.cursor == 8 * BATCH


class TestFailures:
    @pytest.mark.parametrize("depth", [1, 2])
    async def test_a_failed_get_raises_at_its_own_step(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        """The error of a prefetched get surfaces when the loop reaches that rollout, after everything the loop did
        before it without prefetching (train, release, save of the earlier rollouts)."""

        async def run(prefetch: int) -> tuple[SimpleNamespace, list[str]]:
            events: list[str] = []
            args = _make_args(prefetch_rollout=prefetch, num_rollout=8, save_interval=2)
            state = _install(monkeypatch, args, events, get_errors={3: OSError("data read failed")})
            with pytest.raises(OSError, match="data read failed"):
                await with_disposer(train_driver.train, _startup_info(args))
            return state, events

        base_state, base_events = await run(0)
        state, events = await run(depth)

        assert sorted(state.trained) == sorted(base_state.trained) == [0, 1, 2]
        assert state.saved == base_state.saved == {1: 2 * BATCH}
        assert _ordered(events) == _ordered(base_events)
        assert "generate_start:4" not in events

    @pytest.mark.parametrize("depth", [1, 2])
    async def test_a_failed_train_cancels_the_pending_gets(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        """The train error propagates, and gets started for rollouts that will never train are cancelled before
        the executor is disposed, so none is left running or with an unretrieved error."""
        gate = asyncio.Event()
        events: list[str] = []
        args = _make_args(prefetch_rollout=depth, num_rollout=6)
        _install(
            monkeypatch,
            args,
            events,
            get_gates={1: gate},
            train_errors={0: RuntimeError("train failed")},
        )

        with pytest.raises(RuntimeError, match="train failed"):
            await with_disposer(train_driver.train, _startup_info(args))

        assert "generate_cancelled:1" in events
        assert events.index("generate_cancelled:1") < events.index("executor_dispose")
        assert not [e for e in events if e.startswith("generate_done:") and e != "generate_done:0"]
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


class TestResume:
    @pytest.mark.parametrize("depth", [1, 2])
    async def test_resume_from_a_save_gives_the_same_next_batches(self, monkeypatch: pytest.MonkeyPatch, depth: int):
        """A run that prefetches, stops after a save, and resumes from that save's cursor trains the same batches as
        one uninterrupted run without prefetching."""
        reference, _ = await _run(monkeypatch, 0, num_rollout=10, save_interval=3)

        first, _ = await _run(monkeypatch, depth, num_rollout=10, save_interval=3, debug_exit_after_rollout=7)
        assert first.saved == {2: 3 * BATCH, 5: 6 * BATCH}
        resumed, _ = await _run(
            monkeypatch, depth, num_rollout=10, save_interval=3, start_rollout_id=6, cursor=first.saved[5]
        )

        assert {**{i: first.trained[i] for i in range(6)}, **resumed.trained} == reference.trained


class TestScheduledSaves:
    """--save-rollout-ids: a save at a rollout fixed before the run (qwen38: one early save at step 3, rollout 2) is a
    save boundary like the periodic ones, so it can be combined with prefetching (a sentinel cannot)."""

    # 4 rollouts per epoch, a save at each epoch end (3, 7, 11) and the last rollout (13), plus the early save at 2
    RUN = dict(num_rollout=14, save_interval=4, save_rollout_ids=[2], num_rollout_per_epoch=4)

    @staticmethod
    def _rows_per_rollout(state: SimpleNamespace) -> dict[int, list[int]]:
        return {rollout_id: list(rows) for rollout_id, rows in sorted(state.trained.items())}

    @pytest.mark.parametrize("depth", [0, 1, 2])
    async def test_the_listed_rollout_saves_and_nothing_past_it_is_fetched_first(
        self, monkeypatch: pytest.MonkeyPatch, depth: int
    ):
        state, events = await _run(monkeypatch, depth, **self.RUN)

        assert state.saved == {2: 3 * BATCH, 3: 4 * BATCH, 7: 8 * BATCH, 11: 12 * BATCH, 13: 14 * BATCH}
        for rollout_id in state.saved:
            assert max(_gets_started_before(events, f"executor_save:{rollout_id}")) == rollout_id
        assert state.trained == {i: list(range(i * BATCH, (i + 1) * BATCH)) for i in range(14)}

    @pytest.mark.parametrize("depth", [1, 2])
    @pytest.mark.parametrize("resume_from", [2, 7], ids=["early-save", "epoch-save"])
    async def test_resume_from_the_save_consumes_the_rows_of_an_uninterrupted_run(
        self, monkeypatch: pytest.MonkeyPatch, depth: int, resume_from: int
    ):
        """Stop a prefetching run two rollouts after the save (so it had fetched past the save before it stopped),
        resume from the save's data cursor: per rollout, the rows trained equal those of one uninterrupted run
        without prefetching."""
        reference, _ = await _run(monkeypatch, 0, **self.RUN)
        first, events = await _run(monkeypatch, depth, debug_exit_after_rollout=resume_from + 3, **self.RUN)
        assert f"generate_start:{resume_from + 1}" in events and resume_from in first.saved
        resumed, _ = await _run(
            monkeypatch, depth, start_rollout_id=resume_from + 1, cursor=first.saved[resume_from], **self.RUN
        )

        rows = {**{i: first.trained[i] for i in range(resume_from + 1)}, **resumed.trained}
        assert self._rows_per_rollout(SimpleNamespace(trained=rows)) == self._rows_per_rollout(reference)
        assert sorted(rows) == list(range(14))

    @pytest.mark.parametrize("depth", [1, 2])
    async def test_an_unscheduled_save_after_a_prefetch_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, depth: int
    ):
        """What the validation keeps out (--save-trigger-sentinel with prefetching): a save the prefetcher did not
        know of would persist a cursor past the trained rollouts; the driver refuses it rather than save it."""
        sentinel = tmp_path / "save-now"
        sentinel.write_text("")
        events: list[str] = []
        args = _make_args(
            prefetch_rollout=depth, num_rollout=6, save=str(tmp_path), save_trigger_sentinel=str(sentinel)
        )
        state = _install(monkeypatch, args, events)

        with pytest.raises(AssertionError, match="saved after a later rollout's get"):
            await with_disposer(train_driver.train, _startup_info(args))
        assert state.saved == {}

    def test_the_prefetcher_knows_the_listed_saves(self):
        args = _make_args(prefetch_rollout=2, num_rollout=20, save_interval=10, save_rollout_ids=[2, 5])
        prefetcher = RolloutPrefetcher.from_args(args, SimpleNamespace(get=None), num_rollout_per_epoch=None)
        assert [i for i in range(20) if prefetcher._save_due(i)] == [2, 5, 9, 19]


class TestShouldSave:
    @pytest.mark.parametrize(
        "save_interval, save_rollout_ids, per_epoch, expected",
        [
            (None, None, None, []),
            (None, [2], None, [2]),
            (4, None, None, [3, 7, 9]),
            (4, [2, 3], None, [2, 3, 7, 9]),
            (100, [0], 5, [0, 4, 9]),
        ],
    )
    def test_periodic_saves_and_listed_ids(self, save_interval, save_rollout_ids, per_epoch, expected):
        args = SimpleNamespace(save_interval=save_interval, save_rollout_ids=save_rollout_ids, num_rollout=10)
        assert [i for i in range(10) if should_save(args, i, per_epoch)] == expected


class TestPrefetcherUnit:
    async def test_fetched_beyond_tracks_the_furthest_started_get(self):
        fetched: list[int] = []

        async def fetch(rollout_id: int) -> RolloutDataPack:
            fetched.append(rollout_id)
            return RolloutDataPack(sample_indices=[rollout_id])

        prefetcher = RolloutPrefetcher(fetch=fetch, depth=2, last_rollout_id=7, save_due=lambda i: i == 4)
        assert not prefetcher.fetched_beyond(0)
        assert (await prefetcher.get(0)).sample_indices == [0]
        assert prefetcher.fetched_beyond(1)
        assert not prefetcher.fetched_beyond(2)
        for rollout_id in (1, 2, 3):
            assert (await prefetcher.get(rollout_id)).sample_indices == [rollout_id]
        assert (await prefetcher.get(4)).sample_indices == [4]
        assert not prefetcher.fetched_beyond(4)  # rollout 4 saves: nothing past it is fetched before its save
        assert (await prefetcher.get(5)).sample_indices == [5]
        assert prefetcher.fetched_beyond(6)
        for rollout_id in (6, 7):
            assert (await prefetcher.get(rollout_id)).sample_indices == [rollout_id]
        assert fetched == list(range(8))  # each once, in order, none past the last rollout
        await prefetcher.dispose()

    async def test_depth_zero_calls_through(self):
        calls: list[int] = []

        async def fetch(rollout_id: int) -> RolloutDataPack:
            calls.append(rollout_id)
            return RolloutDataPack()

        prefetcher = RolloutPrefetcher(fetch=fetch, depth=0, last_rollout_id=9, save_due=lambda i: False)
        await prefetcher.get(3)
        assert calls == [3]
        assert not prefetcher.fetched_beyond(0)


class TestValidation:
    def _args(self, **overrides: Any) -> SimpleNamespace:
        return SimpleNamespace(
            **{"prefetch_rollout": 1, "debug_train_only": True, "save_trigger_sentinel": None, **overrides}
        )

    def test_accepts_a_training_only_run(self):
        validate_prefetch_rollout_args(self._args())
        validate_prefetch_rollout_args(
            self._args(prefetch_rollout=0, debug_train_only=False, save_trigger_sentinel="x")
        )

    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(prefetch_rollout=-1), ">= 0"),
            (dict(debug_train_only=False), "--debug-train-only"),
            (dict(save_trigger_sentinel="/tmp/save-now"), "--save-trigger-sentinel"),
        ],
    )
    def test_rejects(self, overrides: dict[str, Any], message: str):
        with pytest.raises(AssertionError, match=message):
            validate_prefetch_rollout_args(self._args(**overrides))
