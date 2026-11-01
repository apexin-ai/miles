"""Fetch the next rollouts' training data while the current one trains (train.py ``--prefetch-rollout``)."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from miles.utils.misc import should_run_periodic_action

if TYPE_CHECKING:
    from miles.utils.data import RolloutDataPack

logger = logging.getLogger(__name__)


def validate_prefetch_rollout_args(args) -> None:
    assert args.prefetch_rollout >= 0, f"--prefetch-rollout must be >= 0, got {args.prefetch_rollout}"
    if args.prefetch_rollout == 0:
        return
    # Without engines the data of a rollout does not depend on the weights, so fetching it before the previous
    # rollout has trained (and published its weights) changes nothing but when it is fetched.
    assert args.debug_train_only, "--prefetch-rollout needs --debug-train-only: generated rollouts follow the weights"
    # A sentinel can appear while a later rollout is already fetched; that save would persist a cursor past the
    # trained rollouts. Periodic saves are known in advance, so the prefetcher never fetches across them.
    assert args.save_trigger_sentinel is None, "--prefetch-rollout cannot be combined with --save-trigger-sentinel"


class RolloutPrefetcher:
    """Starts ``rollout_executor.get`` of the next ``depth`` rollouts while the current one trains.

    Each get starts only after the previous one has returned, so the data source hands out its batches in rollout
    order, exactly as without prefetching. No get starts before the checkpoint save of an earlier rollout: a save
    persists the data cursor and the executor's untrained batches, which then hold exactly the trained rollouts,
    as without prefetching, so a resume or a plan migration that reads the cursor sees the same state. Nor does a
    get start past the last rollout the loop trains (``--num-rollout``, ``--debug-exit-after-rollout``).

    With ``depth`` 0 every get runs when the loop asks for it, as before.
    """

    def __init__(
        self,
        *,
        fetch: Callable[[int], Awaitable["RolloutDataPack"]],
        depth: int,
        last_rollout_id: int,
        save_due: Callable[[int], bool],
    ) -> None:
        self._fetch = fetch
        self._depth = depth
        self._last_rollout_id = last_rollout_id
        self._save_due = save_due
        self._pending: dict[int, asyncio.Task] = {}
        self._tail: asyncio.Task | None = None
        self._started_through: int | None = None

    @classmethod
    def from_args(cls, args, rollout_executor, num_rollout_per_epoch: int | None) -> "RolloutPrefetcher":
        last_rollout_id = args.num_rollout - 1
        if args.debug_exit_after_rollout is not None:
            last_rollout_id = min(last_rollout_id, args.start_rollout_id + args.debug_exit_after_rollout - 1)
        return cls(
            fetch=rollout_executor.get,
            depth=args.prefetch_rollout,
            last_rollout_id=last_rollout_id,
            save_due=lambda rollout_id: should_run_periodic_action(
                rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
            ),
        )

    async def get(self, rollout_id: int) -> "RolloutDataPack":
        if self._depth == 0:
            return await self._fetch(rollout_id)

        start = time.monotonic()
        prefetched = (task := self._pending.pop(rollout_id, None)) is not None
        if not prefetched:
            assert not self._pending, f"rollout {rollout_id} requested while {sorted(self._pending)} are prefetched"
            task = self._start(rollout_id)
        # The next gets are queued before this one is awaited: each runs as soon as its predecessor returns.
        for next_id in range(rollout_id + 1, min(rollout_id + self._depth, self._last_rollout_id) + 1):
            if self._save_due(next_id - 1):
                break
            if next_id not in self._pending:
                self._pending[next_id] = self._start(next_id)
        pack, fetch_seconds = await task
        if prefetched:
            logger.info(
                f"rollout {rollout_id}: data prefetched (fetch {fetch_seconds:.2f} s, waited "
                f"{time.monotonic() - start:.2f} s)"
            )
        return pack

    def fetched_beyond(self, rollout_id: int) -> bool:
        """Whether the get of a rollout after ``rollout_id`` has started (and may have moved the data cursor)."""
        return self._started_through is not None and self._started_through > rollout_id

    def _start(self, rollout_id: int) -> asyncio.Task:
        assert self._started_through is None or rollout_id == self._started_through + 1, (
            rollout_id,
            self._started_through,
        )
        task = asyncio.create_task(self._fetch_after(self._tail, rollout_id))
        self._tail = task
        self._started_through = rollout_id
        return task

    async def _fetch_after(self, previous: asyncio.Task | None, rollout_id: int) -> tuple["RolloutDataPack", float]:
        if previous is not None:
            await asyncio.wait([previous])
            if previous.cancelled() or previous.exception() is not None:
                # The loop raises that error when it awaits the previous rollout; this one is never trained.
                raise RuntimeError(f"rollout {rollout_id} not fetched: the fetch of the rollout before it failed")
        start = time.monotonic()
        pack = await self._fetch(rollout_id)
        return pack, time.monotonic() - start

    async def dispose(self) -> None:
        """Cancel the gets of rollouts the loop will not train (it raised); the executor is disposed after this."""
        pending, self._pending = self._pending, {}
        for task in pending.values():
            task.cancel()
        await asyncio.gather(*pending.values(), return_exceptions=True)
        if pending:
            logger.warning(f"Dropped the prefetched rollouts {sorted(pending)}: the loop ended before training them")
