import argparse

from miles.utils.args.schema import A, Arg, BaseConfig


class OrchestratorOnlyConfig(BaseConfig):
    load: A[str | None, Arg(reset=True)] = None
    requested_load: str | None
    clear_memory_every_step: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Whether train.py clears the trainer's CUDA cache (synchronize + gc.collect + empty_cache on "
                "every rank) after every rollout. The clear hands the memory to colocated rollout engines; a "
                "pure training run (--debug-train-only, no offload, no critic, no colocate) has nobody to hand "
                "it to, so --no-clear-memory-every-step keeps the cache between steps and clears only around a "
                "checkpoint save and before an eval. Only the synchronous train.py driver reads it."
            ),
        ),
    ] = True
