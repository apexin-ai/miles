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
    driver_log_memory_interval: A[
        int,
        Arg(
            help=(
                "With the cache kept between steps (--no-clear-memory-every-step, no save or eval due): train.py "
                "asks every trainer rank for its step-end memory line (one RPC to every rank, ~0.2 s at 256 ranks) "
                "only after a rollout whose rollout_id + 1 is a multiple of N; 0 never. Default 1: after every "
                "rollout, as before. The clears around a save or an eval print their own lines either way."
            ),
        ),
    ] = 1
    skip_train_only_weight_update: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Pure training runs only (--debug-train-only, no inference engine): train.py skips the weight update "
                "after each trained rollout, which there reaches no engine (every trainer rank returns no weight "
                "version) but costs a round trip to every rank. The ORCHESTRATOR_STEP_END fault hook still runs; the "
                "update before the first rollout stays. Default off."
            ),
        ),
    ] = False
