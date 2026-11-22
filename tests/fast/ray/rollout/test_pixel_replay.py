import gc
import pickle
import weakref

import pytest
import torch
from tests.fast.ray.rollout.test_output_snapshot_copies import _checkpoint_bytes, _executor

from miles.ray.rollout import rollout_executor as executor_module
from miles.utils.data import remove_rollout_data_refs
from miles.utils.object_store import RayObjectStore, _MooncakeStoreObjectRef


async def test_publish_failure_retry_release_and_checkpoint_replay(monkeypatch, tmp_path):
    # Exercise the executor's actual get/save/load/release path. Only the transport is in-memory.

    class Store(RayObjectStore):
        def put(self, *, value, value_spec):
            if self.fail_at == len(self.live):
                self.fail_at = None
                raise RuntimeError("injected put failure")
            copied = pickle.loads(pickle.dumps(value, protocol=5))
            ref = _MooncakeStoreObjectRef(payload=copied)
            self.live.append(weakref.ref(ref))
            return ref

        def remove(self, ref):
            self.removed.append(ref.payload["tokens"][0][0])

    def convert(args, data, **kwargs):
        return {"sample_indices": [s.index for s in data]}

    def split(args, batch, config):
        return [
            {
                "sample_indices": batch["sample_indices"],
                "tokens": [[i]],
                "multimodal_train_inputs": [{"pixel_values": torch.full((3, 4), float(i))}],
            }
            for i in range(64)
        ]

    monkeypatch.setattr(executor_module, "convert_samples_to_train_data", convert)
    monkeypatch.setattr(executor_module, "split_train_data_by_dp", split)
    arms = []
    for compact in (False, True):
        root = tmp_path / str(compact)
        store = Store(frees_objects=False)
        store.live, store.fail_at, store.removed = [], 7, []
        monkeypatch.setattr(executor_module.object_store, "get_instance", lambda: store)

        def fresh():
            ex = _executor(root)
            ex.args.compact_ray_train_pixels = compact
            return ex

        ex = fresh()
        with pytest.raises(RuntimeError, match="injected put failure"):
            await ex.get(1)
        gc.collect()
        assert len(store.live) == 7 and all(ref() is None for ref in store.live)
        # Save the captured, untrained rollout and replay it on a new executor.
        await ex.save(0)
        checkpoint = _checkpoint_bytes(root)
        assert checkpoint
        replay = fresh()
        await replay.load(0, load=str(root))

        async def refuse_generation(**kwargs):
            pytest.fail("retry must use the saved rollout")

        replay._generate_rollout_data = refuse_generation
        first = await replay.get(1)
        assert len(first.data_ref) == 64 and first.sample_indices == [0, 1]
        assert all(ref() is not None for ref in store.live[7:])
        remove_rollout_data_refs(replay.args, first)
        assert store.removed == list(range(64))
        await replay.release(1)
        await replay.release(1)
        assert replay._unreleased_store_refs == {}
        result = [
            (r.payload["tokens"], r.payload["multimodal_train_inputs"][0]["pixel_values"].numpy().tobytes())
            for r in first.data_ref
        ]
        assert [t for t, _ in result] == [[[i]] for i in range(64)]
        arms.append((result, checkpoint))
    assert arms[0] == arms[1]
