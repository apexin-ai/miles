"""Measure an unmodified Ray put on CPython 3.12 using function-local monitoring."""

import argparse
import ast
import inspect
import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import ray
import torch
from ray._private.serialization import SerializationContext
from ray._private.worker import Worker

from miles.utils.object_store import RayObjectStore


class PutProbe:
    def __init__(self):
        self.thread = threading.get_ident()
        self.rows = []
        self.edges = {}
        self.tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
        lines, start = inspect.getsourcelines(Worker.put_object)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        (assignment,) = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "self.core_worker.put_object"
        ]
        self.write_start = start + assignment.lineno - 1
        self.write_end = start + assignment.end_lineno - 1

    def __enter__(self):
        mon = sys.monitoring
        ev = mon.events
        mon.use_tool_id(self.tool, "miles-ray-put-benchmark")
        for event, callback in [(ev.LINE, self.line), (ev.PY_START, self.start), (ev.PY_RETURN, self.end)]:
            mon.register_callback(self.tool, event, callback)
        mon.set_local_events(self.tool, Worker.put_object.__code__, ev.LINE)
        mon.set_local_events(self.tool, SerializationContext.serialize.__code__, ev.PY_START | ev.PY_RETURN)
        return self

    def __exit__(self, *_):
        mon = sys.monitoring
        for fn in (Worker.put_object, SerializationContext.serialize):
            mon.set_local_events(self.tool, fn.__code__, 0)
        for event in (mon.events.LINE, mon.events.PY_START, mon.events.PY_RETURN):
            mon.register_callback(self.tool, event, None)
        mon.free_tool_id(self.tool)

    def start(self, code, offset):
        if threading.get_ident() != self.thread:
            return
        self.edges["serialize_start"] = time.perf_counter_ns()

    def end(self, code, offset, value):
        if threading.get_ident() != self.thread:
            return
        self.edges["serialize_end"] = time.perf_counter_ns()
        self.edges["size"] = value.total_bytes

    def line(self, code, line):
        if threading.get_ident() != self.thread:
            return
        if line == self.write_start and "write_start" not in self.edges:
            self.edges["write_start"] = time.perf_counter_ns()
        if line > self.write_end and "write_end" not in self.edges:
            self.edges["write_end"] = time.perf_counter_ns()

    def put(self, store, value, spec):
        self.edges = {}
        start = time.perf_counter_ns()
        ref = store.put(value, spec)
        end = time.perf_counter_ns()
        e = self.edges
        assert start <= e["serialize_start"] <= e["serialize_end"] <= e["write_start"] <= e["write_end"] <= end
        self.rows.append(
            dict(
                total=end - start,
                serialize=e["serialize_end"] - e["serialize_start"],
                write=e["write_end"] - e["write_start"],
                returned=end - e["write_end"],
                entry=e["serialize_start"] - start + e["write_start"] - e["serialize_end"],
                size=e["size"],
            )
        )
        return ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure native Ray put phases for trusted, pre-split train shards (CPython 3.12+)."
    )
    parser.add_argument("shards", type=Path, help="torch.save file containing a list of tensor/primitive shard dicts")
    parser.add_argument("--address", required=True, help="address of an existing Ray cluster")
    args = parser.parse_args()
    shards = torch.load(args.shards, weights_only=True, map_location="cpu")
    ray.init(address=args.address)
    store = RayObjectStore(frees_objects=False)
    with PutProbe() as probe:
        refs = [probe.put(store, value, None) for value in shards]
    print(json.dumps({"ray": ray.__version__, "units": "nanoseconds; size in bytes", "shards": probe.rows}))
