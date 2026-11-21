"""Compare online and offline HF exports of the same 12-layer Qwen3.8 C3 checkpoint.

Run with two export directories and optionally --output <json> for a per-tensor receipt.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from miles.backends.training_utils.weight_update.hf_passthrough import tensor_hashes, weight_map


def inventory(directory: Path) -> dict:
    index = weight_map(directory)
    assert len(index) == 794 and sum(k.startswith("mtp.") for k in index) == 31
    shards = {p.name for p in directory.glob("*.safetensors")}
    assert shards == set(index.values())
    hashes = tensor_hashes(directory, index, index)
    entries = {}
    for shard in sorted(shards):
        with safe_open(directory / shard, framework="pt") as f:
            assert set(f.keys()) == {k for k, v in index.items() if v == shard}
            for name in f.keys():
                tensor = f.get_slice(name)
                entries[name] = {"dtype": tensor.get_dtype(), "shape": tensor.get_shape(), "sha256": hashes[name]}
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("online", type=Path)
    parser.add_argument("offline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    online, offline = inventory(args.online), inventory(args.offline)
    differences = sorted(k for k in online.keys() | offline.keys() if online.get(k) != offline.get(k))
    decays = [v for k, v in offline.items() if k.endswith(".linear_attn.A_log")]
    assert len(decays) == 9 and all(v["dtype"] == "F32" for v in decays), "expected native FP32 offline reference"
    if args.output:
        args.output.write_text(
            json.dumps({"online": online, "offline": offline, "differences": differences}, indent=2)
        )
    print(f"{len(online)} keys, {len(differences)} differences: {differences}")
    raise SystemExit(bool(differences))
