"""CPU contracts for native Qwen images; full-model quality is measured separately."""

import ast
import gc
import re
import sys
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLVisionModel

from miles_plugins.models.qwen3_8_next.vision import set_visual_scope
from miles_plugins.models.qwen3_8_next.vision_inputs import (
    INTEGER_FINAL_INSTRUCTION,
    apply_final_answer_format,
    image_rope_positions,
    normalize_image_messages,
    scatter_image_embeddings,
)
from miles_plugins.models.qwen3_8_next.vision_sft import encode_sft_sample

_CONFIG = {
    "image_token_id": 9001,
    "vision_start_token_id": 9002,
    "vision_end_token_id": 9003,
    "vision_config": {"spatial_merge_size": 2},
}


def _positions(ids, grids):
    return image_rope_positions(
        ids,
        grids,
        image_token_id=9001,
        vision_start_token_id=9002,
        vision_end_token_id=9003,
        merge_size=2,
    )


def _tiny_visual():
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=24,
        out_hidden_size=16,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config._attn_implementation = "sdpa"
    return Qwen3VLVisionModel(config)


def test_native_image_positions_match_transformers_without_constructing_the_decoder():
    ids = torch.tensor([[2, 9002] + [9001] * 6 + [9003, 3, 9002] + [9001] * 4 + [9003, 4]])
    grids = torch.tensor([[1, 4, 6], [1, 4, 4]])
    stub = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    stub.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, stub)
    native, _ = Qwen3VLModel.get_rope_index(stub, ids, (ids == 9001).int(), image_grid_thw=grids)
    assert torch.equal(_positions(ids[0], grids).T.unsqueeze(1), native)


@pytest.mark.parametrize(
    "ids,grids",
    [
        ([2, 9001, 9003], [[1, 2, 2]]),
        ([9002, 9001, 9001, 9003], [[1, 2, 2]]),
        ([9002, 9001, 9003], [[1, 3, 2]]),
        ([9002, 9001, 9003], [[2, 2, 2]]),
        ([9002, 9001, 9003], [[1, 0, 2]]),
        ([9002, 9001, 9003], [[1, 2, 2], [1, 2, 2]]),
        ([9002, 9001], [[1, 2, 2]]),
    ],
)
def test_invalid_image_spans_fail_closed(ids, grids):
    with pytest.raises(ValueError):
        _positions(ids, grids)


def test_normalization_preserves_input_and_rejects_orphan_images():
    original = [{"role": "user", "content": "A<image>B<image>C"}]
    normalized = normalize_image_messages(original, 2)
    assert original[0]["content"] == "A<image>B<image>C"
    assert [part["type"] for part in normalized[0]["content"]] == ["text", "image", "text", "image", "text"]
    with pytest.raises(ValueError, match="resolved images"):
        normalize_image_messages(original, 1)
    with pytest.raises(ValueError, match="only in user"):
        normalize_image_messages([{"role": "assistant", "content": "<image>"}], 1)
    with pytest.raises(ValueError, match="video or audio"):
        normalize_image_messages([{"role": "user", "content": [{"type": "video"}]}], 0)


def test_tp_visual_gradient_slices_sum_to_the_unsharded_gradient():
    torch.manual_seed(11)
    decoder = torch.randn(16, 1, 8, requires_grad=True)
    images = torch.randn(3, 8, requires_grad=True)
    positions = torch.tensor([2, 3, 4])
    expected = torch.autograd.grad(
        scatter_image_embeddings(decoder, positions, images).square().sum(), (decoder, images)
    )
    image_grads, decoder_grads = [], []
    for rank in range(4):
        shard = decoder.detach()[rank * 4 : (rank + 1) * 4].requires_grad_()
        replica = images.detach().clone().requires_grad_()
        value = scatter_image_embeddings(shard, positions, replica, tp_rank=rank, sequence_parallel=True)
        dg, ig = torch.autograd.grad(value.square().sum(), (shard, replica))
        decoder_grads.append(dg)
        image_grads.append(ig)
    torch.testing.assert_close(torch.cat(decoder_grads), expected[0])
    torch.testing.assert_close(torch.stack(image_grads).sum(0), expected[1])
    assert torch.count_nonzero(image_grads[-1]) == 0


@pytest.mark.parametrize("scope", ["none", "projector", "full"])
def test_visual_scopes_have_real_gradients_and_weight_updates(scope):
    torch.manual_seed(42)
    visual = _tiny_visual()
    set_visual_scope(visual, scope, sequence_parallel=True)
    before = {name: p.detach().clone() for name, p in visual.named_parameters()}
    result = visual(torch.randn(16, 24), torch.tensor([[1, 4, 4]]), return_dict=True).pooler_output
    if scope != "none":
        optimizer = torch.optim.SGD((p for p in visual.parameters() if p.requires_grad), lr=0.01)
        (result * torch.randn_like(result)).sum().backward()
        for name, p in visual.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                assert p.grad.abs().sum() > 0, name
                assert p.sequence_parallel
            else:
                assert p.grad is None, name
        optimizer.step()
    for name, p in visual.named_parameters():
        assert torch.equal(before[name], p) == (not p.requires_grad), name


def test_visual_recompute_preserves_outputs_and_gradients():
    torch.manual_seed(42)
    eager, recompute = _tiny_visual(), _tiny_visual()
    recompute.load_state_dict(eager.state_dict())
    recompute.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    patches, grids = torch.randn(16, 24), torch.tensor([[1, 4, 4]])
    outputs = [model(patches, grids, return_dict=True).pooler_output for model in (eager, recompute)]
    torch.testing.assert_close(*outputs)
    for value in outputs:
        value.square().sum().backward()
    for left, right in zip(eager.parameters(), recompute.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)


class _Tokenizer:
    _special = {"<|image_pad|>": 9001, "<|vision_start|>": 9002, "<|vision_end|>": 9003}

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        result = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<|vision_start|><|image_pad|><|vision_end|>" if item["type"] == "image" else item["text"]
                    for item in content
                )
            result += f"[START]{message['role']}\n{content}[END]\n"
        return result + ("[START]assistant\n" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        pattern = "(" + "|".join(map(re.escape, self._special)) + "|.)"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))
        return {
            "input_ids": [self._special.get(match[0], ord(match[0][0])) for match in matches],
            "offset_mapping": [match.span() for match in matches],
        }


class _Processor:
    tokenizer = _Tokenizer()

    def __call__(self, text, images, **kwargs):
        raw = self.tokenizer(text[0])["input_ids"]
        counts = iter([grid[1] * grid[2] // 4 for grid in images])
        expanded = []
        for token in raw:
            expanded.extend([token] * (next(counts) if token == 9001 else 1))
        return {
            "input_ids": torch.tensor([expanded]),
            "image_grid_thw": torch.tensor(images),
            "pixel_values": torch.zeros(sum(grid[1] * grid[2] for grid in images), 24),
        }


def test_multiturn_masks_follow_whole_native_conversation_after_image_expansion():
    messages = [
        {"role": "user", "content": "<image>Q"},
        {"role": "assistant", "content": "7", "step_loss_mask": 0},
        {"role": "user", "content": "<image>R"},
        {"role": "assistant", "content": "9"},
    ]
    tokens, mask, inputs = encode_sft_sample(_Processor(), _CONFIG, messages, [[1, 4, 6], [1, 4, 4]])
    assert len(tokens) == len(mask) == len(inputs["qwen38_position_ids"])
    assert sum(token == 9001 for token in tokens) == 10
    selected = "".join(chr(token) for token, active in zip(tokens, mask, strict=True) if active)
    assert selected == "9[END]\n"
    assert not any(active for token, active in zip(tokens, mask, strict=True) if token == 9001)


def test_aligned_output_instruction_is_not_supervised_and_preserves_source_messages():
    messages = [{"role": "user", "content": "<image>Q"}, {"role": "assistant", "content": "9"}]
    options = []

    class Tokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            options.append(kwargs)
            return super().apply_chat_template(messages, **kwargs)

    processor = _Processor()
    processor.tokenizer = Tokenizer()
    tokens, mask, inputs = encode_sft_sample(
        processor,
        _CONFIG,
        messages,
        [[1, 4, 4]],
        template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
        final_answer_format="integer-only",
    )
    assert all(item["enable_thinking"] and item["reasoning_effort"] == "low" for item in options)
    assert "".join(chr(token) for token, active in zip(tokens, mask, strict=True) if active) == "9[END]\n"
    assert sum(token == 9001 for token in tokens) == 4
    assert len(inputs["qwen38_position_ids"]) == len(tokens)
    assert INTEGER_FINAL_INSTRUCTION in "".join(chr(token) for token in tokens if token < 9001)
    assert messages == [{"role": "user", "content": "<image>Q"}, {"role": "assistant", "content": "9"}]


def test_aligned_output_instruction_retains_an_existing_system_message():
    source = [{"role": "system", "content": "Read the image."}, {"role": "user", "content": "Q"}]
    aligned = apply_final_answer_format(source, "integer-only")
    assert len(aligned) == len(source)
    assert aligned[0]["content"] == "Read the image.\n\n" + INTEGER_FINAL_INSTRUCTION
    assert source[0]["content"] == "Read the image."
    assert apply_final_answer_format(source, "dataset") == source
    with pytest.raises(ValueError, match="Unknown final-answer format"):
        apply_final_answer_format(source, "unknown")


def test_qsa_materialized_rope_uses_absolute_token_rows_for_packed_documents():
    source = Path(__file__).parents[4] / "miles_plugins/models/qwen3_8_next/ops/qsa_indexer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen38NextQSAIndexer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "project_qk")
    calls = []

    def rope(x, table, positions):
        calls.append(positions.tolist())
        return x

    namespace = {
        "Tensor": torch.Tensor,
        "torch": torch,
        "PackedBlockLayout": object,
        "gemma_rmsnorm_last_dim": lambda x, *a: x,
        "compress_keys_by_mean_packed": lambda keys, layout: keys[[0, 3]],
        "apply_indexer_rope": rope,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    stub = SimpleNamespace(
        n_heads=1,
        head_dim=2,
        norm_eps=1e-6,
        q_layernorm=None,
        k_layernorm=None,
        compress_ratio=3,
        index_qk_proj=lambda x: (torch.ones(6, 4), None),
    )
    layout = SimpleNamespace(block_local=torch.tensor([0, 0]), block_first_token=torch.tensor([0, 3]))
    namespace["project_qk"](stub, torch.zeros(6, 2), torch.tensor([0, 1, 2, 0, 1, 2]), torch.zeros(6, 4), layout, True)
    assert calls == [[0, 1, 2, 3, 4, 5], [0, 3]]


def test_visual_export_names_are_native_hf_names():
    from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf

    value = torch.ones(2)
    result = convert_qwen3_8_next_to_hf(None, "module.module.vision_model.merger.norm.weight", value)
    assert result == [("model.visual.merger.norm.weight", value)]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "name,shape",
    [
        ("vision_model.blocks.0.mlp.linear_fc1.weight", (24, 16)),
        ("vision_model.merger.linear_fc1.bias", (64,)),
        ("vision_model.blocks.0.mlp.linear_fc2.weight", (16, 24)),
    ],
)
def test_native_visual_bridge_does_not_apply_decoder_tp_splitting(name, shape, tp_size):
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    source = torch.arange(torch.Size(shape).numel()).reshape(shape)
    replicas = bridge._weight_split_across_tp(name, source, torch.empty_like(source), tp_size)
    assert len(replicas) == tp_size
    assert all(torch.equal(replica, source) for replica in replicas)
    with pytest.raises(ValueError, match="replicated vision shape"):
        bridge._weight_split_across_tp(name, source, torch.empty(1), tp_size)


def test_hf_expert_reads_reuse_bounded_source_mappings(monkeypatch):
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    reads = []

    def load(name):
        reads.append(name)
        return torch.ones(2)

    monkeypatch.setattr(
        Qwen3_5Bridge, "_get_safetensor_io", lambda self, path: SimpleNamespace(load_one_hf_weight=load)
    )
    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    io = bridge._get_safetensor_io("unused")
    gate = io.load_one_hf_weight("gate")
    down = io.load_one_hf_weight("down")
    for _ in range(512):
        assert io.load_one_hf_weight("gate") is gate
        assert io.load_one_hf_weight("down") is down
    assert reads == ["gate", "down"]
    io.load_one_hf_weight("next-layer")
    assert io.load_one_hf_weight("gate") is not gate
    assert io.load_one_hf_weight.cache_info().currsize == 2
    bridge._cached_cpu_reader.cache_clear()
    assert io.load_one_hf_weight.cache_info().currsize == 0


def test_device_weight_cache_keeps_mmap_source_alive_until_eviction(monkeypatch):
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(Qwen3_5Bridge, "_weight_to_mcore_format", lambda self, name, weights: weights)
    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    source = torch.tensor([1.0, 2.0])
    reference = weakref.ref(source)
    first = bridge._weight_to_mcore_format("weight", [source])[0]
    assert bridge._weight_to_mcore_format("weight", [source])[0] is first
    del source
    gc.collect()
    assert reference() is not None
    for value in (3.0, 4.0):
        bridge._weight_to_mcore_format("other", [torch.full((2,), value)])
    gc.collect()
    assert reference() is None
    assert torch.equal(first, torch.tensor([1.0, 2.0]))


def test_qwen_runtime_does_not_require_the_optional_deepseek_v41_encoder(monkeypatch):
    from sglang.srt.entrypoints import openai

    from miles.utils.chat_template_utils import deepseek

    monkeypatch.delattr(openai, "encoding_dsv41", raising=False)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.openai.encoding_dsv41", None)
    assert deepseek.V4.template is not None
    with pytest.raises(ImportError):
        _ = deepseek.V41.template
