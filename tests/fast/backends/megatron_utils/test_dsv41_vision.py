import ast
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from miles_plugins.models.deepseek_v4_1.image_processing import expand_image_tokens, image_grid, patchify, span_length
from miles_plugins.models.deepseek_v4_1.multimodal import image_positions, scatter_image_embeddings
from miles_plugins.models.deepseek_v4_1.processor import DeepSeekV41Processor
from miles_plugins.models.deepseek_v4_1.vision import Aligner, DeepSeekV41Visual, VisionConfig, vision_rope
from miles_plugins.models.deepseek_v4_1.weights import FP4_VALUES, decode_weight


@pytest.fixture
def small_config():
    return VisionConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=2,
        downsample_ratio=3,
        min_pixels=16,
        max_image_tokens=64,
    )


def test_native_visual_tensor_inventory():
    with torch.device("meta"):
        visual = DeepSeekV41Visual(VisionConfig(), 5120)
    names = list(visual.state_dict())
    assert len(names) == 266
    assert sum(name.startswith("vision.") for name in names) == 259
    assert sum(name.startswith("aligner.") for name in names) == 4
    assert {name for name in names if "." not in name} == {"image_start", "image_end", "image_newline"}


def test_native_rope_reciprocal_arithmetic():
    cos, sin = vision_rope(32, 48, 64, 10000.0, "cpu")
    inverse = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    h, w = torch.meshgrid(torch.arange(32), torch.arange(48), indexing="ij")
    phase = (torch.stack((h, w), dim=-1).reshape(-1, 2, 1).float() * inverse).flatten(1).unsqueeze(1)
    assert torch.equal(cos, phase.cos()) and torch.equal(sin, phase.sin())


def test_native_vision_norms_keep_fp32_in_bf16_model(small_config):
    visual = DeepSeekV41Visual(small_config, 16).bfloat16()
    assert visual.vision.patch_embed.proj.weight.dtype == torch.bfloat16
    assert visual.vision.blocks[0].norm1.weight.dtype == torch.float32
    assert visual.vision.norm.weight.dtype == torch.float32
    assert visual.vision.norm.weight.keep_in_fp32


def test_native_message_blocks_preserve_text_and_images():
    processor = object.__new__(DeepSeekV41Processor)
    original = [
        {"role": "user", "content": "<image>\nCount."},
        {"role": "assistant", "content": "3", "reasoning_content": ""},
    ]
    normalized = processor.normalize_messages(original, ["scene.png"])
    assert normalized[0]["content_blocks"][1] == {"type": "image_url", "image_url": {"url": "scene.png"}}
    assert normalized[1] == original[1]
    assert original[0]["content"] == "<image>\nCount."
    with pytest.raises(ValueError, match="resolved"):
        processor.normalize_messages(original)


@pytest.mark.parametrize("scope", ["none", "projector", "full"])
def test_visual_gradient_and_real_weight_updates(small_config, scope):
    torch.manual_seed(42)
    visual = DeepSeekV41Visual(small_config, 16)
    visual.set_trainable_scope(scope, sequence_parallel=True)
    before = {name: param.detach().clone() for name, param in visual.named_parameters()}
    patches = torch.randn(20, 3, 2, 2)
    features = visual(patches, torch.tensor([[4, 5]]))
    if scope != "none":
        optimizer = torch.optim.SGD((p for p in visual.parameters() if p.requires_grad), lr=0.01)
        (features * torch.randn_like(features)).sum().backward()
        for name, param in visual.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
                assert param.grad.abs().sum() > 0, name
                assert param.sequence_parallel
            else:
                assert param.grad is None
        optimizer.step()
    for name, param in visual.named_parameters():
        assert torch.equal(before[name], param) == (not param.requires_grad), name


def test_aligner_padding_and_channel_major_unfold(small_config):
    aligner = Aligner(small_config, 16)
    x = torch.randn(20, 16)
    image = F.pad(x.reshape(4, 5, 16).permute(2, 0, 1), (0, 1, 0, 2))
    cells = image.unfold(1, 3, 3).unfold(2, 3, 3).permute(1, 2, 0, 3, 4).reshape(4, -1)
    expected = aligner.w2(F.gelu(aligner.w1(cells)))
    torch.testing.assert_close(aligner(x, 4, 5), expected)


def test_recompute_preserves_outputs_and_gradients(small_config):
    eager = DeepSeekV41Visual(small_config, 16)
    recompute = DeepSeekV41Visual(small_config, 16, recompute=True)
    recompute.load_state_dict(eager.state_dict())
    patches, grids = torch.randn(20, 3, 2, 2), torch.tensor([[4, 5]])
    outputs = [model(patches, grids) for model in (eager, recompute)]
    torch.testing.assert_close(*outputs)
    for output in outputs:
        output.square().sum().backward()
    for left, right in zip(eager.parameters(), recompute.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("width,height", [(1, 1), (56, 36), (746, 1176), (100000, 1), (1, 100000)])
def test_resize_stays_inside_budget(width, height):
    config = VisionConfig()
    h, w = image_grid(width, height, config)
    assert h > 0 and w > 0 and h % 14 == 0 and w % 14 == 0
    assert span_length(h // 14, w // 14, 3) <= config.max_image_tokens


def test_patch_channel_order_and_normalization(small_config):
    image = Image.new("RGB", (4, 4), (255, 0, 127))
    patches, grid = patchify(image, small_config)
    assert grid == (2, 2)
    assert patches.shape == (4, 3, 2, 2)
    assert torch.equal(patches[:, 0], torch.ones_like(patches[:, 0]))
    assert torch.equal(patches[:, 1], -torch.ones_like(patches[:, 1]))


def test_multi_image_expansion_preserves_loss_offsets(small_config):
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (6, 8))]
    tokens, mask, inputs = expand_image_tokens(
        [7, 99, 8, 99, 9, 10], images, 99, small_config, loss_mask=[0, 0, 0, 0, 1, 1]
    )
    first, second = [span_length(h, w, 3) for h, w in inputs["dsv41_grids"].tolist()]
    assert inputs["dsv41_image_positions"].tolist() == [1, first + 2]
    assert len(tokens) == first + second + 4
    assert sum(mask) == 2 and mask[-2:] == [1, 1]
    assert not any(value for token, value in zip(tokens, mask, strict=True) if token == 99)


def test_image_count_mismatch_and_target_images_fail(small_config):
    with pytest.raises(ValueError, match="placeholders"):
        expand_image_tokens([99], [], 99, small_config)
    with pytest.raises(ValueError, match="supervision"):
        expand_image_tokens([99], [Image.new("RGB", (4, 4))], 99, small_config, loss_mask=[1])


def test_tp_scatter_partial_gradients_sum_to_unsharded():
    positions = torch.tensor([1, 2, 3])
    features = torch.randn(3, 4, requires_grad=True)
    decoder = torch.randn(16, 1, 4, requires_grad=True)
    global_loss = scatter_image_embeddings(decoder, positions, features).square().sum()
    expected = torch.autograd.grad(global_loss, (decoder, features))
    partial_decoder, partial_features = [], []
    for rank in range(4):
        local = decoder.detach()[rank * 4 : (rank + 1) * 4].clone().requires_grad_()
        replica = features.detach().clone().requires_grad_()
        loss = scatter_image_embeddings(local, positions, replica, tp_rank=rank, sequence_parallel=True).square().sum()
        grads = torch.autograd.grad(loss, (local, replica))
        partial_decoder.append(grads[0])
        partial_features.append(grads[1])
    torch.testing.assert_close(torch.cat(partial_decoder), expected[0])
    torch.testing.assert_close(torch.stack(partial_features).sum(0), expected[1])


def test_image_span_validation():
    assert image_positions(torch.tensor([2]), torch.tensor([[3, 3]]), 3, "cpu").tolist() == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="overlap"):
        image_positions(torch.tensor([2, 3]), torch.tensor([[3, 3], [3, 3]]), 3, "cpu")
    with pytest.raises(ValueError, match="outside"):
        scatter_image_embeddings(torch.zeros(2, 1, 4), torch.tensor([2]), torch.zeros(1, 4))


def test_native_fp4_unpack_and_scale():
    packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16).view(torch.int8)
    scale = torch.full((16, 1), 2.0)
    result = decode_weight(packed, scale, name="layers.0.ffn.experts.0.w1.weight")
    table = torch.tensor(FP4_VALUES)
    expected = (
        torch.tensor(
            [[table[int(value) & 15] * 2, table[int(value) >> 4] * 2] for value in packed.view(torch.uint8).flatten()]
        )
        .reshape(16, 32)
        .bfloat16()
    )
    torch.testing.assert_close(result, expected)


def test_native_fp8_block_scale():
    weight = torch.ones(64, 64).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    expected = scale.repeat_interleave(32, 0).repeat_interleave(32, 1).bfloat16()
    torch.testing.assert_close(decode_weight(weight, scale), expected)
    with pytest.raises(ValueError, match="scale"):
        decode_weight(weight)


def test_hf_export_includes_all_visual_tensors_and_vl_bias(small_config):
    path = Path("miles/backends/megatron_utils/megatron_to_hf/deepseekv4_1.py")
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "convert_deepseekv4_1_to_hf"
    )
    scope = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    convert = scope["convert_deepseekv4_1_to_hf"]
    visual = DeepSeekV41Visual(small_config, 16)
    for name, param in visual.state_dict().items():
        assert convert(SimpleNamespace(), "module.module.visual." + name, param)[0][0] == name
    assert (
        convert(None, "module.module.decoder.layers.0.mlp.router.expert_bias_vl", torch.ones(4))[0][0]
        == "layers.0.ffn.gate.bias_vl"
    )


def test_sft_rollout_marks_completed_and_preserves_sample_groups(monkeypatch):
    from miles.utils.types import Sample
    from miles_plugins.models.deepseek_v4_1 import sft

    sample = Sample(index=0, prompt=[{"role": "assistant", "content": "3"}], multimodal_inputs={"images": ["x.png"]})
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "3"))
    monkeypatch.setattr(sft, "_processor", lambda _: processor)
    monkeypatch.setattr(sft, "encode_sft_sample", lambda *args, **kwargs: ([7, 8, 9], [0, 1, 1], {"patches": 1}))
    args = SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint="x",
        rollout_batch_size=1,
        apply_chat_template_kwargs={},
        seq_length=8,
    )
    data = SimpleNamespace(get_samples=lambda _: [[sample]])
    result = sft.generate_rollout(args, 0, data)
    assert result == [[sample]]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.response == "3" and sample.response_length == 2 and sample.loss_mask == [1, 1]
    assert sample.multimodal_train_inputs == {"patches": 1}


@pytest.mark.parametrize("restore", [True, False])
def test_native_reload_witness_checks_every_visual_tensor_and_restores_args(monkeypatch, tmp_path, restore):
    from miles_plugins.models.deepseek_v4_1 import checkpoint_audit

    parameters = [(f"module.module.visual.tensor{i}", torch.nn.Parameter(torch.ones(2))) for i in range(266)]
    parameters += [
        ("module.module." + name, torch.nn.Parameter(torch.ones(2)))
        for name in ("mlp.router.weight", "mlp.experts.linear_fc1.weight0", "self_attention.linear_q_up_proj.weight")
    ]
    saved = {name: parameter.detach().clone() for name, parameter in parameters}
    model = [SimpleNamespace(named_parameters=lambda: iter(parameters))]
    args = SimpleNamespace(
        async_save=False,
        save="native",
        load="base",
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
    )
    original = vars(args).copy()
    module = ModuleType("miles.backends.megatron_utils.checkpoint")

    def load(*positional, **kwargs):
        assert args.load == "native" and args.finetune and args.no_load_optim
        assert torch.isnan(parameters[0][1]).all()
        if restore:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(saved[name])
        return (0, False)

    module.load_checkpoint = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("DSV41_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoint_audit.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_audit.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    if restore:
        checkpoint_audit.verify_saved_model(args, 1, model)
        assert (tmp_path / "checkpoint-reload/rank-000.json").is_file()
        assert all(torch.equal(parameter, saved[name]) for name, parameter in parameters)
    else:
        with pytest.raises(ValueError, match="failed to restore"):
            checkpoint_audit.verify_saved_model(args, 1, model)
        assert not (tmp_path / "checkpoint-reload/rank-000.json").exists()
    assert vars(args) == original


def test_discarded_optimizer_step_does_not_nest_audit_wrappers(monkeypatch, tmp_path):
    from miles_plugins.models.deepseek_v4_1 import audit

    calls = []

    def original():
        calls.append(1)
        return (True, 0, 0)

    optimizer = SimpleNamespace(step=original)
    model = [SimpleNamespace(named_parameters=lambda: iter([]))]
    args = SimpleNamespace(dsv41_freeze_language=False)
    monkeypatch.setenv("DSV41_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("DSV41_VISUAL_SNAPSHOT", raising=False)
    monkeypatch.setattr(audit.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    audit.before_step(args, 0, 0, model, optimizer, None)
    audit.before_step(args, 1, 0, model, optimizer, None)
    assert optimizer.step._dsv41_original_step is original
    optimizer.step()
    assert calls == [1] and optimizer.step is original


@pytest.mark.parametrize("agree", [True, False])
def test_visual_snapshot_uses_only_its_stage_replica_group(monkeypatch, tmp_path, agree):
    from miles_plugins.models.deepseek_v4_1 import checkpoint_audit

    core = ModuleType("megatron.core")
    group = SimpleNamespace(size=lambda: 2)
    core.parallel_state = SimpleNamespace(get_tensor_and_data_parallel_group=lambda: group)
    monkeypatch.setitem(sys.modules, core.__name__, core)
    parameters = [(f"module.module.visual.tensor{i}", torch.nn.Parameter(torch.ones(2))) for i in range(266)]
    model = [SimpleNamespace(named_parameters=lambda: iter(parameters))]

    def gather(hashes, expected, **kwargs):
        assert kwargs["group"] is group and len(hashes) == 2
        hashes[:] = [expected, expected if agree else "different"]

    monkeypatch.setattr(checkpoint_audit.dist, "all_gather_object", gather)
    monkeypatch.setattr(checkpoint_audit.dist, "get_rank", lambda: 0)
    root = tmp_path / "snapshot"
    if agree:
        checkpoint_audit.save_visual_snapshot(model, root)
        assert (root / "visual.safetensors").is_file()
        assert (root / "visual.json").is_file()
    else:
        with pytest.raises(ValueError, match="disagree"):
            checkpoint_audit.save_visual_snapshot(model, root)
        assert not root.exists()
