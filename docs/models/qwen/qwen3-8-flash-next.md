---
title: Qwen3.8-Flash-Next
description: RL recipe for Qwen3.8-Flash-Next, the GDN + QSA hybrid MoE preview of the Qwen4 architecture, with hyper-connections and a host-resident PLE table.
---

The complete Qwen3.8-Flash-Next RL implementation is open at the Miles pull request:
[`radixark/miles#2777`](https://github.com/radixark/miles/pull/2777). It lands as a set of
three: that branch, the SGLang
[`sglang-miles-qwen38next`](https://github.com/sgl-project/sglang/tree/sglang-miles-qwen38next)
branch, and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89).
The image in [section 3](#3-environment-setup) pins all three.

For native image supervision rather than generated RL rollouts, see
[Image SFT](#7-image-sft). That integration has separate qualification evidence;
the RL recipe and its sliced CI test do not establish visual training quality.

## 1. Model Introduction

[Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
is Qwen's **176 B-parameter (6 B active) GDN + QSA hybrid Mixture-of-Experts preview of the
Qwen4 architecture**. Despite the shared prefix it is not a variant of the dense
[Qwen3.8-27B](/models/qwen/qwen3-8) — it is the next step of the *Next* line that
[Qwen3-Next](/models/qwen/qwen3-next) started, and structurally it has far more in common
with that model than with anything else carrying the Qwen3.8 name.

The block spec is still a uniform GPT decoder, but three components sit outside what a
stock Megatron layer provides, and each one is why a piece of this recipe exists:

**Hyper-connections replace the block layernorms.** The checkpoint ships none: each
hyper-connection's own `hc_norm` *is* the pre-block norm, so the spec drops every block
layernorm and a leftover TE fused norm corrupts the forward pass silently rather than
loudly. Miles fills Megatron's HC `ModuleSpec` slots, but Qwen's output contraction is not
DeepSeek-V4's `learned_output_contract` — it is the same low-rank gated mean as the
per-layer hyper-connections, with the RMS taken per stream rather than over the whole
`n*C` vector. That needed a new Megatron spec slot, `hc_head_contraction`, which is what
Megatron#89 adds; the DeepSeek-V4 default path and its parameter names are untouched.

**QSA (Qwen Sparse Attention) on the full-attention layers.** Twelve of the 48 layers are
full attention. Each projects its own indexer queries and compressed keys, scores them, and
keeps `indexer_budget` key tokens per query; attention then reads only those. Miles
reimplements the indexer and a forward *and backward* Triton sparse-attention kernel rather
than importing SGLang's inference path, with the selection rows built torch-side so the
kernel needs no causal or segment logic of its own.

**A frozen, host-resident PLE table.** The per-layer-embedding n-gram table is ~102 GB. It
lives in host memory, TP-row-sharded, and is deliberately *not* a checkpointed parameter;
token ids reach it over an explicit side channel that raises rather than defaulting when
nothing was published. It is also on the weight-update check's skip list — see
[section 5.5](#55-notable-quirks).

**Key highlights:**

- **48 layers, hybrid**: 36 GDN linear-attention layers + 12 QSA full-attention layers, from
  the released config's `layer_types` (equivalently, every 4th layer is full attention).
- **512-expert MoE at top-10**, `moe-ffn-hidden-size 640`, plus a gated shared expert.
- **Hyper-connections** at every block, with a model-supplied output contraction.
- **PLE** n-gram embeddings, frozen and host-resident.
- **Attention output gate**, `--qk-layernorm`, `--apply-layernorm-1p`.
- **Shape**: hidden 2560, 24 attention heads, 2 query groups, `kv-channels 256`,
  vocab 248320, `--rotary-base 10000000` at `--rotary-percent 0.25`.

Two things the model args deliberately leave out: MTP (`--mtp-num-layers` is omitted, the
MTP tensors are not mapped yet) and the hyper-connection / PLE / QSA fields themselves,
which have no Megatron CLI flags and are derived from the checkpoint by the spec in
`miles_plugins/models/qwen3_8_next/qwen3_8_next.py`.

On the parameter count: Qwen's published headline is 176 B total / 6 B active, while the
docstring in `scripts/models/qwen3.8-flash-next.py` counts 180 B / ~7.4 B from the released
`config.json` shapes. Nothing in the recipe depends on which figure you quote.

## 2. Supported Variants

| Variant | `--model-name` | Layers | Purpose | GPUs |
|---|---|---|---|---|
| Full | `Qwen3.8-Flash-Next` | 48 | the real model | 32 (8 × 4) |
| Smoke slice | `Qwen3.8-Flash-Next-4layer` | 4 | CI and single-node bring-up | 1 × 4 or 1 × 8 |

`--model-name` selects between them and sets the matching `megatron_model_type`
(`qwen3.8-flash-next` / `qwen3.8-flash-next-4layer`), which in turn resolves the model args
and the `torch_dist` path. The launcher asserts the node shape, so a mismatched
`--num-nodes` / `--num-gpus-per-node` fails immediately rather than mid-run.

## 3. Environment Setup

Use the `docker.io/radixark/miles:qwen38next` image. It is the rolling
[`radixark/miles:dev`](/developer/ci/02-docker-build) image with the three moving parts checked out at
the versions this recipe was built against, and nothing else changed — every prebuilt wheel,
TransformerEngine patch and version pin comes from `dev`. It is multi-arch, so the same tag
serves GB300 (aarch64) and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2777`](https://github.com/radixark/miles/pull/2777) `afd78afd` |
| SGLang | [`sglang-miles-qwen38next`](https://github.com/sgl-project/sglang/tree/sglang-miles-qwen38next) `599d7403` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

### 3.1 Download model + dataset

```bash
hf download Qwen/Qwen3.8-Flash-Next --local-dir /root/models/Qwen3.8-Flash-Next
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

For the smoke slice, download
[`CharyZeng/Qwen3.8-Flash-Next-4layer`](https://huggingface.co/CharyZeng/Qwen3.8-Flash-Next-4layer)
into `/root/models/Qwen3.8-Flash-Next-4layer` instead — the launcher resolves
`--hf-checkpoint` as `<--model-dir>/<--model-name>`, which is the same path CI uses.

### 3.2 HF → Megatron `torch_dist` conversion

Unlike the launcher-prepared recipes, this one takes the converted reference checkpoint as
given: `--ref-load` resolves to `<--ckpt-dir>/<megatron_model_type>_torch_dist`, so the
conversion has to have run first. The output re-shards at load, so the conversion layout
does not have to match the training one:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3.8-flash-next)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/models/Qwen3.8-Flash-Next \
   --save          /root/ckpt/qwen3.8-flash-next_torch_dist \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 1
```

Swap `qwen3.8-flash-next` for `qwen3.8-flash-next-4layer` throughout to convert the smoke
slice.

## 4. Launch

The launcher assumes an **already-running ray cluster**: bring one up across the nodes,
`export MILES_SCRIPT_EXTERNAL_RAY=1`, then run on the head node.

Full model, 8 nodes × 4 GPUs:

```bash
cd /root/miles
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next \
  --num-nodes 8 --num-gpus-per-node 4 \
  --num-rollout 5 --rollout-max-response-len 4096
```

Single-node smoke slice:

```bash
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next-4layer \
  --num-nodes 1 --num-gpus-per-node 8
```

Paths come from `--model-dir` (default `/root/models`), `--data-dir` (default
`/root/datasets`), `--ckpt-dir` (default `/root/ckpt`) and `--megatron-path` (default
`/root/Megatron-LM`). Saving is off by default (`skip_saving`); turning it on writes
checkpoints under `<--save-dir>/<--run-id>/checkpoints` every 10 rollouts, without optimizer
or RNG state.

## 5. Recipe Configuration

### 5.1 Parallelism

| Shape | TP | PP | CP | EP | ETP | Rollout engine |
|---|---|---|---|---|---|---|
| 8 × 4 (full) | 2 | 8 | 1 | 4 | 1 | 8 GPUs, SGLang TP 8 / EP 8 |
| 1 × 4 (4layer) | 2 | 2 | 1 | 2 | 1 | 4 GPUs, SGLang TP 4 / EP 4 |
| 1 × 8 (4layer) | 2 | 2 | 1 | 4 | 1 | 4 GPUs, SGLang TP 4 / EP 4 |

`--sequence-parallel` is on in every shape. Activation checkpointing is full and uniform at
one layer, with `--micro-batch-size 1` and `--max-tokens-per-gpu 8192`.

### 5.2 Algorithm

GRPO, DAPO-Math-17k, thinking mode on:

```bash
--advantage-estimator grpo
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
--rollout-batch-size 4
--n-samples-per-prompt 8
--rollout-temperature 0.8
--apply-chat-template-kwargs '{"thinking_mode":"thinking"}'
```

Adam at `--lr 1e-6`, constant schedule, `--weight-decay 0.1`,
`--adam-beta1 0.9 --adam-beta2 0.98`.

### 5.3 Rollout & SGLang

Rollout is **colocated** — trainer and engines share the GPUs — and the trainer offloads to
**disk** rather than host RAM during rollout (`--offload-train-target disk`,
`--offload-train-disk-dir /tmp/train_offload`). Worth keeping in view when sizing hosts: the
PLE table is already holding ~102 GB of host memory for the full model. The 4-layer CI slice
overrides the target back to `--offload-train-target cpu`.

```bash
--sglang-linear-attn-prefill-backend flashinfer   # GDN prefill
--sglang-moe-runner-backend triton
--sglang-chunked-prefill-size 8192
--sglang-disable-radix-cache
--sglang-mem-fraction-static 0.7
--linear-attention-backend flashqla               # trainer-side GDN
--qkv-format thd
```

The recipe loosens router
health checking (`--router-health-failure-threshold 40`,
`--router-health-check-interval-secs 15`, `--router-health-success-threshold 1`) and raises
`--rollout-health-check-interval` / `--rollout-health-check-timeout` to 300 s.

### 5.4 What CI watches

`tests/e2e/megatron/model_scripts/test_qwen3_8_next_4layer_ci.py` runs the 4-layer slice on
8 × H200 in `stage-c-8-gpu-h200` with rollout-routing replay on, and gates these metrics:

- `train/grad_norm`
- `train/ppo_kl`
- `train/train_rollout_logprob_abs_diff`
- `train/train_rollout_kl`
- `rollout/raw_reward`

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up: it is
the direct measure of whether the SGLang and Megatron forward passes agree, and on this
architecture that covers the GDN, QSA and hyper-connection paths at once. See
[True On-Policy](/examples/infra-features/true-on-policy) for what the metric does and does
not tell you.

### 5.5 Notable quirks

- **Weight-update checking skips two prefixes.** `--check-weight-update-equal` runs with
  `--check-weight-update-skip-list visual. ple_embedding.` — the PLE table is frozen and
  never shipped, so comparing it would fail on a parameter that is working as intended.
- **Every block layernorm is dropped.** The checkpoint has none; each hyper-connection's
  `hc_norm` is the pre-block norm. A leftover TE fused norm corrupts silently.
- **`--moe-aux-loss-coeff 0`.** Routing is not auxiliary-loss balanced here.
- **Triton and Inductor caches are pinned to `/tmp`** (`TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`) with `TORCHINDUCTOR_COMPILE_THREADS=1`.
- **`SGLANG_DISABLE_MULTIMEM_AG=1`** and `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1` are set for
  the engines.
- The model lives in `miles_plugins/models/qwen3_8_next/`, and weight conversion in
  `miles/backends/megatron_utils/megatron_to_hf/qwen3_8_next.py`.

## 6. Pairs Well With

- [Qwen3-Next](/models/qwen/qwen3-next) — the previous generation of the same line
- [Qwen3.8](/models/qwen/qwen3-8) — the dense 27 B that shares the name and not the architecture
- [Disk Offload](/advanced/disk-offload)
- [True On-Policy](/examples/infra-features/true-on-policy)

## 7. Image SFT

### Scope And Qualification

`scripts/run_qwen3_8_next_vision_sft.py` implements image SFT for the **complete
48-layer** `Qwen/Qwen3.8-Flash-Next` checkpoint. It does not substitute a decoder
slice, a text-only checkpoint, a LoRA adapter, or a different Qwen family.

The provider attaches the entire native Qwen3-VL visual encoder and merger before
Megatron constructs DDP and the optimizer. `--vision-trainable` selects `none`,
`projector`, or `full`; `--freeze-language` trains vision with the decoder frozen.
Without that flag the recipe jointly trains language and vision. The native PLE
lookup table remains an immutable host asset; the discrete QSA indexer is frozen
because its top-k selection does not have task-loss gradients. MTP is not enabled.

The implementation uses the native image processor and whole-conversation chat
template, masks image/user tokens, and supervises the assistant spans after image
token expansion. The released template's thinking option is `enable_thinking`,
not `thinking_mode`. For non-thinking integer supervision, the target is the
integer itself: do not prepend a GLM-style `</think>` suffix.

Image positions use the checkpoint's three-axis, interleaved mRoPE configuration.
The QSA indexer uses physical packed-token rows when indexing these materialized
positions; text-only position resets are not image positions. Image embeddings
replace their exact placeholder rows on each sequence-parallel slice. Visual
replicas participate in tensor-parallel gradient reduction, including slices with
no local image tokens.

Focused CPU contracts pass for image/mRoPE alignment, assistant masks, visual
gradients and updates, recomputation, TP scatter, native weight naming, evaluation
parsing, and launcher safety. The full 48-layer model completed the fixed
32-update vision-only recipe on one eight-GPU B300 node with TP2/PP4/EP2,
batch size 32, learning rate `1e-6`, zero weight decay and `enable_thinking=True`.
Both visual replicas had finite, nonzero gradient probes for all 333 visual
tensors; sampled BF16 values changed in 224 tensors on the final update. Their
complete visual hashes agreed, and snapshots at updates 0/8/16/24/32 passed
exact safetensors readback. All eight complete decoder hashes remained unchanged.
Native Megatron save/reload passed on every rank after deliberately corrupting
every registered parameter, with exact complete-parameter hash restoration.
This qualifies model-only reload, not optimizer/data-cursor resume. The native
checkpoint is iteration 31 because rollout indices start at zero. A preceding
two-update, batch-8 qualification separately passed the same model-only reload
check; its sampled change count was 235 on the second update.

The 32-update run used training source `9e1cb86174cca03279c55db263cc40f8b08fe467`.
Its [fixed protocol](../../assets/images/qwen3-8-flash-next-vision-sft/protocol.json) has SHA256
`5ca5c2a3174c2b09664ef0f605fd9fb7eb223fa83b90a2e97f887db473be27db`;
the final full visual hash is
`4a595c9f5cdb3c856879fe2254d579cfff43413b4246eea475074ac7a736e275`.
The project retains all per-rank runtime/source audits, snapshot receipts and
native reload receipts under `checkpoints/experiments/vision32-v1/audit/`, with
their consolidated qualification in `logs/train-vision32-v1-qualification.json`.
The [recorded training qualification](../../assets/images/qwen3-8-flash-next-vision-sft/training-qualification.json)
includes all eight ranks, actual imported source hashes, and snapshot receipts.
Training used PyTorch `2.11.0+cu130`, Transformers `5.8.1` and Ray `2.56.0`, with
Megatron at `eed6851e5088cff7c10ee195de01f83e21094652`.
**A measured Qwen held-out improvement remains unqualified until the fixed
evaluation finishes.** Training completion and parameter changes do not establish
visual quality.

### Data And Launch

The example reuses the archived `geometric_stack_count` scenes referenced by
[issue #1](https://github.com/leviking98z-rgb/miles/issues/1) and
[issue #4](https://github.com/leviking98z-rgb/miles/issues/4), not their model scores.
The archive pins 1024 train, 32 development and 256 held-out scenes with exact
generator targets and decoded-pixel hashes. Generation requires the pinned
Sphinx checkout; see `prepare.py --help` for its required revision and options.
The model receives the question and image, not scene metadata or the oracle.

```bash
python -m examples.qwen3_8_next_vision_sft.prepare --help
python scripts/run_qwen3_8_next_vision_sft.py --help

MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_8_next_vision_sft.py \
  --hf-checkpoint /root/models/Qwen3.8-Flash-Next \
  --prompt-data /root/datasets/geometry-v1/sft_exact.jsonl \
  --output-dir /root/runs/qwen38-image \
  --freeze-language --enable-thinking
```

Run on an allocated GPU node with a compatible Megatron and SGLang environment.
Use `MILES_SCRIPT_EXTERNAL_RAY=1` only for an already-running cluster. The launcher
does not globally kill existing SGLang, Miles, Redis or Ray processes by default.
Unsupported CP, interleaved PP, routing replay, dynamic microbatching and fused
RoPE combinations fail instead of silently changing the image contract.

Recipe defaults and their validation are owned by the launcher's `ScriptArgs`.
The initial experiment fixes 32 optimizer updates, one 32-example batch per
update, constant learning rate, zero weight decay, and visual snapshots at updates
0/8/16/24/32. All hyperparameters and the development/held-out separation must be
fixed before reading held-out results; do not select a learning rate or checkpoint
from the held-out curve.

### Audits And Evaluation

The opt-in audit writes per-rank parameter inventories, finite reduced-gradient
probes, actual parameter-change probes, and imported runtime source paths/hashes.
Frozen-language mode hashes all registered decoder parameters before and after
training. Visual snapshots require
identical full visual hashes on all replicas and an exact safetensors readback.
These visual snapshots are **evaluation-only and nonresumable**. Native Megatron
checkpoints are saved separately; optimizer/data-cursor resume is not established
by a model-only save or a visual snapshot.

Run the evaluator on the same allocated GPU node after training, not on a control
machine. It starts the full native SGLang model and evaluates the source baseline
plus each explicitly supplied visual snapshot. A visual-only update is accepted
only after every training rank closes its unchanged-language audit. Native visual
weight loading is checked by exact TP-shard readback before generation.

```bash
python -m examples.qwen3_8_next_vision_sft.evaluate \
  --model /root/models/Qwen3.8-Flash-Next \
  --data /root/datasets/geometry-v1 \
  --output /root/runs/qwen38-image/evaluation --enable-thinking \
  --visual-snapshots \
    /root/runs/qwen38-image/RUN_ID/audit/snapshots/update-000008 \
    /root/runs/qwen38-image/RUN_ID/audit/snapshots/update-000016 \
    /root/runs/qwen38-image/RUN_ID/audit/snapshots/update-000024 \
    /root/runs/qwen38-image/RUN_ID/audit/snapshots/update-000032

python -m examples.qwen3_8_next_vision_sft.curves \
  --evaluation /root/runs/qwen38-image/evaluation \
  --data /root/datasets/geometry-v1 \
  --output /root/runs/qwen38-image/curve
```

The evaluator records raw text for genuine-image, no-image and deterministic
swapped-image controls, using greedy decoding with 4096 new tokens, native expanded
prompt IDs, and `integer_first_final_line_v1`. It retains the template's assistant
prefill separately from generated text and includes it when scoring, so an unfinished
native `<think>` block cannot become a valid final answer. Use `--enable-thinking`
consistently for the reasoning protocol in training and evaluation.
`--limit` is for bring-up only and cannot
produce an accepted quality curve. The plotter re-scores raw predictions and
rejects missing records, changed prompts/images, mismatched dataset hashes or
inconsistent aggregates. It reports answer-macro accuracy, paired wrong/right
transitions, invalid-answer changes and the always-one reference. Every plotted
dot must be a measured checkpoint; dips are preserved, not smoothed away.

With multiple allocated eight-GPU nodes, evaluate the baseline on one worker and
disjoint groups of visual snapshots on others, passing `--skip-baseline` on those
checkpoint workers. Keep every generation, data, and control option identical.
After all workers finish, combine their output directories with
`python -m examples.qwen3_8_next_vision_sft.merge --evaluations WORKER_DIRS
--data DATA_DIR --output MERGED_DIR`. The merger rejects duplicate updates or
different protocols, source hashes, and package versions, then re-scores the raw
predictions before accepting a complete curve. Checkpoint workers require the
completed frozen-language audits from training.

A falling training loss, successful weight loading, or scores from the reference
GLM/DeepSeek issues are not evidence of a Qwen evaluation improvement.
