---
title: DeepSeek-V4.1 Flash 1017 Image SFT
description: Experimental joint language/vision SFT and causal vision-only controls for the native 1017 checkpoint.
---

## Status And Scope

This is a new training path, not a claim that the earlier text-only SFT trained
vision. The previous text recipe froze router gates and preserved vision as
source assets. This provider registers the native ViT, aligner and three image
boundary embeddings before DDP and optimizer construction. Joint mode leaves
language router weights trainable. It uses no LoRA adapters.

The original model is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Its visual inventory has 259 ViT
tensors, four aligner tensors and three boundary embeddings. Preserve the
composite `vision_config`; a text-only cut checkpoint is insufficient.

The term joint/full here is **not every byte of the source model**: native
FP8 Engram lookup tables remain immutable assets, discrete top-k indexer
parameters have no task-loss backward and are excluded from Adam, and
text/vision routing-selection biases are fixed checkpointed buffers. Engram
projections and router gate weights are trainable in joint mode. MTP/DSpark
are outside this target-model SFT recipe.

An initial complete 40-layer integration run on 24 H20 nodes (192 ranks,
TP8/PP6/DP4/EP32, CP1) completed one finite joint image update. With weight
decay disabled, all 266 visual tensors had finite nonzero reduced-gradient
probes and actual weight changes on every visual replica. All 32 replicas
had the same complete final visual hash. Language router, attention and expert
probes also changed. This establishes a real full-model image optimizer step,
not convergence or held-out improvement.

A separate two-update run saved a complete native `torch_dist` model checkpoint.
All 192 ranks passed in-memory parameter corruption followed by model-only
reload; every visual tensor was included, with a complete visual hash check on
each of the 32 replicas. This does not test optimizer or data-cursor resume.

Training resume, native HF export and held-out image-task improvement remain
unqualified until the acceptance procedure below succeeds. CPU tests alone do
not establish those results. The native visual forward and image preprocessing
were also checked bitwise against the pinned DeepSeek reference on H20.

## Image Contract

Use one JSONL record per conversation, a `messages` field and an `images` list
of readable local paths. Put one `<image>` placeholder per path, in the same
order. The launcher uses Miles' multimodal message builder to associate the
paths with the placeholders. The processor then uses the real checkpoint image
token, not the literal training placeholder.

```json
{"messages":[{"role":"user","content":"<image>How many marked objects are inside the requested sheet?"},{"role":"assistant","reasoning_content":"","content":"3"}],"images":["/root/datasets/image-sft/images/example.png"]}
```

The existing native DeepSeek renderer owns chat formatting. Supervision offsets
come from a single whole-conversation encoding; image spans are always masked.
The native resize and 3x3 channel-major aligner order are preserved. Every image
slot, including start/end/newline, carries the checkpoint's image token ID.
Engram hashing excludes image spans, and image tokens select the native visual
routing bias. Missing images or token/feature mismatches fail rather than fall
back to text-only training.

## Launch

On an already allocated external Ray cluster:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_deepseek_v4_1_vision_sft.py \
  --hf-checkpoint /root/models/DeepSeek-V4.1-Flash-1017 \
  --prompt-data /root/datasets/image-sft/train.jsonl \
  --output-dir /root/runs/dsv41-vision
```

The launcher owns topology and optimizer defaults. The initial implementation
supports image SFT at CP1/BSHD/MBS1, with sequence parallelism at TP>1.
`--vision-trainable full` trains the whole visual tower; `projector` trains the
aligner and boundary embeddings; `none` gives a frozen-vision language control.
`--freeze-language --vision-trainable full` is the causal vision-only arm.
Do not wrap the frozen language forward in `no_grad`: visual gradients still
need to pass through it. All visual tensors stay in the module tree, including
in the frozen arm, so native checkpoints store a consistent visual state.

The HF bridge and raw HF exporter map `visual.*` to native `vision.*`,
`aligner.*` and boundary keys. Export must replace those trained tensors, not
overlay them with historical frozen source shards. Engram source assets require
their existing native preservation path.

For qualification runs, `audit.before_step` can be registered through Miles'
existing before-train-step hook with `DSV41_AUDIT_DIR` set to a unique attempt
directory. `checkpoint_audit.install_final_save_check()` can be installed by
the custom Megatron initialization hook to test the final synchronous save.
It corrupts in-memory visual parameters and language probes, reloads the saved
model without optimizer/RNG state, and checks complete visual hashes. This is
deliberately a model-only reload test, not a training-resume claim. Do not use
it during a run that continues after its final save.

## Acceptance Procedure

Follow the corrected experiment in
[Miles issue #1](https://github.com/leviking98z-rgb/miles/issues/1), not its
withdrawn GEO3K zero-shot result. Reuse its pinned geometric scene specifications
and exact integer labels, with disjoint train/held-out images and no oracle
information in model inputs. A different DeepSeek base is a new experiment and
needs its own baseline; the archived GLM scores do not transfer.

1. Compare native image preprocessing and visual forward against the pinned
   DeepSeek reference, then test image swaps/removal. Confirm real image inputs
   reach both training and evaluation; a displayed placeholder is not evidence.
2. Audit the optimizer parameter inventory before the first update. Verify
   finite nonzero gradients and actual parameter changes for ViT, aligner,
   boundary embeddings and unfrozen language/router weights. Record the
   explicit immutable/selector exclusions above; weight decay alone is not a
   gradient witness. TP visual replicas must agree after gradient reduction.
3. Save, perturb and reload the trained visual and language parameters, and
   verify exported native visual tensors. Record model-only reload separately
   from optimizer/data-cursor resume; do not conflate the two.
4. Evaluate the original baseline and only the final trained checkpoint on the
   same held-out images, greedy decoding and token budget, without checkpoint
   selection on validation. Include a vision-only arm with unchanged language
   hashes to isolate the visual contribution.
5. Report paired wrong-to-right/right-to-wrong changes, answer validity, macro
   accuracy and the majority-answer baseline, not training loss alone. Retain
   per-sample predictions, image/label hashes, source commit and run manifests.

The focused CPU contracts are in
`tests/fast/backends/megatron_utils/test_dsv41_vision.py`: complete tensor
inventory, spatial ordering, real visual weight updates, recompute parity,
TP-slice gradient sums, image-mask offsets and native export names. These tests
are not an end-to-end full-model or visual-quality qualification.
