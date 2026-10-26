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

The saved visual state was reconstructed through the native checkpoint reader
and HF name converter, exported as 266 native-name safetensors, and checked
against the complete trained visual hash and an exact file readback. Native
visual RMSNorm weights are FP32 at runtime; preserving that dtype is necessary
for bitwise export even though the original source shard stores them as BF16.
The complete language-plus-vision native HF export also completed: 96,085 keys
and 510,286,156,120 tensor bytes. All 266 visual tensors in that export exactly
match the independent trained visual export, including the FP32 norms. The
subsequent inference load of this complete export is not yet qualified.

The earlier joint smoke and model-only reload checks used a pinned official
Miles runtime with the feature modules overlaid. Their provenance is distinct
from the completed exact-fork vision-only and joint runs described below.
Training resume, complete jointly updated native HF inference reload and
joint-training held-out improvement remain unqualified. CPU tests alone do
not establish those results. The native visual forward and image
preprocessing were also checked bitwise against the pinned DeepSeek reference
on H20.

The DeepSeek geometric baseline completed all 256 held-out scenes and the
image-swap/removal controls with the archived exact scorer, greedy 4096 new
tokens and context 8192. Strict correct counts are 51/256 with the real image,
11/256 with a swapped image and 19/256 without an image. The real-image arm
has 203 invalid final answers; always answering `1` scores 97/256. These are
baseline results. The final vision-only comparison is recorded below;
joint-training evaluation remains pending.

For engineering acceptance, the completed vision-only run establishes actual
visual gradient flow, parameter updates, export/inference loading and an
observed held-out strict-score increase. This is the requested acceptance scope;
additional regression-case analysis and causal-isolation experiments are not
required for it. It does not remove the inference-control limitation below.

## Completed Exact-Fork Joint Run

`joint-geometric-fork-v9` completed the same fixed 32-update, 1024-image recipe
from the original base at fork `98ec6854`, with language training enabled.
All 6144 rank/update records passed finite-gradient/weight and successful-step
checks. All 266 visual tensors have nonzero gradient and weight-change witnesses;
the 32 visual replicas agree at every update. Their final complete hash is
`925382e6d008d8028fce2ec7347b514d1f37a04ae45cc24173f99444ffed9111`.

Of 576 sampled language parameter probes, 552 show changes over the run:
192 router probes, 192 attention probes and 168 expert probes. The remaining
24 expert probes show no change at their sampled positions; this is not a
complete language-tensor delta audit. All 192 ranks passed the final model-only
corruption/reload check at iteration 31, including full visual hash equality.
The checkpoint has 192 native DCP shards totaling 1,104,937,017,684 bytes.
The independent receipt is `data/joint-geometric-fork-v9-proof.json` under the
run project. No joint held-out score or optimizer-resume qualification is
claimed from these training and reload checks.

## Completed Exact-Fork Vision-Only Run

`vision-control-fork-v9` completed 32 updates of batch 32 on all 40 language
layers at fork commit `98ec6854`, with TP8/PP6/DP4/EP32 and CP1. The fixed recipe
uses 1024 exact-label training images, LR `1e-6`, zero weight decay and
seed/shuffle seed 42. All 192 rank initialization receipts resolve loaded Miles
and model modules to that committed fork, not the earlier overlay runtime.

All 6144 rank/update receipts passed. Every rank's complete language-parameter
hash remained unchanged before/after and across the 32 updates. The 32 visual
replicas agree after every update, and all 266 visual tensors have finite
reduced-gradient witnesses and actual weight changes over the run. The 160
pipeline ranks without visual parameters also completed their empty-optimizer
steps successfully. First/last batch losses are 12.782811/10.013475; they are
different batches and do not establish a held-out accuracy improvement.

The final nonresumable visual snapshot exactly matches the audited parameter
hash `01c4097607d97f6be26c84624aa62252475ecd1a9236625cbc90d8606aa83120`.
Its safetensors file SHA256 is
`bc9a9928154de983eac329119eddb4f5609ccc7ab8c88cf4fcf6d524a60e4a7e`.
The duplicate-free native HF staging path removes historical visual keys from
mixed source shards, retains byte-identical language weights, and preserves
the trained FP32 visual norms. All 24 inference replicas loaded this staged
model and completed the same-source held-out image/swap/removal evaluation.
This inference load qualifies the visual-only overlay,
not the separate complete HF export of jointly updated language weights.

### Final Vision-Only Evaluation

All 768 baseline and 768 final responses completed, with all 24 replicas in
each arm exiting successfully. The unchanged archived exact scorer gives
51/256 (19.921875%) before training and 62/256 (24.21875%) after training,
a gain of 4.296875 percentage points. Paired transitions are 31 wrong-to-right
and 20 right-to-wrong. Answer-macro accuracy rises from 19.300342% to
24.928592%. Final control counts are 16/256 for swapped images and 20/256
without images, versus 11/256 and 19/256 at baseline.

This is a modest strict-score gain, not evidence of a broad recognition or
benchmark improvement: invalid final answers remain 193/256, down from
203/256. Of the net 11 additional correct responses, 10 come from final-answer
validity transitions; the 32 examples valid in both outputs improve from
31 to 32 correct. Always answering `1` scores 97/256 (37.890625%), above
both arms. The run therefore supports the audited vision-training/export/eval
path and a measured task-specific strict-score change, not general accuracy,
multiple-seed reproducibility, or training resume.

The validity decomposition is a scoring identity, not a causal attribution to
format learning. Of the 30 invalid-to-correct transitions, 15 baseline answers
exhausted 4096 tokens without closing thinking and 15 stopped with a noninteger
first final line. Conversely, 19 correct baseline answers became unfinished
thinking and one became a noninteger final line. Total unfinished-thinking
counts actually rise from 140 to 165. The three supervised target tokens include
the thinking-end token, the integer and EOS, so visual conditioning can affect
both answer content and the frozen decoder's termination behavior.

A further inference-control gap remains: all 256 no-image response strings
differ between runs, including 147 extracted-answer differences, despite the
byte-identical language state. The existing mixed dynamic-batch evaluations
therefore do not establish deterministic inference parity. Numerical/batching
variation must be isolated before attributing the entire score difference to
changed visual representations; the small net no-image score difference does
not resolve this per-example drift.

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
