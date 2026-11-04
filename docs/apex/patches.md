# 当前有效补丁清单（apex 主线）

本文件是 BRANCH-PR 方案第 5.3.2 节第 6 条要求的清单：**下一次换上游基线重建候选时，只按本清单重放，不重放 `U..apex/main` 里的全部历史提交。** 每一项写来源、基线、在当前候选里的提交、状态。历史上被取代的提交列在最后一节，重建时不要重放。

- `apex/main` = **`09a5adcf`**（2026-11-04 09:1x UTC 快进），加上之后的文档提交。`09a5adcf` = 候选 4 = Qwen3.8-V1 正式 run 用的 miles（`apex/pr/qwen38-v1-final-lock2`）。上游基线 U = radixark/miles main **`c62ea5d8c2851f63a21fcb42296fdaa6910bce69`**（没有变）。
  - 历史：候选 2 `bd622ca7`（2026-10-31 23:4x UTC 首次建立，= `apex/qwen38-text-sft-rebase`）→ 候选 3 `28dc6e20`（2026-11-01 05:5x UTC）→ 候选 4 `09a5adcf`。
  - 候选 4 的验收记录：协调者 scratchpad `q38/r3/req/V1-09a5adcf-gate.accept.md`。全文抄在 apex-workspace `miles/progress/2026-11/` 的 11-04 记录里。
- 本清单覆盖 `c62ea5d8..` 候选 2 的全部 117 个提交（L1–L3）、候选 3 的 3 个功能分支（L2b）和候选 4 的 90 个非 merge 提交（L2c），加上之后的文档 / 脚本提交。
- 「提交」一列：L1–L3 写**候选 2 里的 SHA**，L2b 写候选 3 里的 SHA，L2c 写 `09a5adcf` 里的 SHA（都是短 SHA）。`apex/pr/*` 分支的 SHA 写在「来源」或「分支」列。
- miles 之外、V1 运行也依赖的补丁（训练仓库的 Megatron 补丁、fla 补丁、运行时 hook）列在「L4」。它们不在 miles 树里。
- 状态：`有效` = 树里的实现来自这一项；`上游 open` = 上游 PR 还没合入，合入后按第 5.3.3 节判断能否去掉。

## L0：上游基线

| 项 | 值 |
|---|---|
| radixark/miles main | `c62ea5d8c2851f63a21fcb42296fdaa6910bce69`（2026-10-31） |
| 已经从基线获得、不再单独保留的上游修复 | #4530（scheduler 不重复 step）、#3899（`finalize=` 注入）、#4562（`load_auto_bridge`） |

## L1：上游 open PR 的提交（固定 head，`cherry-pick -x`，保留原作者）

| 上游 PR | 固定 head | 选入的提交 → 候选 2 | 本地适配 / 补充（候选 2） | 状态 |
|---|---|---|---|---|
| #3879（只取冻结子功能） | `9ed6eacf` | `1f5639ff` → `123a38a7` | `2f275dcc`：Qwen3.8 indexer 用共享 freeze helper；保留 provider 全参数冻结断言 | 上游 open |
| #4133 | `02cda78b` | `02cda78b` → `a3fbc478`（**混有本地适配**：tiled scorer 直接喂 block id、mRoPE、thd / sbhd 分开） | `91f46c43`：monitor 读 block id；TP8 修复（M-030）：`0bd5c396`（改 dense query tile，**改错了地方，已被下一个提交撤回**）+ `9ba4f70a`（`rebase/fix-qsa-fill` 上 `3c47f395` + `421c3653` 的净改动：`_fill_kernel`、`_mark_kernel` 不按 `stride_b` 特化，kernel 体与 #4133 相同）。`f5439d3a`：本地 int64 行偏移测试（来自 `a3fbc478`）的源码解析也认 `@triton.jit(...)`。重建时只重放 `1193d35b` 对 `69c8d5c0` 的改动（装饰器 + 3 个测试文件）。`apex/pr/qsa-block-index-fill-bits` = `99aa2abe`（以 #4133 head 为基，GPU 已确认） | 上游 open |
| #2796 | `f01c353a` | `f01c353a` → `777aff66` | 无（生产代码与我们旧 `09c9393a` 相同，只剩这一份） | 上游 open |
| #4471 | `b45e6532` | `b45e6532` → `4765b582` | `e6cca277`：Qwen3.8 模型级 CP 防呆的错误文字 | 上游 open |
| #3964（观察器结构） | `e320b1a0` | `0d9e23e2` 手工移植 → `3bff1fa1`（作者 Junwei Zhou，`Ported-from` + `Co-authored-by: rasoolfa`） | `ee1d283a`：严格契约测试（训练过的存档缺 scheduler 状态就拒绝，与上游 #4530 契约相反） | 上游 open |
| #3691 | `f4d93f1d` | `d9ffa024` → `5bbb3395` | `aaa82af2`：rank0 覆盖检查的异常先广播再统一抛出 | 上游 open |
| #3925（只取转换器部分） | `a03df3bd` | `a03df3bd` → `bd622ca7`（作者 Yu Shi Jie） | 冲突处理：保留上游后加的 `hf_checkpoint` 覆盖、Ray 工具的 `argparse.Namespace` 复制；没取：删 `tools/convert_to_hf.py`、删 `dist_ckpt_save_pre_mcore_014`、#3924（M-031） | 上游 open |

## L2：我们的修复（`apex/pr/<topic>`，以上游 main 为基，push 在 apexin-ai/miles，不开上游 PR）

每项在候选 2 里是「pick + adapter」两个提交（adapter 删掉旧实现，只留一份）。重建时：pick 分支的提交，再按 adapter 消息里的 `Kept:` 把本地补充接上。

| M | 分支（head） | 候选 2：pick / adapter | adapter 保留的本地补充 |
|---|---|---|---|
| M-020 | `apex/pr/converter-skip-layout-version`（`c54f5535`） | `1e18e612` / — | 无 |
| M-001 | `apex/pr/ple-sp-input-grad`（`b709d6a7`） | `e77a71eb` / `ee2276bf` | GPU 测试 `tests/fast-gpu/test_qwen3_8_next_ple_sp_grads.py` |
| M-017 | `apex/pr/qwen38-reject-cp`（`c8e4c4c9`） | `5c9764ba` / `7bf5fa6b` | 错误文字「with or without --allgather-cp」；CP2 / CP4 × flag 开关 8 例测试 |
| M-002 | `apex/pr/qwen38-gdn-tp-avg`（`a81edead`） | `a9da4a0b` / `e439d0aa` | TP 副本检查器（见 L3）与 PR 的 TP2 测试互相印证的 3 个断言 |
| M-011、M-012 | `apex/pr/skip-unsupervised-step`（`8cfb1301`；它的父提交 `fc03c573` 是 #2796 的另一份移植，**不 pick**） | `835e5366` / `097ce2f9` | `finalize=` 注入、`TestFinalizeInjection`、torch-native 路径 |
| M-009 | `apex/pr/qwen38-indexer-tiled-scorer`（`f99ad296`） | `9a09cd7e` / `bbfa2c92` | #4133 接线（thd block id、sbhd token 选择）、冻结、mRoPE、monitor 接口 |
| M-016 | `apex/pr/hf-export-base-dtype`（`a1ac84f1`） | `a4390aad` / `fb4d8ed0` | 无 |
| M-005 | `apex/pr/hf-export-fused-experts`（`9dc7d990`） | `7ac361f5` / `60cb1045` | 无 |
| M-006 | `apex/pr/hf-export-complete`（`c56442ac`） | `25d983e2` / `84f9edb6` | Qwen3.8 `mtp` 组 `only_if_absent=False`（**严格规则，协调者 14:49 UTC 定**：本版不训 MTP，导出里出现 `mtp.*` 就拒绝，一律从 base 取；以后做 MTP 联训再按配置放开）；`verify_export` 命令行；`check_export` |
| M-015 | `apex/pr/ple-provenance`（`1d59fa8d`） | `c9bf8f00` / `ce928b7c` | `check_export`、`MILES_PLE_MANIFEST`、`check_tp_replicas` 在 `on_load` 之前 |

合并顺序要求：M-016、M-005 在 M-006 之前（否则 Qwen3.8 raw 导出会失败）。

## L2b：我们的功能分支（`apex/feat/<topic>`，候选 3 起）

| 分支（head） | 候选 3 提交 | 说明 |
|---|---|---|
| `apex/feat/qwen38-flops`（`25160fd1`，FIX-A） | `22916f9c`（`cherry-pick -x`） | MFU 把 Qwen3.8 的 `indexed_attention` 层按 QSA 计 |
| `apex/feat/qwen38-vision-text-fallback`（`e78215b0`，FEAT-vision） | `5ac62d95`、`512fda15`、`f2916b4e`（`cherry-pick -x`） | 文本 microbatch 跳过视觉塔；text / image / mixed 数据模式；`images=None` 文本行。helm 快照随它更新 |
| `apex/feat/hf-export-shard-names`（`5a5611df`，POD） | `c947280b`（**移植**，`Ported-from` 3 个提交） | raw 导出所有分片命名 `model-i-of-N`。分支写在候选 1 的 `hf_passthrough` 上；候选 2 已是 M-006 PR 的实现，所以移植：调用放在 `publisher.complete_model` 之后（writer、所有 collective 之后，避开 CRV-FIX-026 的错序）；测试移到 `test_hf_export_complete.py`；去掉 hardlink 用例（候选 2 没有 `MILES_HF_PASSTHROUGH=hardlink`）。重建时按 `c947280b` 重放，不 pick 分支 |

## L2c：候选 4 = Qwen3.8-V1 正式 run（`28dc6e20..09a5adcf`，90 个非 merge 提交，候选 4 起）

- 来源：集成分支 `apex/pr/qwen38-v1-final-lock2` = `09a5adcf`。它是叠起来的候选链（`apex/cand-vl2` → `apex/cand-qwen38-v1` `b5c72b4d` → HC → F5 → F8 → CP → final-lock），不是一题一个分支。
- 90 个提交都能在下表的某个 `apex/pr/*` 或 `apex/feat/*` 分支上找到（按 SHA 或 patch-id，0 个只在候选分支上）。表里「分支（head）」写的是一题一个的分支。没有单题分支的写叠链分支，并注明。
- 唯一的 merge 提交 `bc3feecd`（把 `ae27b11b` 并到 `8325e8bd`）不用重放。
- 「V1」列：Qwen3.8-V1 attempt `20261103-155422` 的 argv 里是否打开（`launch.json`，2026-11-04 核对）。「代码在，不生效」= 树里有，V1 的开关没开。
- 重建时：按下表逐题 pick 「提交」列，不 pick 叠链分支的整条历史。新开关默认都保持旧行为（P5 `--qwen38-ple-deterministic-conv-grad` 例外，默认开）；V1 的值由训练仓库 run-config 显式写。

### L2c-1 视觉

| 题目 | M | 分支（head） | 候选 4 提交 | V1 |
|---|---|---|---|---|
| 预分词图像行、THD packing、rope fusion | 无（功能） | `apex/feat/qwen38-vl-pretok`（`dd669f37`）；单题：`apex/pr/qwen38-vision-pretok`（`8ccd24e8`）、`apex/pr/qwen38-vision-packing`（`9ba16139`）、`apex/pr/qwen38-vision-rope-fusion`（`f912e2ff`） | `a4a7aa66`、`24a26ecc`、`1f469ae4`、`dd669f37` | 生效（`--qwen38-vision-data mixed`） |
| 视觉审计不算 `model_companion`、按实际图像判梯度、读回检查不改整数张量 | M-034 | `apex/pr/qwen38-vision-audit`（`50378025`；代码 = `47ced2df`） | `52faf403`（红测）、`2c85ab10`、`262e4ad2`（修复）、`fca5d952`（black）、`5e52a932`（配方 `--num-layers 4`）、`c3cc9759`（配方 TP8 词表不补齐）；`42e280a5`（文档：joint160 非零 TP 读回失败 = M-002） | 生效（`--qwen38-audit-dir`） |
| ViT lr / wd 倍率并入 Megatron param groups；`--qwen38-vision-load-from-scope` | 无（功能，F7） | `apex/pr/qwen38-vision-param-groups`（`00a6a8ec`）、`-v2`、`-on-28dc`。候选里的版本经 `apex/pr/qwen38-vision-lazy` 进来，生产代码与 `9a0c416d` 系相同，只差 helm 快照的冲突上下文 | `24b1f4fa`（红测）、`9c8a11b3`、`8abe3546`、`2a8453f1`、`55056d35`（红测）、`4ca9bae1`、`3163b4bf`、`fd091291`（black）、`f3678acb`（红测）、`01c052cf`、`af0e7996`（black） | 生效（`--qwen38-vit-lr-mult 0.2`）。**改优化器 param group 布局** |
| 没有图像的 step 不更新视觉塔（lazy Adam） | M-036 | `apex/pr/qwen38-vision-lazy`（`af348da5`）；`apex/pr/qwen38-vision-lazy-v2` | `94cfd0e7`（红测）、`bb9c8ac1`、`7dffa0f0`、`d4cce86d`（black）、`af348da5` | 生效（`--qwen38-vision-lazy vision`） |
| HDO：offload fraction < 1 时的 lazy vision；续训保留 GPU 子优化器 group step | M-023 | `apex/pr/hdo-offload-fraction-lazy-vision`（`812ba035`） | `384e0c91` | 代码在，不生效（V1 `--optimizer-offload-fraction 1.0`，不安装） |
| 测试修正（rope 表测试、helm 快照视觉键顺序） | 无 | 同上一行的叠链分支 | `d4052b36`、`202e0fd1` | 只有测试 |

### L2c-2 PLE、GDN、QSA indexer

| 题目 | M | 分支（head） | 候选 4 提交 | V1 |
|---|---|---|---|---|
| PLE gate conv 反向逐个释放 fp32 [T, W] 缓冲（P1） | M-035 | `apex/pr/ple-gate-conv-bwd-memory`（`59679a64`） | `d5f84cb2` | 生效 |
| SP 下 PLE 只算本 rank 切片加 conv halo（P4） | M-035 | `apex/pr/ple-sp-local-slice`（`0b6f462e`）、`apex/pr/ple-sp-local-slice-on-p1`（`657408b4`） | `96d772f7`、`2558d758`（black）、`49d1e398`（TP4 测试） | 生效 |
| PLE 卷积权重梯度确定性累加（P5，默认开） | M-038 | `apex/pr/ple-deterministic-conv-grad`（`06fda527`） | `fa5b2098`、`4d771839`、`06fda527`（测试） | 生效（`--qwen38-ple-deterministic-conv-grad`） |
| GDN 按 TP 头切分 | M-002 | `apex/pr/qwen38-next-gdn-head-sharded`（`8066c666`；实现 `6caa6283` 与候选里的 `5d28ef63` 只差 helm 快照） | `5d28ef63`、`4acb3e00`（CPU 测试）、`7031ee7d`（GPU 手动检查） | 生效（`--qwen38-gdn-head-sharded`）。**改存档布局**：复制 / 切分两种存档不能互读 |
| QSA indexer bf16 打分、按 TP 切行、host `cu_seqlens` 规划 tile | M-009 | `apex/pr/qwen38-indexer-fast-score`（`130b9f95`；`f8f7337f` / `130b9f95` = 候选里的 `c57b9a4d` / `2776fe18`） | `c57b9a4d`、`2776fe18` | 生效（`--qwen38-indexer-tp-shard`、`--qwen38-indexer-score-dtype bf16`） |
| HC gate 链 matmul 精度开关 | 无（性能） | `apex/pr/qwen38-hc-tensor-core`（`75c9bbf0`，叠链：题目 = 最上面 3 个提交） | `fb7cdce8`、`8b729f9f`、`75c9bbf0`（测试） | 生效（`--qwen38-hc-matmul-precision tf32`） |
| fla 内核钉住一个 config、NB 传常数（causal_conv1d、layer_norm_gated、l2norm） | M-027 | `apex/pr/qwen38-cp-perf`（`747174a4`）、`apex/pr/qwen38-cp-perf2`（`b481f732`），叠链 | `3757c513`、`09a5adcf` | 生效（`--qwen38-fla-conv-pinned`） |

### L2c-3 训练循环、数据、续训

| 题目 | M | 分支（head） | 候选 4 提交 | V1 |
|---|---|---|---|---|
| `--prefetch-rollout N`：当前 rollout 训练时预取下一批数据 | 无（性能） | `apex/pr/train-prefetch-rollout`（`3e4402e3`，与 `ae27b11b` 只差 helm 快照） | `ae27b11b`（merge `bc3feecd` 不重放） | 生效（`--prefetch-rollout 1`） |
| `--no-clear-memory-every-step`：纯训练 step 之间保留 CUDA 缓存 | 无（性能） | `apex/pr/train-clear-memory-every-step`（`67f1fa3a`） | `c29f2f3b`、`6e5fd4ed`、`46719ea6`（helm 快照） | 生效 |
| `--save-rollout-ids`：在固定的 rollout id 存档（配合预取） | 无 | `apex/pr/qwen38-early-save-prefetch`（`c2461d10`，生产代码 = `d58d42b5`）、`apex/pr/snapshot-save-rollout-ids`（`80b9a067`） | `2826c5b2`、`f83e0711`（black） | 生效（`--save-rollout-ids 2`） |
| `--qwen38-reload-check-iterations`：早期存档也做存档 → 读回 → 比较 | 无 | `apex/pr/qwen38-early-reload-check`（`0a1d8487`，生产代码 = `9d7da9f2`） | `64d9e16c`（红测）、`9d7da9f2` | 生效（`--qwen38-reload-check-iterations 2`） |
| `--start-rollout-id` 时读回已恢复 rollout 的数据状态，不是 `start_rollout_id - 1` 的（改上游 `miles/ray/placement_group.py`） | M-040 | `apex/pr/start-rollout-id-data-state`（`8dd82c89`） | `8325e8bd` | 代码在，不生效（V1 没传 `--start-rollout-id`） |
| executor 跳过每个 rollout 的 deepcopy | 无（性能） | `apex/pr/sync-sft-skip-snapshot-deepcopy`（`d9e1780c`） | `2e80c003` | 生效 |
| data_source 用结构复制代替 `copy.deepcopy` | 无（性能） | `apex/pr/data-source-shallow-copy`（`8452b35f`） | `90fd1cb8` | 生效 |
| `read_file` 用 orjson 解析 JSONL | 无（性能） | `apex/pr/read-file-orjson`（`1ee29279`） | `04445ae9` | 生效 |
| `--balance-data-lockstep-group G`：EP 组内同步跑 microbatch 的 DP rank 作为一个槽位均衡（F5） | 无（性能） | `apex/pr/balance-data-cost-model`（`42b02898`，叠链：题目 = 最上面 3 个提交） | `c8e8dc82`、`6beb7d2d`（black）、`42b02898`（测试） | 生效（`--balance-data-lockstep-group 2`） |
| 减少主机同步：loss、视觉输入、视觉塔长度 | 无（性能） | `apex/pr/qwen38-host-syncs-loss-vision-inputs-v1`（`241db50f`）、`apex/pr/qwen38-vision-tower-host-lengths-v1`（`bf0bd7f1`） | `e0851c08`、`1dc8b516`、`73dadd32`、`0944568d`、`f176c994`（black） | 生效 |
| 减少主机同步：PLE、get_batch | 无（性能） | `apex/pr/qwen38-host-syncs-ple-getbatch-v1`（`18c400b0`） | `82d88d4b`、`6eeb4ee7`、`07590876`、`8d683ef6`、`b8366eca`、`b5c72b4d`（测试） | 生效 |

### L2c-4 在树里、V1 不生效的路径

| 题目 | M | 分支（head） | 候选 4 提交 | V1 |
|---|---|---|---|---|
| F8：overlap-grad-reduce 下视觉 bucket 顺序、拒绝拆分塔模块的 bucket 布局 | 无（我们功能的初版缺陷；M-039 只引用 F8） | `apex/pr/overlap-grad-reduce-vision-order`（`7cfd6404`） | `ffab4e4b`、`646c0854`、`eca73a62`（测试）、`92ddc6fb`（black）、`5fb08142`（测试） | 不生效（V1 不开 overlap-grad-reduce） |
| F8：视觉 bucket group 延后到 `finish_grad_sync`、`ddp_trace`、`grad_ready_check`、`f8_guard` | 同上 | `apex/pr/qwen38-cp-vision-f8`（`57d7478c`，叠链） | `6f302893`、`69b8bb40`、`422ea01d`、`57d7478c` | 不生效（overlap 关；`Q38_F8_GUARD_FORCE` 没设） |
| allgather-CP：QSA、PLE、视觉塔（zigzag CP 仍拒绝） | M-017、M-018 | `apex/pr/qwen38-cp`（`d501b6be`）、`apex/pr/qwen38-cp-vision`（`8c8198fb`），叠链 | `d501b6be`、`1d216217`（红测）、`956bdf8e`（红测）、`68ec1c21`、`a15fa2dd`、`1e028e12`、`028dc8d2`、`8c8198fb` | 不生效（V1 CP1） |

### L2c-5 有分支、不在候选 4 的提交

| 分支（提交） | 情况 |
|---|---|
| `apex/pr/qsa-indexer-det-compress`（`e3a1c28b`） | indexer 压缩 key 不用 `index_add_`（M-041）。复审已过（2026-11-03），不在 V1。V2 候选 |
| `apex/pr/qwen38-f8-order-hash`（`869d959b`、`da2736ec`、`3e318b1b`） | F8 的 bucket 发出顺序检查。V1 不带 F8。V2 再定 |
| `apex/pr/qwen38-a2a-overlap`（`b08b7f1d`） | EP A2A overlap（E1）。测过：数值正确但更慢，**不采用** |
| `apex/pr/qwen38-cp-vision-tests`（3 个测试）、`apex/pr/pack-test-fixes`（`38ceb5e4`）、`apex/pr/qwen38-cp-vision-f8-perf`（`9b963a8a`，帮助文字）、`apex/pr/packing-conv-boundary-test`（`97239972`，基于旧集成分支） | 只有测试或文档。下一个候选一起合 |

## L4：V1 运行依赖、但不在 miles 树里的补丁（住在训练仓库）

这些改动 **lives in training repo, not miles**：在 apexin-ai/qwen38-flash-training（V1 运行 `e08db15`），由 `manifests/lock.json` 锁定，每次 Pod 启动时打上或安装。换 miles / Megatron / fla 版本或重建候选时，要一起核对。

### L4-1 Megatron 补丁（`patches/megatron/`，基于 radixark/Megatron-LM `fdf82356`，Pod 启动时 `git apply`，lock 写 sha256）

| 补丁 | 内容 | 上游 | M |
|---|---|---|---|
| `0001-fully-reshardable-bucket-padding-pr5077.patch` | fully_reshardable 存多 bucket buffer 时的偏移（取上游生产代码）。V1 一个 buffer 一个 bucket，实际不改行为 | NVIDIA #5077，2026-08-21 合入（`32a03b77`）；radixark fork `fdf82356` 没有 | M-043 |
| `0002-dp-reshardable-padding-only-shards-pr8455.patch` | dp_reshardable 存 / 读只含 bucket padding 的 DP 分片 | NVIDIA #8455 open draft（head `adbe8909`）；#6191 已合入（作对照） | M-013 |
| `0003-hc-mlp-norm-manager-reset.patch` | fused-HC MLP 路径用完清空 `mlp_norm_manager` | 无 PR | M-037 |
| `0004-hdo-pinned-async-copy-back.patch` | HDO 的 fp32 CPU 主参数放 pinned 内存，回拷 H2D 异步；主机写主参数前等 H2D 流（PERF F2，每步省约 1.6–1.85 s） | NVIDIA #4547（2026-08-21 合入，`faecf6b5`）已把 fp32 主参数放 pinned；没有 0004 的主机等待。radixark fork `fdf82356` 没有 #4547 | M-042 |

- 注意：infra 报告和 11-02 的 PERF 记录里有另一个「补丁 0004」，是 E1 A2A overlap 用的 Megatron 改动（HC `h_res=None` 的 `detach` / `record_stream`）。它没进 lock，和上表的 0004 不是同一个。

### L4-2 fla 补丁

| 补丁 | 内容 | M |
|---|---|---|
| `patches/fla/patch_conv_int64.py` | fla 0.5.2 `modules/conv/triton/kernels.py` 的 causal_conv1d 偏移改 int64（T × D > 2^31 时溢出）。Pod 启动时改镜像文件，lock 写改前 / 改后 sha256 | M-019 |

### L4-3 运行时 hook（训练仓库 `code/qwen38_hooks.py`，`custom_init` 安装）

前 3 项直接改 Megatron / miles 的行为，计划在 V2 前移进 miles（训练仓库只留开关）。

| hook | 改了什么 | V1 配置 | M |
|---|---|---|---|
| `install_short_mb_no_recompute`（`qwen38_hooks.py:747`） | 替换 Megatron `TransformerBlock.forward`：EP / TP / CP 组内最大长度 ≤ T 的 microbatch 跳过 full recompute | `schedule.json` `no_recompute_max_tokens 32768` | 无（待建，V2 前移进 miles） |
| `install_long_mb_empty_cache`（`qwen38_hooks.py:821`） | 再包一层 `TransformerBlock.forward`：长 microbatch 前 `memory_reserved()` 超阈值就 `empty_cache`（option C） | `empty_cache_long_mb_tokens 65536`、`empty_cache_reserved_gib 218` | 无（待建，同上；与 `--no-clear-memory-every-step` 放一起） |
| `install_gc_freeze`（`qwen38_hooks.py:906`） | 包 `miles.backends.megatron_utils.actor.train`：第一次 train 时 `gc.collect()` + `gc.freeze()` | argv `--manual-gc` | 无（待建，同上） |
| `step_timing`、`host_memory`、`gbs_migration`、milestone HF 导出、DDP 布局检查、视觉审计挂接；`code/qwen_pretokenized_source.py` 的视觉预处理线程池 | 观测、编排、插件点 | — | 不移（留在训练仓库） |

- 其他运行环境：cuBLAS 13.4.3.11 `LD_PRELOAD`（F1，lock `cublas`）和 `NCCL_ALGO`、`OMP/MKL_NUM_THREADS`、`RAY_object_manager_pull_timeout_ms` 等环境变量也在训练仓库，不在 miles。

## L3：只留在我们这边的提交

| 项 | 候选 2 提交 | 为什么不能上游 / 为什么保留 |
|---|---|---|
| 同事 PR #5（DeepSeek V4.1 1017 视觉 SFT） | `1a5e9645` … `6adcdea0`（11 个，作者 leviking98z-rgb） | 同事在 apexin-ai/miles 的功能 PR，由同事决定 |
| 同事 PR #6（Qwen3.8 视觉 SFT 与评测，含 M-003 `5dc6337f`、M-022 `1ab656ab`） | `d8e711b0` … `55331e08`（24 个） | 同上（用户 10:10 决定 M-003、M-022 放这里） |
| PR #5 / #6 迁到上游接口 | `f4d7c44f`、`cc6392ce`、`3eba2c3c`、`5b8100e2`、`ed8dd50d`、`4e33ffde`、`bf39a692`（mRoPE spec）、`d6cbe57a`（bridge 缓存按 stride，M-003 的后续） | 只适配上面的私有代码 |
| helm 快照 | `a1817ba2` | 快照含 9 个视觉插件字段（上游没有） |
| TP 副本逐字节检查器 | `dd98b69d`、`d027d14a`、`91f066f7`（加上 M-002 adapter `e439d0aa` 的断言） | 我们的验收诊断工具，不是上游 bug 修复 |
| indexer 抽样 recall monitor | `7bfa49fe`、`9cb4b7f6`、`91f46c43` | 实验观测工具 |
| fused log-probs 的 Qwen3.8 SFT 测试 | `4ba755df`、`8e88af53` | 只测我们的配置（M-025） |
| PLE SP 梯度的 GPU 测试 | `346051f4`、`b1038bef` | M-001 PR 不带它 |
| 冻结 indexer 测试的 Gloo fixture 清理 | `82869d19` | 我们的测试 |
| 集成文档与本清单 | `208cff48`、`775336f6`、`9b3e44bb`、`1166f214`、`dc809ea9`、`69c8d5c0`、`ff39424a` 及之后的文档提交 | apex 内部记录 |
| `apex/main` 准备脚本 | `docs/apex/promote-apex-main.sh`、`docs/apex/test-promote-apex-main.sh` | apex 内部流程。门禁凭严格的验收记录（`candidate:` / `status: PASSED`），见脚本开头 |

## 已被取代、重建时不要重放的提交

这些提交还在历史里，但它们的内容已被上面某一项取代（adapter 消息里的 `Replaces:`，或内容逐字相同）。

| 旧提交 | 内容 | 被谁取代 |
|---|---|---|
| `ce643b2c` | indexer 冻结（`requires_grad_(False)`） | L1 #3879 `123a38a7` + `2f275dcc` |
| `d79681d6`、`8966fc76`、`54873c5c` | 旧 tiled scorer 及其测试 | L2 M-009 `9a09cd7e` / `bbfa2c92` |
| `3f056976`、`c54a1c42` | 旧 bitmap kernel 与 int32 修复 | L1 #4133 `a3fbc478`（旧 kernel 已删） |
| `b11ba2c3` | PLE SP 梯度修复（与 PR 逐字相同） | L2 M-001 |
| `43c6785c`、`d1b7b0e7` | Qwen3.8 CP 防呆及测试修正（与 PR 相同） | L2 M-017 |
| `2a570539` | GDN TP 梯度平均（与 PR 逐字相同） | L2 M-002 |
| `09c9393a`、`0c35f0d7`、`86251703`、`4a3798ff` 的生产部分 | token 分母、全局 skip、expert-bias 告警、未 step 优化器补状态 | L1 #2796 `777aff66` + L2 M-011 / M-012（`4a3798ff` 的测试 `TestFinalizeInjection` 保留在 L2 M-011 的 `Kept`） |
| `1c2a5f0d`、`2d1f3e09` | 严格 scheduler 恢复的旧结构 | L1 #3964 `3bff1fa1` + `ee1d283a` |
| `692848d0`、`64fc789c`、`f8c2c337` | 旧 raw 导出补全 | L2 M-006 |
| `510216a2`、`c5a65dbd` | 旧专家打包 | L2 M-005 |
| `1e122742` | 旧 `A_log` dtype | L2 M-016 |
| `0713f313`、`390e562c` | 旧 PLE 来源绑定 | L2 M-015 |
| `afc60779` | black 格式化 | 重建时对新代码重新跑 black |
| `0bd5c396` 的 kernel 部分、`3c47f395`（只在 `rebase/fix-qsa-fill`） | TP8 的前两次错误修复（dense tile 逐行、buffer 16 字节对齐） | L1 #4133 的 `9ba4f70a`（M-030） |

## 维护

- 每次往候选里加修复（第 5.3.2 节第 2 条），同时在本文件加一行；被取代的旧提交移到最后一节。
- 每次 `apex/main` 前进（`docs/apex/promote-apex-main.sh`），同时改开头的 `apex/main` SHA 和历史一行。训练仓库 lock 换 Megatron / fla 补丁或运行时 hook 时，同时改 L4。
- 上游合入某个 L1 / L2 项后，按第 5.3.3 节逐个行为判断，只去掉被覆盖的部分，并在这里改状态。
