# apex/qwen38-text-sft 分支说明

本分支是 Qwen3.8-Flash-Next 文本 SFT 的集成分支，只在 `apexin-ai/miles` 上维护，不合回 radixark main。

2026-10-31 起，本分支 rebase 到 radixark main `c62ea5d8`，并换用上游已有的修复，推到 `apex/qwen38-text-sft-rebase` 复审（候选 1 = `69c8d5c0`；候选 2 在其上追加 10 个 `apex/pr/*`，见「候选 2」一节）。旧分支 `apex/qwen38-text-sft`（`712aebd8`，基线 `65feb4a6`）保留只读。门禁通过后，从同一个 SHA 建 `apex/main`（方案待用户确认）。下文「rebase 到 c62ea5d8」一节记录这次的变化；更早各节记录的是 `65feb4a6` 上的集成过程，其中的 SHA 指旧分支。

## 基线

| 项 | 值 |
|---|---|
| miles 基线 | `c62ea5d8c2851f63a21fcb42296fdaa6910bce69`（radixark main，2026-10-31 fetch；旧分支为 `65feb4a6`） |
| 镜像 | CPU 测试仍在 `docker.io/radixark/miles@sha256:6038e614942f40a4681dff0f02ab7a8239412247da084db678811be9f50fac88`（tag `dev-202610300556`，镜像内 miles 为 `65feb4a6`）里跑，用本分支检出覆盖。新镜像 `dev-202610310237`（多架构 digest `sha256:07eb3a22bae4dd6f1969e2128b205e290602abeeb181a6cf82500a719f3d1ba7`）待 SMOKE 做身份核查和 GPU 门禁后再切换 |
| Megatron | 镜像自带 `fdf8235667af026219195fd6a1cdf303a12db974`（radixark/Megatron-LM `miles-main`） |

运行时用本分支的检出目录覆盖镜像里的 `/root/miles`（`pip install -e . --no-deps` 或 PYTHONPATH）。记录各 rank 实际 import 的源码 SHA，不要只记镜像 SHA。

## 包含的 PR

| PR | 原 head | 内容 |
|---|---|---|
| apexin-ai/miles #5 | `38c5092628b779bf6cda6966f4797d85655d898a` | DeepSeek V4.1 1017 图像 SFT 路径与检查 hook，11 个提交 |
| apexin-ai/miles #6 | `df26f78bfe6eb6bdf09e694587207deeabd621d1` | Qwen3.8-Next 原生图像 SFT、评测、mbridge 修复，24 个提交 |

35 个提交按原顺序搬到基线上，保留原作者和作者时间。每个提交的 message 末尾有 `(cherry picked from commit <原 SHA>)`，可据此对照原 PR。

3 个原提交有冲突，共 6 个文件。冲突按上游 typed config 解决：

| 原提交 | 冲突文件 | 处理 |
|---|---|---|
| `12f7e128` | `miles/backends/megatron_utils/initialize.py` | 保留上游 `args.backend` 写法。别名注册见 `27e1c93b` |
| `12f7e128` | `miles_plugins/models/deepseek_v4/arguments.py` | 删除 PR 的 `add_dsv4_arguments`，三个 `dsv41_*` 字段改进 `Dsv4MegatronPluginsConfig` |
| `98ec6854` | `miles/utils/ray_utils.py`、`tests/fast/utils/test_ray_utils.py` | 用上游实现。上游 #3640 已有同样的 Ray head GCS 查询 |
| `8ab157d7` | `miles/backends/megatron_utils/megatron_to_hf/qwen3_8_next.py` | 保留上游签名，加 PR 的视觉前缀两行 |
| `8ab157d7` | `miles/utils/arguments.py` | 用上游版本。Qwen 视觉参数改进 `Qwen38VisionMegatronPluginsConfig`，在 `miles/utils/args/runtime.py` 六处注册 |

## rebase 到 c62ea5d8（2026-10-31）

做法：先把旧分支 82 个线性提交原样重放到 `c62ea5d8`（试 rebase，`dc809ea9`），再按 REBASE-plan 把上游已有的修复换进来。上游提交用 `cherry-pick -x` 保留原作者；每个引入或适配提交的消息里写 `Upstream-PR`、`Upstream-Head`、`Upstream-Commit`、`Upstream-Scope`、`Local-Adaptation`。

试 rebase 的 3 处冲突：

| 旧提交 | 冲突 | 处理 |
|---|---|---|
| `49323ff5` | helm 快照（生成文件） | 先取上游版本，最后按字段重录（`a1817ba2`） |
| `09c9393a`（原 `b4a21e5d`） | loss-sentinel 与上游 #3899 的 `finalize=` 注入 | 合成 `(model, num_tokens=None, *args, finalize=finalize_model_grads, **kwargs)`，调用 `finalize(model, num_tokens, ...)` |
| `1c2a5f0d`（原 `a833a1a0`） | scheduler 恢复与上游 #4530 | 保留我们的观察器和严格拒绝，去掉上游推断式 `scheduler_restored`；后由 `3bff1fa1` 换成 #3964 的观察器结构 |

在 `dc809ea9` 之上的提交：

| 提交 | 作者 | 内容 | 上游来源 |
|---|---|---|---|
| `3bff1fa1` | Junwei Zhou（`Co-authored-by` rasoolfa） | 唯一的 scheduler 恢复观察器和唯一决策点：已恢复不 step，训练过的存档缺 scheduler 状态就拒绝，其余 step(0)；绝对恢复；只在原 `load_state_dict` 成功后记录 | #3964 `0d9e23e2` 手工移植（cherry-pick 不适用） |
| `ee1d283a` | Junwei Zhou | 观察器的严格契约测试；改写与严格拒绝相反的上游测试 | 本地 |
| `5bbb3395` | Palanieppan Muthaia | bridge HF 导出缺少源 checkpoint 张量时拒绝 | #3691 `d9ffa024` |
| `aaa82af2` | Junwei Zhou | rank0 覆盖检查的异常先广播再统一抛出；双 rank Gloo 测试（修前 rank1 Gloo 超时） | 对 #3691 的补充 |
| `123a38a7` | Hejian Sang | indexer 共用 top-k 入口和冻结 helper | #3879 `1f5639ff`（只取冻结子功能） |
| `2f275dcc` | Junwei Zhou | Qwen3.8 indexer 用 `freeze_indexer_parameters` 冻结；保留 provider 全参数断言、tiled scorer、stable tie | #3879 的 Qwen 接线（取自 `1bc0c392`，未整提交引入） |
| `a3fbc478` | sha7doww | 新 packed QSA kernel：只计算选中的 block。删除旧 bitmap kernel 和旧 int32 修复；tiled scorer 直接把 block id 写进 `[T, K]` int32 缓冲交给 `build_qsa_block_index`；thd 检查 blocks，sbhd 保留 token 选择；保留 mRoPE | #4133 `02cda78b`，含本地适配 |
| `91f46c43` | Junwei Zhou | indexer monitor 从采样行的 block id 判断 membership，不再读 bitmap | 本地 |
| `777aff66` | Shikhar | token 分母不计全 mask 样本（与我们 `09c9393a` 的公式相同，只剩一份实现） | #2796 `f01c353a` |
| `4765b582` | Hejian Sang | `--allgather-cp` 遇到假设 zigzag 的 attention 层时拒绝 | #4471 `b45e6532` |
| `e6cca277` | Junwei Zhou | Qwen3.8 模型级 CP 防呆（任何 CP>1 都拒绝）与通用检查并存 | 对 #4471 的补充 |
| `4a3798ff` | Junwei Zhou | 测试改为像生产代码一样用 `finalize=` 注入，并检查 `num_tokens` 的转发 | 本地 |
| `a1817ba2` | Junwei Zhou | 重录 helm 快照：只加 9 个视觉插件字段；CI 镜像多出的 3 个 `router_args` 字段不录 | 本地 |

上游 #4530（scheduler）、#3899（`finalize=`）、#4562（`load_auto_bridge`）已在 `c62ea5d8` 里，经新基线获得。逐项去留和理由见各 topic 报告（`INT-rebase-qsa.md`、`INT-rebase-sched-export.md`、`INT-rebase-losscp.md`）。

## 候选 2：合入 10 个 `apex/pr/*`（2026-10-31）

做法按 BRANCH-PR 方案第 5.3.2 节：在候选 1（`69c8d5c0`）上，每个 `apex/pr/<topic>` 先 `cherry-pick -x`，紧跟一个 adapter 提交，删掉候选 1 里同一修复的旧实现，**每个修复只留一份实现**。adapter 提交消息里写 `Adapts`、`Replaces`、`Kept`、`Tests`。各分区的详细记录在 INT 的 `INT-cand2-model.md`、`INT-cand2-qsa.md`、`INT-cand2-export.md`。

| PR 分支 | M | pick | adapter | 说明 |
|---|---|---|---|---|
| `apex/pr/converter-skip-layout-version` | M-020 | `e3d587bf` | — | 候选 1 没有对应代码；两个 torch_dist→HF 转换器跳过 `weight_layout_version` 标记 |
| `apex/pr/ple-sp-input-grad` | M-001 | `05239a03` | `89334834`（无代码改动） | 生产代码与候选 1 相同；测试用 PR 版本，保留 GPU 测试 |
| `apex/pr/qwen38-reject-cp` | M-017 | `d6907702` | `38d2dc31` | 只留一个模型级 CP 防呆；恢复「与 `--allgather-cp` 无关」的错误文字和 8 例测试矩阵；#4471 通用检查不变 |
| `apex/pr/qwen38-gdn-tp-avg` | M-002 | `850c05f6` | `85e18e48` | `gdn.py` 与候选 1 相同；TP 副本检查器保留，并在 PR 的 TP2 测试里加 3 个断言 |
| `apex/pr/skip-unsupervised-step` | M-011、M-012 | `3a00e1b2` | `a451584a`（无代码改动） | PR 里自带的 #2796 移植没有 pick（与候选 1 的 `777aff66` 等价）；`finalize=` 注入和 torch-native 路径不变 |
| `apex/pr/qwen38-indexer-tiled-scorer` | M-009 | `549c6b93` | `466798d0` | 只留一个分块 scorer：PR 的分块规划和字节预算，输出 block id 给 #4133 的 `build_qsa_block_index`；保留冻结、mRoPE、稳定 tie、monitor 接口。`MILES_QWEN38_INDEXER_TILE_BYTES` 改为在 import 时读一次 |
| `apex/pr/hf-export-base-dtype` | M-016 | `a9867ea5` | `b8047f19` | `hf_base_dtype.py` 用 PR 版本 |
| `apex/pr/hf-export-fused-experts` | M-005 | `0ff2969f` | `e5f98684` | `hf_expert_packing.py` 用 PR 版本 |
| `apex/pr/hf-export-complete` | M-006 | `6c4a6a49` | `2a7a041b` | 补全逻辑用 PR 的「模型注册键组」；`verify_export` 命令行移植到新模块，名字、报告字段、退出码不变；去掉 `MILES_HF_PASSTHROUGH=hardlink` |
| `apex/pr/ple-provenance` | M-015 | `58fb417f` | `2ccd17e8` | 用 PR 的存档顺序（sidecar 早于 `save_checkpoint`，fsync，所有 rank 一起失败）；保留 `check_export` 和 `MILES_PLE_MANIFEST`；`check_tp_replicas` 仍在 `on_load` 之前 |

另有 `afc60779`：black `-l 119` 格式化 export adapter 新加的行（只有这些行）。

候选 2 的第一个提交是 TP8 修复 `0bd5c396`（M-030），它改错了地方；真正的修复是之后的 `9ba4f70a`（`rebase/fix-qsa-fill` 的 `421c3653`）。#4133 block index 的 `_fill_kernel` 在 query tile 21 / 32（每 rank 3 / 1 头即 TP8，以及 group 3 / 1）、K = 512 时报 `misaligned address`：`stride_b` = 512 被 Triton 按「能被 16 整除」特化，行加载变成 `ld.global.v4`，之后的 layout 转换用 `ldmatrix`，ptxas 没有发 warp 重同步，partial warp 出错（FIX-A 分析）。修法是 `_fill_kernel`、`_mark_kernel` 加 `do_not_specialize=["stride_b"]`；kernel 体与 #4133 相同。sm_100 交叉编译：LDSM 从 5–6 降到 0。GPU 确认见 INT 的 `req/INT-rebase-gpu.md`「追加 3」。

**有效补丁清单** 在 [patches.md](patches.md)：下一次换上游基线时只按它重放。`apex/main` 的建法和以后的更新用 [promote-apex-main.sh](promote-apex-main.sh)（默认 dry run；断言新 head 的树等于候选的树、祖先关系、远端没有移动；不强推）；它的玩具测试是 [test-promote-apex-main.sh](test-promote-apex-main.sh)。

注意：

- 单独检出 pick 提交（不带它的 adapter）可能 import 失败。把每对「pick + adapter」当成一个单元。
- **M-006 的未训练 MTP 规则：严格（协调者 2026-10-31 14:49 UTC 定）。** Qwen3.8 注册里 `mtp` 组 `only_if_absent=False`：本版不训 MTP，导出里出现 `mtp.*` 就拒绝，MTP 一律从 base 取。原因：未训练的 mtp 张量可能是没有加载的随机初始化，放行有风险。以后做 MTP 联训时再改成按配置放开。FIX-B-3 的期望键数不变。
- M-015：每次存档多一条无害的 Megatron 告警 `Overwriting old incomplete / corrupted checkpoint...`（`iter_N/` 在 save 之前已有 sidecar）。
- `test_ple_provenance.py` 的 fixture 比 PR 版本多 4 行（复位 `_run_manifest` 和 `MILES_PLE_MANIFEST`），应回补到 PR 分支。

## 迁移修复

每行一个提交。CPU 测试命令和环境见本文末尾。

| 提交 | 修复 | 测试 |
|---|---|---|
| `cc053832` | Qwen 视觉参数补齐 `qwen38_sft_final_answer_format`，删除旧 `miles_plugins/models/qwen3_8_next/arguments.py` | PR 测试集 |
| `a1632e83` | `export_full.py` 给 `convert_to_hf` 传 `args.backend`（上游 #3507 之后 converter 从 backend 读字段） | PR 测试集 |
| `9e4b9a67` | 两个视觉 model provider 改用框架传入的 `args=`（冻结的扁平视图），不再调用 `get_args()` | `tests/fast/backends/megatron_utils/test_vision_provider_contracts.py`，4 项。旧代码上 4 项失败（`args is not initialized`） |
| `27e1c93b` | `init()` 恢复上游的零参数 `register_hf_config_aliases()`。PR 改法使 25 个上游测试失败。transformers 5.19 下零参数注册正常 | 上游测试集与基线相同 |
| `deeacbe3` | 两个最终存档读回检查迁到新契约：init hook 收扁平视图；save 包装首参是 TrainerConfig；load 传 `args=`，load 字段经 `MegatronCkptLoadConfig.apply(args.backend)` 临时设置；扰动模型前先按真实签名 bind；失败时恢复被扰动的切片再抛错 | `tests/fast/backends/megatron_utils/test_vision_checkpoint_contracts.py`，5 项。旧代码上 Qwen 检查被静默跳过，DeepSeek hook 报 TypeError |
| `c0fe0b5c` | 清除 27 处 `config-attribute-access` lint 违规：已注册字段直接访问，非配置反射加具体豁免理由 | `tools/lint_config_attribute_access.py` 0 违规；PR 测试集 |
| `49323ff5` | 重录 `tests/snapshots/helm_backend/kubernetes_launch.txt`：只多 9 个新插件字段（默认值） | helm 快照测试通过 |
| `921d1dfb` | `get_qwen3_8_next_spec` 从 `args.backend` 读 `position_embedding_type`、`mrope_section`（PR #6 原写法在 TrainerConfig 上 AttributeError；CRV P1-INT-2） | `tests/fast/miles_plugins/models/test_backend_specs.py`：修前 2 failed，修后 8 passed |
| `a833a1a0` | sched-resume 与 gdn-tp-sync 在 `load_model_state` 的冲突：两边都保留，`check_tp_replicas` 在 `check_model_hashes` 之后 | `test_scheduler_resume.py`、`test_tp_replica_check.py` |
| `d14dbf9f` | cp-guard 测试与 `921d1dfb` 的语义冲突：测试里的 Megatron 字段放进 `backend` | `test_cp_guard.py` 4 passed |
| `3d5b8c28` | gdn-tp-sync 改写（`752ca5c8..0dda275b`）作为一个提交合入（集成分支不改写已合入提交） | `test_tp_replica_check.py`、`test_gdn_tp_sync.py` |
| `f75e2e2b` | ple-provenance 与 save 的 try/finally 冲突：`on_save` 在 try 内 `save_checkpoint` 之后，`on_load` 在 `check_tp_replicas` 之后 | `test_ple_provenance.py` 15 passed |
| `f1ba67ce` | `test_qsa_indexer_frozen.py` 的 fixture 清理自己创建的 Gloo 组（CRV P2-INT-3） | frozen→monitor 顺序：修前 1 failed，修后 10 passed |
| `2b5a2bc6` | raw HF 导出把逐专家 gate/up/down 打包成基座的 `mlp.experts.gate_up_proj [E,2I,H]`、`down_proj [E,H,I]`（CRV P1-INT-4；`hf_expert_packing.py`，接在 `SnapshotPublisher.write_model`） | `test_hf_expert_packing.py` 9 项（含真实 converter → save_hf_model → hf_passthrough 端到端）；修前端到端抛 `ExportIncomplete` |
| `8d70b962` | packer 初始化错误存入 write_error，rank 0 走完权重 gather 后再报（CRV P1-INT-5） | 双 rank Gloo：损坏 index 下两 rank 都完成 gather，下一 collective 正常；修前 Gloo 连接断开 |
| `611b3dad` | raw HF 导出把 GDN `linear_attn.A_log` 转回基座 dtype（只此白名单，只在落盘路径；CRV P1-INT-6；`hf_base_dtype.py`） | `test_hf_base_dtype.py` 4 项；修前端到端报 `A_log: F32[48] vs base BF16[48]` |

## apex/main（2026-10-31 23:4x UTC 建立）

- `apex/main` = **`bd622ca7ff377405c11be2c2758366ad74999414`**，由 `promote-apex-main.sh` 首次创建（不强推）。它是候选 2 `a6ac20ee` 加一个转换器提交：上游 open PR #3925（head `a03df3bd`）的 `cherry-pick -x`，只取转换器部分（无 `common.pt` 时从 DCP 的 `common_state` 读 args；M-031）。
- 门禁：第 1–4 组在 `a6ac20ee` 上通过并沿用（CX1 确认 `a6ac20ee..bd622ca7` 只改转换器工具、读档模块和测试）；M-020 转换器在 `bd622ca7` 上通过。验收记录 `r3/req/INT-cand2-gate.accept.md`（工作区，不在本仓库）。
- 本门禁不覆盖的开放项：M-015 真正的 `--async-save`；release 格式首次加载和 `kill -9` 资格项（P2，正式训练前做）；CRV-FIX-025、CRV-FIX-026（P1）；M-032、M-033。

## 候选 3：apex/main + 3 个功能分支（2026-11-01）

- 在 `61744769`（= `apex/main` `bd622ca7` + docs）上按 BRANCH-PR 方案第 5.3.2 节加 3 个 `apex/feat/*`，每个只留一份实现（`patches.md`「L2b」）：`qwen38-flops`、`qwen38-vision-text-fallback` 直接 `cherry-pick -x`；`hf-export-shard-names` 移植到候选 2 的导出实现上。
- 训练仓库同时要补丁 0002（`fix/megatron-6191` rebase 到 main，带上 `f2d6826`），由 REPO 做；两机回归（4k 存档 / 续训 / HF 导出 + 16k）由 SMOKE 做。通过后 `apex/main` 快进到候选 3。
- HF 导出的分片现在都叫 `model-i-of-N`：先导出的训练分片，再整块拷贝的基座分片，再 passthrough 分片；`passthrough.json` 的 `shard_renames` 记录原名。

## 发布 / 评测用的 HF 导出（协调者 2026-10-31 23:46 UTC）

- **发布和评测一律用训练里的在线 HF 导出，再跑 `verify_export`。不用离线转换器**（`tools/convert_torch_dist_to_hf.py`、`tools/convert_torch_dist_to_hf_ray.py`）。
- 原因（M-033）：离线转换器只转存档里的训练张量，对 Qwen3.8 输出逐专家拆开的键（4 层 6243 个），不补 base 的 495 个 passthrough 键（PLE 表、视觉、MTP），也不做 fused 专家打包。在线导出是 602 键（4 层）/ 1658 键（全模型），与基座的键集、dtype、shape 一致。
- 离线转换器只用于读档和排查。对 Qwen3.8 要加 `--model-name qwen3_8_next`（M-032：从 HF config 推出的 `qwen4expconfig` 不在分派里）。

## 已知限制

- 视觉路径只有 CPU 契约测试，没有 GPU 验证。最终存档读回检查是 model-only 读回，不证明续训（optimizer、scheduler、RNG、数据游标）。
- 读回检查失败时只恢复被人为扰动的切片再抛错，不是完整的内存回滚；失败后训练进程应退出。多 rank 下一个 rank 失败、其他 rank 停在 barrier 的情形没有测试（CRV P2-INT-1）。
- PR #6 没有修改 QSA indexer 的 `score_blocks`；由合入的 `apex/q38-indexer-tiled` 处理。旧的 `apex/q38-qsa-bitmap-packed`（T×NB bitmap）在 rebase 到 `c62ea5d8` 时由上游 #4133 的新 kernel（`a3fbc478`）取代，代码已删除。新 kernel 的 262K packed 全路径已在 GPU 上验证（候选 2 门禁）；TP8 头形状的 `misaligned address` 由 `9ba4f70a` 修复（M-030）。
- `--no-load-optim` 加训练过的存档：本分支拒绝续训（要求 `--finetune` 开新计划）；上游 #4530 按 `iteration*GBS` 重建。两者契约相反，本分支的上游测试已按严格拒绝改写。
- `qwen38_vision_trainable` 默认值是 PR 原来的 `"full"`。文本 SFT 不用视觉 provider，所以不受影响。用视觉 provider 时要显式设置。
- PR #5 的 DeepSeek/PP 路径没有 GPU 资格验证。

## 功能分支合入

功能分支名为 `apex/q38-<topic>`，从本分支切出。集成负责人把它 rebase 到本分支顶端后快进合入，保持线性历史。每次合入后跑下面的 CPU 测试。合入记录写在下表。

| 合入 | 功能分支 | 新 HEAD | CPU 测试 |
|---|---|---|---|
| 1 | `921d1dfb`（INT）+ `apex/q38-gdn-tp-sync`（FIX-B `a07b184c`、`752ca5c8`） | `595134d1` | 失败集合与基线相同；自带测试 35 passed |
| 2–3 | FIX-A：`cp-guard`、`indexer-freeze`、`indexer-tiled`、`indexer-monitor`、`qsa-bitmap-packed`、`ple-sp-boundary`、`fused-logprobs`、`loss-sentinel`；FIX-B：`sched-resume`（`b054c85f`）；gdn 改写差异；cp-guard 测试修复 | `3d5b8c28` | 失败集合与基线相同；新测试合集 179 passed；逐提交自带测试全部通过 |
| 4 | FIX-B：`sched-resume`（`06663c28`）、`export-complete`、`ple-provenance`、`mbridge-cache`；FIX-A：`qsa-bitmap-packed`（`90787901`）；fixture 修复 | `f1ba67ce` | 失败集合与基线相同（在 `34b68a3f` 跑全集，`f1ba67ce` 只改一个测试文件，单独验证）；新测试合集 85 passed |
| 5 | INT：`2b5a2bc6`（专家 packed 导出） | `2b5a2bc6` | 失败集合与基线相同；`weight_update/` 等自带测试 494 passed |
| 6 | INT：`8d70b962`；FIX-B：`apex/q38-fixb-crv`（CRV-FIX-002、014、015、018、016） | `25cefca9` | 失败集合与基线相同；新测试合集 548 passed |
| 7 | INT：`611b3dad`（A_log dtype） | `611b3dad` | 失败集合与基线相同；`weight_update/` 等 510 passed |
| 8 | FIX-A：`apex/q38-skip-save-state`（从未 step 过的优化器在存档前补状态） | `2281850b` | 失败集合与基线相同；自带测试 72 passed |

## CPU 测试

从合入 1 起，CPU 测试在官方镜像里跑（`miles/qwen38-ci-dh09`，只用 CPU；没有驱动库，用 `/usr/local/cuda/lib64/stubs/libcuda.so` 作 `libcuda.so.1`）。脚本和日志在 CPFS `/work/qwen38/ci/INT/`。判定标准：同 Pod 同设置下，失败集合与基线相同。rebase 后的基线是 `c62ea5d8`：up 3 failed / 2071 passed（`charts/miles_run/test_snapshot.py` 2 项、`test_workplace_launch_uses_the_configured_backend`），ext 2 failed（`test_launch_snapshot.py` 的 2 个参数化用例：镜像的 sglang-router 比快照多 3 个 `router_args` 字段）。旧分支的基线是 `65feb4a6`，失败集合相同。

下面是最初在 Mac 上的记录（macOS 环境，只作参考）。

环境：macOS arm64，Python 3.12.2，torch 2.15.0，transformers 5.19.0；Megatron `fdf82356` 源码目录、sglang 源码目录在 PYTHONPATH 上。Triton 不可用，用 pytest 插件把 `miles.kernels.quant.fp8_blockwise` 替换为调用即报错的 stub。这只验证 CPU 子集，不验证 CUDA ABI。

```bash
PYTHONPATH=<stub-plugin>:<megatron>:<sglang/python>:<sglang/radix_transfer_plugins> \
  python -m pytest -q -p no:cacheprovider -p fp8stub -rfE <tests> -n 4
```

| 测试集 | 路径 |
|---|---|
| PR 测试 | `tests/fast/backends/megatron_utils/test_dsv41_vision.py`、`test_qwen38_vision_sft.py`、`test_qwen38_vision_evaluation.py`、`test_qwen38_vision_scoring.py`、`tests/fast/utils/test_ray_utils.py`、`tests/fast/utils/external_utils/test_ray_job_driver_address.py`、`tests/fast/utils/hf_utils/test_config.py`、`tests/fast/launch_scripts/test_run_qwen3_8_next_vision_sft.py` |
| 上游相关 | `tests/fast/backends/megatron_utils`、`tests/fast/backends/training_utils/loss`、`tests/fast/backends/training_utils/loss_hub`、`tests/fast/miles_plugins/models/qwen3_8_next`、`tests/fast/models/test_kimi_k3_mbridge.py`、`tests/fast/utils/args`、`tests/fast/argument_snapshot`、`tests/fast/tools`、`tests/fast/charts`、`tests/fast/launch_scripts`、`tests/fast/utils/hf_utils`、`tests/fast/utils/test_ray_utils.py` |
| helm/launcher | `tests/fast/utils/external_utils`（不含 `test_colocate_pairing.py`） |

上游相关测试集在本机基线 `65feb4a6` 上有 63 个失败和 8 个错误，原因是 macOS 环境（无 `/proc`、无 triton、无 `nvidia_resiliency_ext`、inductor 配置）。判定标准：本分支的失败集合与基线相同，没有新增。

`49323ff5` 的结果（2026-10-30）：

| 测试集 | 结果 |
|---|---|
| PR 测试 | 145 passed、1 skipped，exit 0（原 147 项中 2 项宽松读回测试由严格契约测试替代） |
| 上游相关 | 63 failed、1755 passed、314 skipped、8 errors；失败集合与基线相同，新增 0 |
| helm/launcher | 878 passed、5 skipped，exit 0 |
| `tools/lint_config_attribute_access.py` | 0 违规 |
