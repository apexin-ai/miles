# apex/qwen38-text-sft 分支说明

本分支是 Qwen3.8-Flash-Next 文本 SFT 的集成分支，只在 `apexin-ai/miles` 上维护，不合回 radixark main。

## 基线

| 项 | 值 |
|---|---|
| miles 基线 | `65feb4a616bd935890cdac6ca06b1dca5cee28dd`（radixark main，router #4321） |
| 镜像 | `docker.io/radixark/miles@sha256:6038e614942f40a4681dff0f02ab7a8239412247da084db678811be9f50fac88`（tag `dev-202610300556`）。镜像内的 miles 就是上面的基线 |
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

## 已知限制

- 视觉路径只有 CPU 契约测试，没有 GPU 验证。最终存档读回检查是 model-only 读回，不证明续训（optimizer、scheduler、RNG、数据游标）。
- 读回检查失败时只恢复被人为扰动的切片再抛错，不是完整的内存回滚；失败后训练进程应退出。多 rank 下一个 rank 失败、其他 rank 停在 barrier 的情形没有测试（CRV P2-INT-1）。
- PR #6 没有修改 QSA indexer 的 `score_blocks`；已由合入的 `apex/q38-indexer-tiled` 和 `apex/q38-qsa-bitmap-packed` 处理。
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

## CPU 测试

从合入 1 起，CPU 测试在官方镜像里跑（`miles/qwen38-ci-dh09`，只用 CPU；没有驱动库，用 `/usr/local/cuda/lib64/stubs/libcuda.so` 作 `libcuda.so.1`）。脚本和日志在 CPFS `/work/qwen38/ci/INT/`。判定标准：同 Pod 同设置下，失败集合与基线 `65feb4a6` 相同。基线：上游相关 3 failed（`charts/miles_run/test_snapshot.py` 2 项、`test_workplace_launch_uses_the_configured_backend`），helm/launcher 2 failed（镜像的 sglang-router 比快照多 3 个 `router_args` 字段）。

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
