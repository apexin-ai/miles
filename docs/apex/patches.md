# 当前有效补丁清单（apex 主线）

本文件是 BRANCH-PR 方案第 5.3.2 节第 6 条要求的清单：**下一次换上游基线重建候选时，只按本清单重放，不重放 `U..apex/main` 里的全部历史提交。** 每一项写来源、基线、在当前候选里的提交、状态。历史上被取代的提交列在最后一节，重建时不要重放。

- 当前候选：`apex/qwen38-text-sft-rebase`（候选 2）。上游基线 U = radixark/miles main **`c62ea5d8c2851f63a21fcb42296fdaa6910bce69`**。
- 本清单覆盖 `c62ea5d8..` 候选 2 的全部 117 个提交（加上之后的文档 / 脚本提交）。
- 「提交」一列写的是**候选 2 里的 SHA**（短 SHA）；`apex/pr/*` 分支的 SHA 写在「来源」里。
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
| #4133 | `02cda78b` | `02cda78b` → `a3fbc478`（**混有本地适配**：tiled scorer 直接喂 block id、mRoPE、thd / sbhd 分开） | `91f46c43`：monitor 读 block id；`0bd5c396`：TP8 fill kernel 修复（M-030，`apex/pr/qsa-block-index-fill-bits`，GPU 待确认） | 上游 open |
| #2796 | `f01c353a` | `f01c353a` → `777aff66` | 无（生产代码与我们旧 `09c9393a` 相同，只剩这一份） | 上游 open |
| #4471 | `b45e6532` | `b45e6532` → `4765b582` | `e6cca277`：Qwen3.8 模型级 CP 防呆的错误文字 | 上游 open |
| #3964（观察器结构） | `e320b1a0` | `0d9e23e2` 手工移植 → `3bff1fa1`（作者 Junwei Zhou，`Ported-from` + `Co-authored-by: rasoolfa`） | `ee1d283a`：严格契约测试（训练过的存档缺 scheduler 状态就拒绝，与上游 #4530 契约相反） | 上游 open |
| #3691 | `f4d93f1d` | `d9ffa024` → `5bbb3395` | `aaa82af2`：rank0 覆盖检查的异常先广播再统一抛出 | 上游 open |

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
| `apex/main` 准备脚本 | `docs/apex/promote-apex-main.sh` | apex 内部流程 |

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

## 维护

- 每次往候选里加修复（第 5.3.2 节第 2 条），同时在本文件加一行；被取代的旧提交移到最后一节。
- 上游合入某个 L1 / L2 项后，按第 5.3.3 节逐个行为判断，只去掉被覆盖的部分，并在这里改状态。
