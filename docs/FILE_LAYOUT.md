# 文件组织与代码保留记录

## 2026-11-20 本次整理

这次只更新、新增 Markdown 文档和两个专题目录，未改写、移动或删除 Python 代码、符号链接、数据、checkpoint，也未调整 Git 暂存区。检查发现的代码问题保留给后续手写，见[检查记录](REVIEW_2026-11-20.md)。

| 位置 | 职责 | 本次处理 |
| --- | --- | --- |
| `01–04` | 已有学习版本、测试和 benchmark | 保持源码与路径 |
| `05_scheduler/` | 调度学习版本 | 修正文档中的旧文件名与接口 |
| `06–08` | 分页缓存、引擎、多 GPU 专题 | 保留专题；更新 07 的 Runner 状态 |
| [09_quantization](../09_quantization/README.md) | 量化原理、权重/KV 实验 | 新增路线与验收要求 |
| [10_cuda_kernels](../10_cuda_kernels/README.md) | CUDA C++ / Triton 单算子实验 | 新增路线、文件组织和测量要求 |
| [llm_inference](../llm_inference/README.md) | 单套持续演进的推理系统 | 按已有代码更新模型、调度器与 Runner 状态 |
| `docs/` | 跨专题路线、架构、检查与实验 | 修复过期描述，补充本次记录 |

后续独立实验放对应专题目录；经过验证且实际接入系统的实现再进入 llm_inference。不要为每个专题复制完整模型。已有编号不重排，量化/CUDA 基础不依赖多 GPU。

量化和 LLM 相关算子目前继续放同一仓库。若以后 CUDA 学习扩展到图像处理、通用并行算法或独立发布的算子库，再另开仓库，并在这里保留集成说明和固定版本依赖；仅需不同依赖时可以另建环境，无需拆仓库。

## 已存在的兼容路径与重复版本

```text
04_batched_decode/decode_batch.py → batch_prefill_decode.py
```

这是已有相对符号链接，供 04 的测试与 benchmark 导入；本次保持原样。复制项目时需要保留符号链接。

当前 `05_scheduler/schedule.py` 与 `llm_inference/scheduler.py` 除文件头外一致。主线后续只在 llm_inference 演进，专题版本用于对照；本次没有合并两份代码。

## 历史迁移说明

旧版文档记录过以下迁移；它描述此前的整理，不表示本次又执行了移动：

| 历史来源 | 位置 / 当前情况 |
| --- | --- |
| `04_batched_decode/batch_prefill_decode_optimized.py` | 主线 `llm_inference/model.py` |
| `05_scheduler/scheduler.py` | 主线 `llm_inference/scheduler.py`，后续已继续编辑 |
| `05_scheduler/tmp.py` | 旧文档记为 `scheduler_callbacks_draft.py`，该目标现在不存在 |

## 两份校验记录的区别

[CODE_PRESERVATION.sha256](CODE_PRESERVATION.sha256) 是之前整理留下的 13 条历史记录，本次没有覆盖。当前主线 scheduler 已变更，旧回调草稿路径已不存在，直接校验旧清单会出现失败；其余 11 条通过。这与后续继续手写代码相容，不能用旧清单判断本次是否改了代码。

[CODE_SNAPSHOT_2026-11-20.sha256](CODE_SNAPSHOT_2026-11-20.sha256) 是本次开始整理时的 Python 文件快照，包括旧模块名符号链接指向的内容。整理后用它检查代码内容没有改变：

```bash
# 在项目根目录执行
sha256sum -c docs/CODE_SNAPSHOT_2026-11-20.sha256
```

清单记录内容哈希；符号链接的目标另外核对。你之后继续写代码会自然改变哈希，不需要为了通过历史快照而还原代码。文件校验也不代替功能测试。
