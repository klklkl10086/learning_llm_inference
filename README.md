# LLM 推理优化学习项目

围绕自己写的 Transformer，逐步理解模型计算、KV cache、调度、量化和 GPU 算子，并接成可以验证、测量的推理系统。

**量化和 CUDA 算子继续放在这个仓库，各开专题子目录。** 它们可以复用现有模型、缓存、测试和性能基线；现阶段无需另开仓库。独立实验放专题目录，验证后需要集成的实现再进入 `llm_inference/`。

## 目录导航

| 目录 | 内容 | 当前状态 |
| --- | --- | --- |
| [01 Transformer](01_transformer/README.md) | 无缓存模型、训练、小语料与测试 | 已有实现与训练产物 |
| [02 KV cache](02_kv_cache/README.md) | 动态拼接、预分配缓存和 benchmark | 已有实现 |
| [03 RoPE](03_rope/README.md) | RoPE 与单请求缓存 | 已有实现 |
| [04 批量执行](04_batched_decode/README.md) | 异长 prefill/decode、对照模型、测试与 benchmark | 已有实现 |
| [05 调度](05_scheduler/README.md) | 请求生命周期、预算与调度策略 | 有草稿，更新路径待修正 |
| [06 分页缓存](06_paged_cache/README.md) | 块分配、block table、回收与共享 | 目前只有学习说明 |
| [07 推理引擎](07_inference_engine/README.md) | ModelRunner、Engine、测量 | Runner 为骨架，Engine 尚未创建 |
| [08 多 GPU](08_distributed_nccl/README.md) | NCCL 与张量并行 | 目前只有学习说明 |
| [09 量化](09_quantization/README.md) | 基础数值实验、权重量化、KV 量化与后端 | 新增学习路线，尚无实现 |
| [10 CUDA 算子](10_cuda_kernels/README.md) | CUDA C++、Triton、归约、融合与性能分析 | 新增学习路线，尚无实现 |
| [llm_inference](llm_inference/README.md) | 持续演进的单套推理系统实现 | 模型可单独使用，系统尚未闭环 |
| [docs](docs/LEARNING_PLAN.md) | 路线、架构、实验记录和检查结果 | 统一维护跨专题说明 |

编号是专题标识，不要求按 `01 → 10` 串行学完。保留 01–04 的演进版本和对照代码，后续系统功能围绕同一份 `llm_inference/` 实现。

## 现在怎么学

主线先形成一个可运行闭环：

```text
读懂模型契约 → 完成 ModelRunner → 修正并验证 Scheduler.update
→ Engine.step() → 多请求动态加入与退出
→ 分块 prefill / 分页缓存 → 按测量结果接入优化 → 多 GPU
```

同时可以开始两条小实验支线，无需等待分页缓存或多 GPU：

- **量化**：张量 INT8 量化/还原 → 单个 Linear 的权重量化 → 分组与 INT4 打包 → KV cache 量化。
- **算子**：vector add → reduction → softmax / LayerNorm → RoPE → 与实际模型瓶颈对应的融合。

建议当前主要精力用于接通最小引擎，每完成一个可运行步骤，再穿插一个量化或算子实验。基础实验可以提前做；声称系统加速前需要端到端基线。

组件边界见[系统设计](docs/ARCHITECTURE.md)，分步任务见[学习路线](docs/LEARNING_PLAN.md)，本次发现的问题见[检查记录](docs/REVIEW_2026-11-20.md)。

## 验证与实验

在项目根目录执行，使用现有环境：

```bash
conda activate kvcache
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s 01_transformer -p 'test_transformer.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s 04_batched_decode -p 'test_decode_batch.py' -v
```

2026-11-20：以上 5 + 8 个 CPU 测试通过；另用相同权重完成主线模型与 04 完整重算的异长 prefill、多轮换序 decode 对照。已有测试仍主要覆盖 01 和 04，不能据此认定调度器和引擎已可运行。细节见[检查记录](docs/REVIEW_2026-11-20.md)。

04 的 `decode_batch.py` 是指向 `batch_prefill_decode.py` 的符号链接，供现有测试和 benchmark 导入。01–02 使用绝对位置编码，03 及后续使用 RoPE；数值对照使用结构匹配的相同权重。

实际性能数据保存在[实验记录](docs/EXPERIMENT_LOG.md)。新实验的选择与测量口径见[优化专题](docs/OPTIMIZATION.md)。目录整理与代码校验说明见[整理记录](docs/FILE_LAYOUT.md)。
