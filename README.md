# LLM 推理优化学习项目

目标是从一个可运行的 Decoder 出发，逐步做出能处理多请求、可测量性能的小型推理引擎，再用实验理解 vLLM 的缓存、调度和执行优化。`01`～`07` 是单卡主线；量化与 CUDA 围绕测到的瓶颈开展，`08` 是多卡通信专题。

| 阶段 | 主要任务 | 当前进度 |
| --- | --- | --- |
| [01 Transformer](01_transformer/README.md) | 模型、训练、无缓存生成 | 已有实现、测试和小语料训练产物 |
| [02 KV cache](02_kv_cache/README.md) | 绝对位置编码的拼接/预分配缓存与基准 | 已有实现和单卡性能基线 |
| [03 RoPE](03_rope/README.md) | 预分配缓存上的旋转位置编码 | 已有实现 |
| [04 异长批量 decode](04_batched_decode/README.md) | 每请求独立缓存、合批执行与验证 | 已有正确性测试和单卡性能基线 |
| [05 调度](05_scheduler/README.md) | 请求状态、continuous batching、token 预算 | `scheduler.py` 编写中 |
| [06 分页缓存](06_paged_cache/README.md) | 物理块、block table、共享与回收 | 仅有阶段说明 |
| [07 推理引擎](07_inference_engine/README.md) | 组装执行循环、性能测量、单卡优化 | 仅有阶段说明 |

后续专题：[推理优化实验（SDPA、量化、CUDA）](docs/OPTIMIZATION.md)；[08 多 GPU 与 NCCL](08_distributed_nccl/README.md)。它们服务于推理引擎的实际问题，不要求现在并行推进。

**当前任务**：进入 [05 调度](05_scheduler/README.md)，实现请求在不同时间进入与结束。不同长度请求的批量 decode 已有[正确性测试](04_batched_decode/test_decode_batch.py)和[性能基线](docs/EXPERIMENT_LOG.md)；具体顺序与过关标准见[学习计划](docs/LEARNING_PLAN.md)。

对照资料：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[vLLM 架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。阅读源码时固定一个提交版本，再查该版本对应的文档。
