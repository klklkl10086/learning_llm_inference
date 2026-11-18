# LLM 推理优化学习项目

目标是从一个可运行的 Decoder 出发，逐步做出能处理多请求、可测量性能的小型推理引擎，再用实验理解 vLLM 的缓存、调度和执行优化。先完成 `01`～`05` 的单卡主线；量化与 CUDA 围绕测到的瓶颈开展，NCCL 放在多卡专题。

| 阶段 | 主要任务 | 当前进度 |
| --- | --- | --- |
| [01 Transformer](01_transformer/README.md) | 模型、训练、无缓存生成 | 已有实现、测试和小语料训练产物 |
| [02 KV cache](02_kv_cache/README.md) | 动态/预分配缓存、RoPE、不同长度请求的批量 decode | 前三类实现已存在；`decode_batch.py` 仍是草稿，未验收 |
| [03 调度](03_scheduler/README.md) | 请求状态、continuous batching、token 预算 | 仅有阶段说明 |
| [04 分页缓存](04_paged_cache/README.md) | 物理块、block table、共享与回收 | 仅有阶段说明 |
| [05 推理引擎](05_inference_engine/README.md) | 组装执行循环、性能测量、单卡优化 | 仅有阶段说明 |

后续专题：[推理优化实验（SDPA、量化、CUDA）](docs/OPTIMIZATION.md)；[多 GPU 与 NCCL](06_distributed_nccl/README.md)。它们服务于推理引擎的实际问题，不要求现在并行推进。

**当前任务**：在 [02 KV cache](02_kv_cache/README.md) 完成不同长度请求的批量 decode，并与逐请求执行比较 logits。完成后再进入调度。具体顺序、过关标准和文件地图见[学习计划](docs/LEARNING_PLAN.md)；实验结果填入[实验记录](docs/EXPERIMENT_LOG.md)。

对照资料：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[vLLM 架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。阅读源码时固定一个提交版本，再查该版本对应的文档。
